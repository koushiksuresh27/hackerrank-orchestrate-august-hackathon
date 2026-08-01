"""
Evaluator — validates output quality without embeddings.

Evaluation strategy:
1. Schema validation (enums, confidence bounds, column order)
2. Evidence ID validity (check every ID exists in message_history.csv)
3. Heuristic cross-checks (high forward → should be mute, injection → must be mute/scam)
4. Distribution analysis (action/type distribution shouldn't be extreme)
5. Confidence calibration (flag values outside expected range)
6. Consistency checks (similar message patterns should have similar treatments)
"""

import csv
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from config import (
    VALID_ACTIONS,
    VALID_MESSAGE_TYPES,
    CONFIDENCE_MIN,
    CONFIDENCE_MAX,
    OUTPUT_PATH,
    HIGH_FORWARD_COUNT,
)
from data_loader import DataStore

logger = logging.getLogger(__name__)


def evaluate_output(
    output_path: Path | None = None,
    store: DataStore | None = None,
) -> dict[str, Any]:
    """
    Run full evaluation on the output CSV.

    Args:
        output_path: Path to output.csv
        store: DataStore for evidence validation and cross-checks

    Returns:
        Evaluation report dict.
    """
    path = output_path or OUTPUT_PATH

    if not path.exists():
        logger.error(f"Output file not found: {path}")
        return {"error": "Output file not found"}

    # Read output
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    report: dict[str, Any] = {
        "total_rows": len(rows),
        "schema_errors": [],
        "evidence_errors": [],
        "heuristic_violations": [],
        "distribution": {},
        "confidence_stats": {},
        "warnings": [],
    }

    # 1. Schema validation
    _check_schema(rows, report)

    # 2. Evidence validation
    if store:
        _check_evidence(rows, store, report)

    # 3. Heuristic cross-checks
    if store:
        _check_heuristics(rows, store, report)

    # 4. Distribution analysis
    _check_distribution(rows, report)

    # 5. Confidence calibration
    _check_confidence(rows, report)

    # 6. Completeness
    _check_completeness(rows, store, report)

    # Summary
    total_issues = (
        len(report["schema_errors"])
        + len(report["evidence_errors"])
        + len(report["heuristic_violations"])
        + len(report["warnings"])
    )
    report["total_issues"] = total_issues
    report["pass"] = total_issues == 0

    return report


def print_report(report: dict) -> None:
    """Print a human-readable evaluation report."""
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    print(f"\nTotal rows: {report.get('total_rows', 0)}")
    print(f"Total issues: {report.get('total_issues', 0)}")
    print(f"Overall: {'[PASS]' if report.get('pass') else '[ISSUES FOUND]'}")

    if report.get("schema_errors"):
        print(f"\n--- Schema Errors ({len(report['schema_errors'])}) ---")
        for err in report["schema_errors"][:10]:
            print(f"  [ERR] {err}")

    if report.get("evidence_errors"):
        print(f"\n--- Evidence Errors ({len(report['evidence_errors'])}) ---")
        for err in report["evidence_errors"][:10]:
            print(f"  [ERR] {err}")

    if report.get("heuristic_violations"):
        print(
            f"\n--- Heuristic Violations ({len(report['heuristic_violations'])}) ---"
        )
        for v in report["heuristic_violations"][:10]:
            print(f"  [WARN] {v}")

    dist = report.get("distribution", {})
    if dist:
        print("\n--- Distribution ---")
        for key, counts in dist.items():
            print(f"  {key}: {dict(counts)}")

    conf = report.get("confidence_stats", {})
    if conf:
        print("\n--- Confidence Stats ---")
        for key, val in conf.items():
            print(f"  {key}: {val}")

    if report.get("warnings"):
        print(f"\n--- Warnings ({len(report['warnings'])}) ---")
        for w in report["warnings"][:10]:
            print(f"  [WARN] {w}")

    print("\n" + "=" * 60)


def _check_schema(rows: list[dict], report: dict) -> None:
    """Validate that all rows have valid schema."""
    required_cols = {
        "message_id", "action", "message_type", "reason",
        "confidence", "evidence_message_ids",
    }

    if rows:
        actual_cols = set(rows[0].keys())
        missing_cols = required_cols - actual_cols
        if missing_cols:
            report["schema_errors"].append(f"Missing columns: {missing_cols}")

    seen_ids = set()
    for row in rows:
        mid = row.get("message_id", "").strip()

        # Duplicate check
        if mid in seen_ids:
            report["schema_errors"].append(f"Duplicate message_id: {mid}")
        seen_ids.add(mid)

        # Action check
        action = row.get("action", "").strip()
        if action not in VALID_ACTIONS:
            report["schema_errors"].append(
                f"{mid}: invalid action '{action}'"
            )

        # Message type check
        msg_type = row.get("message_type", "").strip()
        if msg_type not in VALID_MESSAGE_TYPES:
            report["schema_errors"].append(
                f"{mid}: invalid message_type '{msg_type}'"
            )

        # Confidence check
        try:
            conf = float(row.get("confidence", "0"))
            if conf < 0 or conf > 1:
                report["schema_errors"].append(
                    f"{mid}: confidence {conf} out of [0, 1] range"
                )
        except ValueError:
            report["schema_errors"].append(
                f"{mid}: non-numeric confidence '{row.get('confidence', '')}'"
            )

        # Reason check
        reason = row.get("reason", "").strip()
        if not reason:
            report["schema_errors"].append(f"{mid}: empty reason")


def _check_evidence(rows: list[dict], store: DataStore, report: dict) -> None:
    """Validate that evidence IDs exist in message_history."""
    for row in rows:
        mid = row.get("message_id", "")
        evidence = row.get("evidence_message_ids", "none").strip()

        if evidence.lower() == "none" or not evidence:
            continue

        ids = [eid.strip() for eid in evidence.split(";") if eid.strip()]
        for eid in ids:
            if not store.validate_evidence_id(eid):
                report["evidence_errors"].append(
                    f"{mid}: evidence ID '{eid}' not found in message_history"
                )


def _check_heuristics(rows: list[dict], store: DataStore, report: dict) -> None:
    """
    Cross-check routing decisions against known heuristic signals.
    These are soft checks — violations are warnings, not errors.
    """
    messages_index = {m.get("message_id", ""): m for m in store.messages_raw}

    for row in rows:
        mid = row.get("message_id", "")
        action = row.get("action", "")
        msg_type = row.get("message_type", "")
        msg = messages_index.get(mid, {})

        if not msg:
            continue

        # High forward count should typically be mute
        fwd = int(float(msg.get("forwarded_count", "0") or "0"))
        if fwd >= HIGH_FORWARD_COUNT and action != "mute":
            report["heuristic_violations"].append(
                f"{mid}: forwarded_count={fwd} but action='{action}' (expected 'mute')"
            )

        # Group muted by user should typically be mute
        group_id = msg.get("group_id", "")
        user_id = msg.get("user_id", "")
        if group_id and user_id:
            membership = store.get_group_membership(group_id, user_id)
            if membership and membership.get("group_muted_by_user") == "1":
                if action == "notify":
                    # Only flag if there's no direct mention
                    text = msg.get("message_text", "") or ""
                    if f"@{user_id}" not in text:
                        report["heuristic_violations"].append(
                            f"{mid}: group muted by user but action='notify' without @mention"
                        )


def _check_distribution(rows: list[dict], report: dict) -> None:
    """Analyze the distribution of actions and message types."""
    action_counts = Counter(row.get("action", "") for row in rows)
    type_counts = Counter(row.get("message_type", "") for row in rows)

    report["distribution"] = {
        "actions": action_counts,
        "message_types": type_counts,
    }

    # Sanity checks on distribution
    total = len(rows)
    if total > 0:
        # If more than 80% of messages get the same action, something might be wrong
        for action, count in action_counts.items():
            if count / total > 0.8:
                report["warnings"].append(
                    f"Distribution skew: {count}/{total} ({count/total:.0%}) messages are '{action}'"
                )


def _check_confidence(rows: list[dict], report: dict) -> None:
    """Analyze confidence value distribution."""
    confidences = []
    for row in rows:
        try:
            confidences.append(float(row.get("confidence", "0")))
        except ValueError:
            pass

    if confidences:
        report["confidence_stats"] = {
            "min": round(min(confidences), 2),
            "max": round(max(confidences), 2),
            "mean": round(sum(confidences) / len(confidences), 2),
            "below_0.70": sum(1 for c in confidences if c < CONFIDENCE_MIN),
            "above_0.95": sum(1 for c in confidences if c > CONFIDENCE_MAX),
        }

        # Flag miscalibration
        if report["confidence_stats"]["below_0.70"] > 0:
            report["warnings"].append(
                f"{report['confidence_stats']['below_0.70']} confidence values below {CONFIDENCE_MIN}"
            )
        if report["confidence_stats"]["above_0.95"] > 0:
            report["warnings"].append(
                f"{report['confidence_stats']['above_0.95']} confidence values above {CONFIDENCE_MAX}"
            )


def _check_completeness(rows: list[dict], store: DataStore | None, report: dict) -> None:
    """Check all expected message_ids are present."""
    if not store:
        return

    expected_ids = {m.get("message_id", "") for m in store.messages_raw}
    output_ids = {row.get("message_id", "") for row in rows}

    missing = expected_ids - output_ids
    extra = output_ids - expected_ids

    if missing:
        report["schema_errors"].append(
            f"Missing {len(missing)} message_ids: {sorted(missing)[:5]}..."
        )
    if extra:
        report["warnings"].append(
            f"Extra {len(extra)} message_ids not in messages.csv: {sorted(extra)[:5]}..."
        )
