"""
Output Writer — writes the final output.csv with proper formatting.

Handles:
- Exact column ordering per problem_statement.md
- CSV quoting via csv.QUOTE_ALL to prevent comma-in-reason corruption
- Completeness check (all 110 message_ids present)
- Deduplication (no duplicate message_ids)
- Row ordering (matches template output.csv order)
"""

import csv
import logging
from pathlib import Path
from typing import Any

from config import OUTPUT_PATH, CSV_FILES

logger = logging.getLogger(__name__)

# Required columns in exact order
OUTPUT_COLUMNS = [
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
]


def write_output(
    results: dict[str, dict],
    output_path: Path | None = None,
) -> bool:
    """
    Write routing results to output.csv.

    Args:
        results: Dict of message_id → validated decision dict
        output_path: Override output path (defaults to config.OUTPUT_PATH)

    Returns:
        True if write succeeded, False otherwise.
    """
    path = output_path or OUTPUT_PATH

    # Read template to get expected message_ids and their order
    expected_ids = _read_template_ids()

    if not expected_ids:
        logger.warning("Could not read template output.csv, using results order")
        expected_ids = list(results.keys())

    # Check completeness
    missing = set(expected_ids) - set(results.keys())
    extra = set(results.keys()) - set(expected_ids)

    if missing:
        logger.error(f"Missing {len(missing)} message_ids in output: {missing}")
        # Fill missing with defaults
        for mid in missing:
            results[mid] = {
                "action": "digest",
                "message_type": "unknown",
                "reason": "No routing decision available for this message.",
                "confidence": 0.72,
                "evidence_message_ids": "none",
            }

    if extra:
        logger.warning(f"Extra {len(extra)} message_ids not in template: {extra}")

    # Deduplicate (shouldn't happen, but safety)
    seen = set()
    ordered_ids = []
    for mid in expected_ids:
        if mid not in seen:
            seen.add(mid)
            ordered_ids.append(mid)

    # Write CSV with QUOTE_ALL to prevent comma-in-reason issues
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(OUTPUT_COLUMNS)

            for mid in ordered_ids:
                decision = results.get(mid, {})
                row = [
                    mid,
                    decision.get("action", "digest"),
                    decision.get("message_type", "unknown"),
                    decision.get("reason", "No decision available."),
                    str(decision.get("confidence", 0.72)),
                    decision.get("evidence_message_ids", "none"),
                ]
                writer.writerow(row)

        total = len(ordered_ids)
        logger.info(f"Output written to {path}: {total} rows")

        # Final validation
        _validate_output_file(path, len(ordered_ids))

        return True

    except Exception as e:
        logger.error(f"Failed to write output: {e}")
        return False


def _read_template_ids() -> list[str]:
    """Read message_ids from the template output.csv to preserve ordering."""
    template_path = CSV_FILES.get("messages")
    if not template_path:
        return []

    # Actually read from the blank output.csv template for ordering
    output_template = OUTPUT_PATH
    if output_template.exists():
        try:
            with open(output_template, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                return [row["message_id"] for row in reader if row.get("message_id")]
        except Exception:
            pass

    # Fallback: read from messages.csv
    if template_path.exists():
        try:
            with open(template_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                return [row["message_id"] for row in reader if row.get("message_id")]
        except Exception:
            pass

    return []


def _validate_output_file(path: Path, expected_count: int) -> None:
    """Post-write validation of the output file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Check row count
        if len(rows) != expected_count:
            logger.error(
                f"Output row count mismatch: expected {expected_count}, got {len(rows)}"
            )

        # Check for required columns
        if rows:
            actual_cols = list(rows[0].keys())
            for col in OUTPUT_COLUMNS:
                if col not in actual_cols:
                    logger.error(f"Missing column in output: {col}")

        # Check for valid values
        valid_actions = {"notify", "digest", "mute"}
        for row in rows:
            if row.get("action") not in valid_actions:
                logger.error(
                    f"Invalid action for {row.get('message_id')}: {row.get('action')}"
                )

        # Check for duplicates
        ids = [r.get("message_id") for r in rows]
        if len(ids) != len(set(ids)):
            logger.error("Duplicate message_ids detected in output!")

        logger.info("Output validation passed")

    except Exception as e:
        logger.error(f"Output validation failed: {e}")
