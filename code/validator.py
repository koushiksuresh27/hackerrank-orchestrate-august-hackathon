"""
Validator — post-LLM output validation and repair.

Enforces:
- action ∈ {notify, digest, mute}
- message_type ∈ allowed set
- confidence ∈ [0.70, 0.95]
- reason is a clean string (no CSV-breaking characters)
- evidence_message_ids format (semicolon-separated or "none")

Invalid fields are repaired or defaulted rather than rejected,
ensuring every message gets a valid output row.
"""

import logging
import re
from typing import Any

from config import (
    VALID_ACTIONS,
    VALID_MESSAGE_TYPES,
    CONFIDENCE_MIN,
    CONFIDENCE_MAX,
    DEFAULT_CONFIDENCE,
)

logger = logging.getLogger(__name__)


def validate_and_repair(
    decision: dict,
    message_id: str,
    store: Any = None,  # DataStore, optional for evidence validation
) -> dict:
    """
    Validate and repair an LLM routing decision.

    Args:
        decision: Raw decision dict from agent
        message_id: The message_id for logging
        store: Optional DataStore for evidence ID validation

    Returns:
        Cleaned, valid decision dict with all required fields.
    """
    if not decision or not isinstance(decision, dict):
        logger.error(f"[{message_id}] Null or invalid decision, using defaults")
        return _default_decision(message_id)

    cleaned = {}

    # 1. Validate action
    action = str(decision.get("action", "")).strip().lower()
    if action in VALID_ACTIONS:
        cleaned["action"] = action
    else:
        logger.warning(f"[{message_id}] Invalid action '{action}', defaulting to 'digest'")
        cleaned["action"] = "digest"

    # 2. Validate message_type
    msg_type = str(decision.get("message_type", "")).strip().lower()
    if msg_type in VALID_MESSAGE_TYPES:
        cleaned["message_type"] = msg_type
    else:
        # Try fuzzy matching
        matched = _fuzzy_match_type(msg_type)
        if matched:
            logger.warning(
                f"[{message_id}] Fuzzy-matched message_type '{msg_type}' → '{matched}'"
            )
            cleaned["message_type"] = matched
        else:
            logger.warning(
                f"[{message_id}] Invalid message_type '{msg_type}', defaulting to 'unknown'"
            )
            cleaned["message_type"] = "unknown"

    # 3. Validate confidence
    try:
        conf = float(decision.get("confidence", DEFAULT_CONFIDENCE))
        conf = max(CONFIDENCE_MIN, min(CONFIDENCE_MAX, conf))
        cleaned["confidence"] = round(conf, 2)
    except (ValueError, TypeError):
        logger.warning(f"[{message_id}] Invalid confidence, using default")
        cleaned["confidence"] = DEFAULT_CONFIDENCE

    # 4. Validate and clean reason
    reason = str(decision.get("reason", "")).strip()
    if not reason:
        reason = "Routing decision based on available context."
    reason = _clean_reason(reason)
    cleaned["reason"] = reason

    # 5. Validate evidence_message_ids
    evidence = str(decision.get("evidence_message_ids", "none")).strip()
    evidence = _validate_evidence(evidence, store)
    cleaned["evidence_message_ids"] = evidence

    return cleaned


def _default_decision(message_id: str) -> dict:
    """Return a safe default decision when validation completely fails."""
    return {
        "action": "digest",
        "message_type": "unknown",
        "reason": "Unable to determine routing. Defaulting to digest.",
        "confidence": DEFAULT_CONFIDENCE,
        "evidence_message_ids": "none",
    }


def _clean_reason(reason: str) -> str:
    """
    Clean the reason string to prevent CSV corruption.
    - Strips/replaces problematic characters
    - Truncates to a reasonable length
    - Ensures it's a single sentence
    """
    # Remove newlines and tabs
    reason = reason.replace("\n", " ").replace("\r", " ").replace("\t", " ")

    # Collapse multiple spaces
    reason = re.sub(r"\s+", " ", reason).strip()

    # Remove any quotes that could mess up CSV
    reason = reason.replace('"', "'")

    # Truncate to ~150 chars if too long
    if len(reason) > 150:
        reason = reason[:147] + "..."

    # Ensure it ends with a period
    if reason and not reason.endswith((".","!", "?")):
        reason += "."

    return reason


def _validate_evidence(evidence: str, store: Any = None) -> str:
    """
    Validate evidence_message_ids format and content.
    - Must be semicolon-separated message IDs or "none"
    - IDs should match message_history format (message_XXXX)
    """
    if not evidence or evidence.lower().strip() in ("none", "n/a", "null", ""):
        return "none"

    # Split and clean
    ids = [eid.strip() for eid in evidence.split(";") if eid.strip()]

    if not ids:
        return "none"

    valid_ids = []
    for eid in ids:
        # Check format: should start with "message_"
        if re.match(r"^message_\d+$", eid):
            # Optionally verify against store
            if store and hasattr(store, "validate_evidence_id"):
                if store.validate_evidence_id(eid):
                    valid_ids.append(eid)
                else:
                    logger.debug(f"Evidence ID {eid} not found in message_history")
            else:
                valid_ids.append(eid)
        else:
            logger.debug(f"Invalid evidence ID format: {eid}")

    return ";".join(valid_ids) if valid_ids else "none"


def _fuzzy_match_type(msg_type: str) -> str | None:
    """Try to fuzzy-match a message type to valid options."""
    # Common LLM mistakes
    fuzzy_map = {
        "business": "business_update",
        "business_promo": "promotion",
        "promo": "promotion",
        "marketing": "promotion",
        "chain": "forward",
        "forwarded": "forward",
        "phishing": "scam",
        "fraud": "scam",
        "junk": "spam",
        "safety": "urgent",
        "emergency": "urgent",
        "meeting": "event",
        "appointment": "event",
        "schedule": "event",
        "bill": "payment",
        "transaction": "payment",
        "invoice": "payment",
        "hello": "greeting",
        "hi": "greeting",
        "good_morning": "greeting",
    }
    return fuzzy_map.get(msg_type)
