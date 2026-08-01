"""
Signal Extractor — pre-LLM heuristic signals.

Extracts deterministic signals from message and context data before the LLM call.
These signals serve two purposes:
1. Pre-filter obvious cases (prompt injection, high-forward spam) to avoid wasting API calls.
2. Augment the LLM prompt with structured signal flags so it can reason more accurately.

Returns a SignalResult with flags and an optional pre-decision for clear-cut cases.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from config import HIGH_FORWARD_COUNT, INJECTION_PATTERNS

logger = logging.getLogger(__name__)

# Compiled injection patterns
_INJECTION_REGEXES = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


@dataclass
class SignalResult:
    """Pre-LLM signal extraction result."""

    # Flags
    is_injection: bool = False
    is_high_forward: bool = False
    is_group_muted: bool = False
    is_domain_mismatch: bool = False
    is_unverified_business: bool = False
    is_opted_out: bool = False
    is_sender_unknown: bool = False
    is_during_dnd: bool = False
    has_direct_mention: bool = False
    sender_has_high_dismiss_rate: bool = False
    sender_has_report_history: bool = False
    is_first_contact: bool = False

    # Pre-decision (if set, skip LLM call)
    pre_decision: dict | None = None

    # Signal summary for prompt
    signal_flags: list[str] = field(default_factory=list)

    def should_skip_llm(self) -> bool:
        """Whether signals are strong enough to skip the LLM call entirely."""
        return self.pre_decision is not None


def extract_signals(message: dict, context: dict) -> SignalResult:
    """
    Extract heuristic signals from a message and its context.

    Args:
        message: Raw message row from messages.csv
        context: The context dict built by context_builder

    Returns:
        SignalResult with flags, optional pre-decision, and signal summary
    """
    result = SignalResult()
    msg = context.get("message", {})
    text = msg.get("message_text", "") or ""
    user_id = msg.get("user_id", "")
    forwarded_count = msg.get("forwarded_count", 0)

    # 1. Prompt injection detection
    if _detect_injection(text):
        result.is_injection = True
        result.signal_flags.append("PROMPT_INJECTION_DETECTED")
        result.pre_decision = {
            "action": "mute",
            "message_type": "scam",
            "reason": "The message tries to instruct the router, but the routing decision should be based on the actual content and risk.",
            "confidence": 0.88,
        }
        logger.warning(f"Prompt injection detected in {msg.get('message_id', '')}")
        return result

    # 2. High forward count → spam/chain signal
    if forwarded_count >= HIGH_FORWARD_COUNT:
        result.is_high_forward = True
        result.signal_flags.append(f"HIGH_FORWARD_COUNT ({forwarded_count})")

    # 3. Group muted by user
    membership = context.get("user_group_membership")
    if membership and membership.get("group_muted_by_user"):
        result.is_group_muted = True
        result.signal_flags.append("GROUP_MUTED_BY_USER")

    # 4. Business domain mismatch
    business = context.get("business")
    if business:
        if business.get("domain_mismatch"):
            result.is_domain_mismatch = True
            result.signal_flags.append("DOMAIN_MISMATCH")

        if not business.get("verified"):
            result.is_unverified_business = True
            result.signal_flags.append("UNVERIFIED_BUSINESS")

    # 5. User opted out of promotions
    ubr = context.get("user_business_relation")
    if ubr:
        if not ubr.get("allows_promotions") and ubr.get("promotions_opted_out_at"):
            result.is_opted_out = True
            result.signal_flags.append("USER_OPTED_OUT_PROMOTIONS")

    # 6. Unknown sender (no history)
    sender_history = context.get("sender_history")
    if sender_history is None:
        result.is_first_contact = True
        result.signal_flags.append("FIRST_CONTACT_FROM_SENDER")
    elif sender_history:
        if sender_history.get("dismiss_rate", 0) > 0.6:
            result.sender_has_high_dismiss_rate = True
            result.signal_flags.append("SENDER_HIGH_DISMISS_RATE")

        if sender_history.get("user_reported", 0) > 0:
            result.sender_has_report_history = True
            result.signal_flags.append("SENDER_HAS_REPORTS")

    # 7. Direct mention detection (@ mention of the user)
    if user_id and f"@{user_id}" in text:
        result.has_direct_mention = True
        result.signal_flags.append("DIRECT_MENTION")

    # 8. DND window check
    user = context.get("user")
    if user and msg.get("created_at"):
        dnd = user.get("do_not_disturb_window", "")
        if dnd and _is_during_dnd(msg["created_at"], dnd):
            result.is_during_dnd = True
            result.signal_flags.append("DURING_DND_WINDOW")

    return result


def format_signals_for_prompt(result: SignalResult) -> str:
    """Format signal flags as a section for the LLM prompt."""
    if not result.signal_flags:
        return "=== PRE-ANALYSIS SIGNALS ===\nNo significant signals detected."

    lines = ["=== PRE-ANALYSIS SIGNALS ==="]
    for flag in result.signal_flags:
        lines.append(f"[!] {flag}")
    return "\n".join(lines)


def _detect_injection(text: str) -> bool:
    """Check message text for prompt injection patterns."""
    if not text:
        return False
    for pattern in _INJECTION_REGEXES:
        if pattern.search(text):
            return True
    return False


def _is_during_dnd(created_at: str, dnd_window: str) -> bool:
    """
    Check if a message timestamp falls within the user's DND window.
    DND format: "22:00-07:00" (can wrap past midnight).
    Created_at format: "2026-07-31 14:16"
    """
    try:
        parts = dnd_window.split("-")
        if len(parts) != 2:
            return False
        start_h, start_m = map(int, parts[0].strip().split(":"))
        end_h, end_m = map(int, parts[1].strip().split(":"))

        # Extract hour:minute from timestamp
        time_part = created_at.strip().split(" ")[-1]
        msg_h, msg_m = map(int, time_part.split(":"))

        start_minutes = start_h * 60 + start_m
        end_minutes = end_h * 60 + end_m
        msg_minutes = msg_h * 60 + msg_m

        if start_minutes <= end_minutes:
            # Same-day window (e.g., 08:00-17:00)
            return start_minutes <= msg_minutes <= end_minutes
        else:
            # Overnight window (e.g., 22:00-07:00)
            return msg_minutes >= start_minutes or msg_minutes <= end_minutes
    except (ValueError, IndexError):
        return False
