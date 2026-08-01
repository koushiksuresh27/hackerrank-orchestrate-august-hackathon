"""
Evidence Selector — retrieves relevant historical message IDs as evidence.

For each incoming message, this module searches message_history for past messages
that support the routing decision. Evidence IDs use the message_history ID format
(e.g., "message_0001"), NOT the messages.csv format (e.g., "msg_001").

Selection strategy:
1. Find messages from the same sender to the same user
2. Find messages in the same group (if group message)
3. Find messages from the same business (if business message)
4. Rank by relevance (same sender > same group > same business)
5. Return top matches as semicolon-separated IDs
"""

import logging
from typing import Any

from data_loader import DataStore

logger = logging.getLogger(__name__)

# Maximum number of evidence IDs to return
MAX_EVIDENCE = 3


def select_evidence(
    message: dict,
    context: dict,
    store: DataStore,
) -> str:
    """
    Select the most relevant historical message IDs as evidence for a routing decision.

    Args:
        message: Raw message row from messages.csv
        context: The context dict built by context_builder
        store: The DataStore instance

    Returns:
        Semicolon-separated evidence message IDs, or "none" if no relevant evidence.
    """
    user_id = message.get("user_id", "")
    sender_user_id = message.get("sender_user_id", "")
    group_id = message.get("group_id", "")
    business_id = message.get("business_id", "")
    conversation_type = message.get("conversation_type", "")

    candidates: list[tuple[str, float]] = []  # (message_id, relevance_score)

    user_history = store.get_user_history(user_id)

    # Strategy 1: Same sender → same user (strongest signal)
    if sender_user_id:
        for m in user_history:
            if m.get("sender_user_id") == sender_user_id:
                score = _compute_relevance(m, message, store, user_id, weight=3.0)
                msg_id = m.get("message_id", "")
                if msg_id:
                    candidates.append((msg_id, score))

    # Strategy 2: Same business → same user
    if conversation_type == "business" and business_id:
        for m in user_history:
            if m.get("business_id") == business_id:
                score = _compute_relevance(m, message, store, user_id, weight=2.5)
                msg_id = m.get("message_id", "")
                if msg_id and not any(c[0] == msg_id for c in candidates):
                    candidates.append((msg_id, score))

    # Strategy 3: Same group → same user (weaker, but shows group engagement)
    if conversation_type == "group" and group_id:
        for m in user_history:
            if m.get("group_id") == group_id:
                score = _compute_relevance(m, message, store, user_id, weight=1.5)
                msg_id = m.get("message_id", "")
                if msg_id and not any(c[0] == msg_id for c in candidates):
                    candidates.append((msg_id, score))

    if not candidates:
        return "none"

    # Sort by relevance score descending, take top N
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_ids = [c[0] for c in candidates[:MAX_EVIDENCE]]

    return ";".join(top_ids)


def _compute_relevance(
    hist_msg: dict,
    incoming_msg: dict,
    store: DataStore,
    user_id: str,
    weight: float = 1.0,
) -> float:
    """
    Compute a relevance score for a historical message as evidence.

    Factors:
    - Weight multiplier (same sender > same business > same group)
    - Recency bonus (newer messages are more relevant)
    - User interaction signal (opened/replied vs dismissed/reported)
    - Content similarity (forwarded count pattern)
    """
    score = weight

    # Recency: messages closer in time are more relevant
    hist_date = hist_msg.get("created_at", "")
    if hist_date:
        # Simple heuristic: later dates get higher scores
        try:
            date_part = hist_date.split(" ")[0]
            year, month, day = date_part.split("-")
            # Normalize to a 0-1 range relative to 2026
            days_from_epoch = int(year) * 365 + int(month) * 30 + int(day)
            score += days_from_epoch * 0.0001  # small but breaks ties
        except (ValueError, IndexError):
            pass

    # User interaction signals from message_events
    msg_id = hist_msg.get("message_id", "")
    if msg_id:
        events = store.get_message_events(msg_id)
        for evt in events:
            if evt.get("user_id") == user_id:
                if evt.get("message_opened") == "1":
                    score += 0.5
                if evt.get("message_replied") == "1":
                    score += 1.0
                if evt.get("notification_dismissed") == "1":
                    score += 0.3  # still relevant evidence
                if evt.get("message_reported") == "1":
                    score += 1.5  # strong signal
                if evt.get("muted_after_message") == "1":
                    score += 1.0

    # Forward count pattern match
    hist_fwd = _safe_int(hist_msg.get("forwarded_count", "0"))
    inc_fwd = _safe_int(incoming_msg.get("forwarded_count", "0"))
    if hist_fwd > 0 and inc_fwd > 0:
        score += 0.5  # both are forwards → pattern evidence

    return score


def get_evidence_context(evidence_ids: str, store: DataStore) -> str:
    """
    Build a brief context string about the evidence messages for the LLM prompt.

    Args:
        evidence_ids: Semicolon-separated evidence IDs or "none"
        store: DataStore instance

    Returns:
        Formatted evidence summary for the prompt.
    """
    if evidence_ids == "none" or not evidence_ids:
        return "=== EVIDENCE ===\nNo relevant historical evidence found."

    parts = ["=== EVIDENCE ==="]
    ids = [eid.strip() for eid in evidence_ids.split(";") if eid.strip()]

    for eid in ids:
        hist = store.get_history_message(eid)
        if hist:
            text_preview = (hist.get("message_text", "") or "")[:100]
            if len(hist.get("message_text", "") or "") > 100:
                text_preview += "..."

            events = store.get_message_events(eid)
            event_summary = ""
            for evt in events:
                actions = []
                if evt.get("message_opened") == "1":
                    actions.append("opened")
                if evt.get("message_replied") == "1":
                    actions.append("replied")
                if evt.get("notification_dismissed") == "1":
                    actions.append("dismissed")
                if evt.get("message_reported") == "1":
                    actions.append("reported")
                if evt.get("muted_after_message") == "1":
                    actions.append("muted_after")
                if actions:
                    event_summary = f" → User: {', '.join(actions)}"

            parts.append(
                f"[{eid}] From {hist.get('sender_user_id', '') or hist.get('business_id', '')} "
                f"({hist.get('conversation_type', '')}): "
                f"\"{text_preview}\"{event_summary}"
            )
        else:
            parts.append(f"[{eid}] (not found in history)")

    return "\n".join(parts)


def _safe_int(value) -> int:
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return 0
