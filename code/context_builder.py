"""
Context Builder — assembles per-message context for LLM routing.

For each incoming message, this module gathers:
- User profile and notification behavior
- Group info and user's membership/mute status
- Business account details and user-business relationship
- Relevant message history (sender patterns, group patterns)
- User engagement events (opens, dismissals, reports)
- Daily notification load
- Media metadata (image/voice references)

The output is a structured context dict that gets serialized into the LLM prompt.
"""

import logging
from typing import Any

from data_loader import DataStore

logger = logging.getLogger(__name__)


def build_context(message: dict, store: DataStore) -> dict[str, Any]:
    """
    Build a comprehensive context dict for a single incoming message.

    Args:
        message: A single row from messages.csv
        store: The loaded DataStore instance

    Returns:
        A dict with all relevant context for routing this message.
    """
    user_id = message.get("user_id", "")
    group_id = message.get("group_id", "")
    business_id = message.get("business_id", "")
    sender_user_id = message.get("sender_user_id", "")
    conversation_type = message.get("conversation_type", "")
    media_type = message.get("media_type", "")
    media_id = message.get("media_id", "")

    context: dict[str, Any] = {
        "message": {
            "message_id": message.get("message_id", ""),
            "user_id": user_id,
            "conversation_type": conversation_type,
            "group_id": group_id,
            "business_id": business_id,
            "sender_user_id": sender_user_id,
            "created_at": message.get("created_at", ""),
            "message_text": message.get("message_text", ""),
            "media_type": media_type,
            "media_id": media_id,
            "forwarded_count": _safe_int(message.get("forwarded_count", "0")),
        },
    }

    # --- User profile ---
    user = store.get_user(user_id)
    if user:
        context["user"] = {
            "user_id": user_id,
            "do_not_disturb_window": user.get("do_not_disturb_window", ""),
            "messages_opened_30d": _safe_int(user.get("messages_opened_30d", "0")),
            "messages_replied_30d": _safe_int(user.get("messages_replied_30d", "0")),
            "notifications_dismissed_30d": _safe_int(
                user.get("notifications_dismissed_30d", "0")
            ),
            "messages_reported_30d": _safe_int(
                user.get("messages_reported_30d", "0")
            ),
        }
    else:
        context["user"] = None
        logger.warning(f"No user profile found for {user_id}")

    # --- Group info (if group message) ---
    if conversation_type == "group" and group_id:
        group = store.get_group(group_id)
        if group:
            context["group"] = {
                "group_id": group_id,
                "group_name": group.get("group_name", ""),
                "group_type": group.get("group_type", ""),
                "member_count": _safe_int(group.get("member_count", "0")),
                "admin_count": _safe_int(group.get("admin_count", "0")),
                "messages_30d": _safe_int(group.get("messages_30d", "0")),
            }
        else:
            context["group"] = None

        # User's membership in this group
        membership = store.get_group_membership(group_id, user_id)
        if membership:
            context["user_group_membership"] = {
                "role": membership.get("role", ""),
                "messages_sent_30d": _safe_int(
                    membership.get("messages_sent_30d", "0")
                ),
                "messages_read_30d": _safe_int(
                    membership.get("messages_read_30d", "0")
                ),
                "replies_sent_30d": _safe_int(
                    membership.get("replies_sent_30d", "0")
                ),
                "notifications_dismissed_30d": _safe_int(
                    membership.get("notifications_dismissed_30d", "0")
                ),
                "group_muted_by_user": membership.get("group_muted_by_user", "0")
                == "1",
            }
        else:
            context["user_group_membership"] = None

        # Sender's role in the group
        if sender_user_id:
            sender_membership = store.get_group_membership(group_id, sender_user_id)
            if sender_membership:
                context["sender_group_role"] = sender_membership.get("role", "member")
            else:
                context["sender_group_role"] = "non_member"
    else:
        context["group"] = None
        context["user_group_membership"] = None
        context["sender_group_role"] = None

    # --- Business info (if business message) ---
    if conversation_type == "business" and business_id:
        business = store.get_business(business_id)
        if business:
            context["business"] = {
                "business_id": business_id,
                "display_name": business.get("display_name", ""),
                "brand_name": business.get("brand_name", ""),
                "category": business.get("category", ""),
                "verified": business.get("verified", "0") == "1",
                "official_domain": business.get("official_domain", ""),
                "domain_used_by_sender": business.get("domain_used_by_sender", ""),
                "account_age_days": _safe_int(business.get("account_age_days", "0")),
                "messages_sent_30d": _safe_int(
                    business.get("messages_sent_30d", "0")
                ),
                "user_reports_30d": _safe_int(business.get("user_reports_30d", "0")),
                "domain_used_by_sender_age_days": _safe_int(
                    business.get("domain_used_by_sender_age_days", "0")
                ),
                "domain_mismatch": (
                    business.get("official_domain", "")
                    != business.get("domain_used_by_sender", "")
                ),
            }
        else:
            context["business"] = None

        # User-business relationship
        relation = store.get_user_business_relation(user_id, business_id)
        if relation:
            context["user_business_relation"] = {
                "why_user_knows_account": relation.get(
                    "why_user_knows_account", ""
                ),
                "last_activity_at": relation.get("last_activity_at", ""),
                "allows_promotions": relation.get("allows_promotions", "0") == "1",
                "promotions_opted_out_at": relation.get(
                    "promotions_opted_out_at", ""
                ),
                "activity_count_180d": _safe_int(
                    relation.get("activity_count_180d", "0")
                ),
                "messages_opened_30d": _safe_int(
                    relation.get("messages_opened_30d", "0")
                ),
                "messages_dismissed_30d": _safe_int(
                    relation.get("messages_dismissed_30d", "0")
                ),
                "messages_replied_30d": _safe_int(
                    relation.get("messages_replied_30d", "0")
                ),
            }
        else:
            context["user_business_relation"] = None
    else:
        context["business"] = None
        context["user_business_relation"] = None

    # --- Sender history (patterns from this sender to this user) ---
    context["sender_history"] = _build_sender_history(
        store, user_id, sender_user_id, business_id, conversation_type
    )

    # --- Daily notification load ---
    daily_summaries = store.get_daily_summary(user_id)
    if daily_summaries:
        recent = sorted(daily_summaries, key=lambda x: x.get("date", ""))[-7:]
        total_sent = sum(_safe_int(d.get("notifications_sent", "0")) for d in recent)
        total_dismissed = sum(
            _safe_int(d.get("notifications_dismissed", "0")) for d in recent
        )
        context["notification_load_7d"] = {
            "total_sent": total_sent,
            "total_dismissed": total_dismissed,
            "dismiss_rate": round(total_dismissed / max(total_sent, 1), 2),
            "avg_daily": round(total_sent / max(len(recent), 1), 1),
        }
    else:
        context["notification_load_7d"] = None

    return context


def _build_sender_history(
    store: DataStore,
    user_id: str,
    sender_user_id: str,
    business_id: str,
    conversation_type: str,
) -> dict[str, Any] | None:
    """
    Summarize the sender's past behavior toward this user.
    For business messages, looks at business history.
    For personal/group, looks at sender_user_id history.
    """
    history_messages: list[dict] = []

    if conversation_type == "business" and business_id:
        # Find past messages from this business to this user
        user_history = store.get_user_history(user_id)
        history_messages = [
            m for m in user_history if m.get("business_id") == business_id
        ]
    elif sender_user_id:
        # Find past messages from this sender to this user
        user_history = store.get_user_history(user_id)
        history_messages = [
            m for m in user_history if m.get("sender_user_id") == sender_user_id
        ]

    if not history_messages:
        return None

    # Summarize patterns
    total = len(history_messages)
    forwarded = sum(
        1 for m in history_messages if _safe_int(m.get("forwarded_count", "0")) > 0
    )
    high_forward = sum(
        1 for m in history_messages if _safe_int(m.get("forwarded_count", "0")) >= 5
    )

    # Check how user reacted to these historical messages
    opened = 0
    replied = 0
    dismissed = 0
    reported = 0
    muted_after = 0

    for m in history_messages:
        msg_id = m.get("message_id", "")
        events = store.get_message_events(msg_id)
        for evt in events:
            if evt.get("user_id") == user_id:
                opened += _safe_int(evt.get("message_opened", "0"))
                replied += _safe_int(evt.get("message_replied", "0"))
                dismissed += _safe_int(evt.get("notification_dismissed", "0"))
                reported += _safe_int(evt.get("message_reported", "0"))
                muted_after += _safe_int(evt.get("muted_after_message", "0"))

    return {
        "total_messages": total,
        "forwarded_count": forwarded,
        "high_forward_count": high_forward,
        "user_opened": opened,
        "user_replied": replied,
        "user_dismissed": dismissed,
        "user_reported": reported,
        "user_muted_after": muted_after,
        "engagement_rate": round(
            (opened + replied) / max(total, 1), 2
        ),
        "dismiss_rate": round(dismissed / max(total, 1), 2),
    }


def format_context_for_prompt(context: dict) -> str:
    """
    Format the context dict into a clean, readable string for the LLM prompt.
    Omits None/empty sections to reduce token count.
    """
    parts: list[str] = []

    # Message details
    msg = context.get("message", {})
    parts.append("=== INCOMING MESSAGE ===")
    parts.append(f"Message ID: {msg.get('message_id', '')}")
    parts.append(f"User: {msg.get('user_id', '')}")
    parts.append(f"Conversation Type: {msg.get('conversation_type', '')}")
    parts.append(f"Timestamp: {msg.get('created_at', '')}")
    parts.append(f"Forwarded Count: {msg.get('forwarded_count', 0)}")

    if msg.get("group_id"):
        parts.append(f"Group: {msg['group_id']}")
    if msg.get("business_id"):
        parts.append(f"Business: {msg['business_id']}")
    if msg.get("sender_user_id"):
        parts.append(f"Sender: {msg['sender_user_id']}")
    if msg.get("media_type"):
        parts.append(f"Media: {msg['media_type']} (ID: {msg.get('media_id', '')})")

    text = msg.get("message_text", "")
    if text:
        parts.append(f"\nMessage Text:\n{text}")
    else:
        parts.append("\nMessage Text: [No text — voice/media only]")

    # User profile
    user = context.get("user")
    if user:
        parts.append("\n=== USER PROFILE ===")
        parts.append(f"DND Window: {user.get('do_not_disturb_window', 'N/A')}")
        parts.append(f"Messages Opened (30d): {user.get('messages_opened_30d', 0)}")
        parts.append(f"Messages Replied (30d): {user.get('messages_replied_30d', 0)}")
        parts.append(
            f"Notifications Dismissed (30d): {user.get('notifications_dismissed_30d', 0)}"
        )
        parts.append(f"Messages Reported (30d): {user.get('messages_reported_30d', 0)}")

    # Group info
    group = context.get("group")
    if group:
        parts.append("\n=== GROUP INFO ===")
        parts.append(
            f"Name: {group.get('group_name', '')} ({group.get('group_type', '')})"
        )
        parts.append(f"Members: {group.get('member_count', 0)}")
        parts.append(f"Admins: {group.get('admin_count', 0)}")
        parts.append(f"Group Activity (30d): {group.get('messages_30d', 0)} messages")

    # User's membership in group
    membership = context.get("user_group_membership")
    if membership:
        parts.append(f"\nUser's Role: {membership.get('role', '')}")
        parts.append(
            f"User Sent (30d): {membership.get('messages_sent_30d', 0)}"
        )
        parts.append(
            f"User Read (30d): {membership.get('messages_read_30d', 0)}"
        )
        parts.append(
            f"User Dismissed (30d): {membership.get('notifications_dismissed_30d', 0)}"
        )
        parts.append(f"Group Muted by User: {membership.get('group_muted_by_user', False)}")

    sender_role = context.get("sender_group_role")
    if sender_role:
        parts.append(f"Sender's Group Role: {sender_role}")

    # Business info
    business = context.get("business")
    if business:
        parts.append("\n=== BUSINESS INFO ===")
        parts.append(
            f"Name: {business.get('display_name', '')} ({business.get('category', '')})"
        )
        parts.append(f"Verified: {business.get('verified', False)}")
        parts.append(f"Official Domain: {business.get('official_domain', '')}")
        parts.append(
            f"Domain Used by Sender: {business.get('domain_used_by_sender', '')}"
        )
        parts.append(
            f"Domain Mismatch: {business.get('domain_mismatch', False)}"
        )
        parts.append(f"Account Age: {business.get('account_age_days', 0)} days")
        parts.append(f"User Reports (30d): {business.get('user_reports_30d', 0)}")

    # User-business relationship
    ubr = context.get("user_business_relation")
    if ubr:
        parts.append(f"\nUser Knows Business Via: {ubr.get('why_user_knows_account', 'unknown')}")
        parts.append(f"Allows Promotions: {ubr.get('allows_promotions', False)}")
        if ubr.get("promotions_opted_out_at"):
            parts.append(
                f"Opted Out of Promotions At: {ubr['promotions_opted_out_at']}"
            )
        parts.append(
            f"Activity Count (180d): {ubr.get('activity_count_180d', 0)}"
        )
        parts.append(
            f"Messages Opened (30d): {ubr.get('messages_opened_30d', 0)}"
        )
        parts.append(
            f"Messages Dismissed (30d): {ubr.get('messages_dismissed_30d', 0)}"
        )

    # Sender history
    sh = context.get("sender_history")
    if sh:
        parts.append("\n=== SENDER HISTORY ===")
        parts.append(f"Total Past Messages: {sh.get('total_messages', 0)}")
        parts.append(f"Forwarded Messages: {sh.get('forwarded_count', 0)}")
        parts.append(f"High-Forward Messages (≥5): {sh.get('high_forward_count', 0)}")
        parts.append(
            f"User Engagement Rate: {sh.get('engagement_rate', 0)}"
        )
        parts.append(f"User Dismiss Rate: {sh.get('dismiss_rate', 0)}")
        if sh.get("user_reported", 0) > 0:
            parts.append(f"User Reported: {sh['user_reported']} times")
        if sh.get("user_muted_after", 0) > 0:
            parts.append(f"User Muted After: {sh['user_muted_after']} times")

    # Notification load
    nl = context.get("notification_load_7d")
    if nl:
        parts.append("\n=== NOTIFICATION LOAD (7d) ===")
        parts.append(f"Total Sent: {nl.get('total_sent', 0)}")
        parts.append(f"Dismiss Rate: {nl.get('dismiss_rate', 0)}")
        parts.append(f"Avg Daily: {nl.get('avg_daily', 0)}")

    return "\n".join(parts)


def _safe_int(value: Any) -> int:
    """Safely convert a value to int, defaulting to 0."""
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return 0
