import logging
import re
from typing import Any

from data_loader import DataStore

logger = logging.getLogger(__name__)

# Scam signal patterns
_OTP_PATTERNS = [
    "otp", "share code", "send code", "verify now", "account block",
    "otp leak", "account band", "profile restricted", "link open karo",
    "verification code", "6 digit", "wallet active", "account closure"
]
_OTP_REGEX = re.compile("|".join(re.escape(p) for p in _OTP_PATTERNS), re.IGNORECASE)

_CHAIN_PATTERNS = [
    "forward", "share", "blessings", "luck", "chain", "10 people",
    "sabko bhejo", "groups me share", "positive energy", 
    "do not ignore", "break the chain", "midnight"
]
_CHAIN_REGEX = re.compile("|".join(re.escape(p) for p in _CHAIN_PATTERNS), re.IGNORECASE)

_INJECTION_PATTERNS = [
    "routing override", "set action", "ignore sender", "assistant instruction",
    "classify as", "set confidence", "action=notify", "action=mute",
    "action=digest", "ignore all previous"
]
_INJECTION_REGEX = re.compile("|".join(re.escape(p) for p in _INJECTION_PATTERNS), re.IGNORECASE)

_ACTIVE_TX_PATTERNS = [
    "order", "booking", "payment", "delivery", "appointment"
]
_ACTIVE_TX_REGEX = re.compile("|".join(re.escape(p) for p in _ACTIVE_TX_PATTERNS), re.IGNORECASE)


def _is_during_dnd(created_at: str, dnd_window: str) -> bool:
    try:
        if not dnd_window or not created_at:
            return False
        parts = dnd_window.split("-")
        if len(parts) != 2:
            return False
        start_h, start_m = map(int, parts[0].strip().split(":"))
        end_h, end_m = map(int, parts[1].strip().split(":"))

        time_part = created_at.strip().split(" ")[-1]
        msg_h, msg_m = map(int, time_part.split(":"))

        start_minutes = start_h * 60 + start_m
        end_minutes = end_h * 60 + end_m
        msg_minutes = msg_h * 60 + msg_m

        if start_minutes <= end_minutes:
            return start_minutes <= msg_minutes <= end_minutes
        else:
            return msg_minutes >= start_minutes or msg_minutes <= end_minutes
    except (ValueError, IndexError, AttributeError):
        return False


def extract_signals(message: dict, store: DataStore) -> dict:
    """Computes deterministic boolean signals for a message BEFORE any LLM call."""
    msg_text = (message.get("message_text") or "").strip()
    msg_text_lower = msg_text.lower()
    
    sender_id = message.get("sender_user_id", "")
    user_id = message.get("user_id", "")
    group_id = message.get("group_id", "")
    business_id = message.get("business_id", "")
    forwarded_count = int(message.get("forwarded_count") or 0)
    created_at = message.get("created_at", "")
    media_type = (message.get("media_type") or "").strip().lower()
    
    # Store lookups
    user = store.get_user(user_id) or {}
    group_member = store.get_group_member(user_id, group_id) or {}
    business = store.get_business(business_id) or {}
    user_biz_history = store.get_user_business_relation(user_id, business_id) or {}
    
    # Message history
    history = store.get_user_history(user_id) or []
    sender_history = [m for m in history if m.get("sender_user_id") == sender_id]

    # --- SCAM SIGNALS ---
    is_otp_scam = bool(_OTP_REGEX.search(msg_text_lower))
    is_chain_forward = forwarded_count > 5 and bool(_CHAIN_REGEX.search(msg_text_lower))
    is_high_forward = forwarded_count > 5
    is_prompt_injection = bool(_INJECTION_REGEX.search(msg_text_lower))
    
    is_domain_mismatch = False
    is_suspicious_business = False
    
    if business_id and business:
        domain_used = business.get("domain_used_by_sender")
        official_domain = business.get("official_domain")
        if domain_used and official_domain and domain_used != official_domain:
            is_domain_mismatch = True
            
        verified = str(business.get("verified", "")).lower() in ["1", "true"]
        user_reports_30d = int(business.get("user_reports_30d") or 0)
        account_age = int(business.get("account_age_days") or 999)
        
        if not verified and user_reports_30d > 30 and account_age < 90:
            is_suspicious_business = True

    # --- USER SIGNALS ---
    is_quiet_hours = _is_during_dnd(created_at, user.get("do_not_disturb_window", ""))
    
    notifications_dismissed = int(user.get("notifications_dismissed_30d") or 0)
    is_high_dismissal_user = notifications_dismissed > 60
    
    messages_replied = float(user.get("messages_replied_30d") or 0)
    messages_opened = float(user.get("messages_opened_30d") or 0)
    user_reply_rate = messages_replied / max(messages_opened, 1.0)
    if not user:
        user_reply_rate = 0.0

    # --- GROUP SIGNALS ---
    is_group_muted = str(group_member.get("group_muted_by_user", "")).lower() in ["1", "true"]
    is_user_group_admin = str(group_member.get("role", "")).lower() == "admin"
    is_direct_mention = user_id in msg_text if user_id else False

    # --- BUSINESS SIGNALS ---
    is_opted_out = user_biz_history.get("promotions_opted_out_at") is not None and user_biz_history.get("promotions_opted_out_at") != ""
    
    why_knows = user_biz_history.get("why_user_knows_account", "")
    has_active_transaction = bool(_ACTIVE_TX_REGEX.search(why_knows))
    
    is_verified_business = False
    if business:
        is_verified_business = str(business.get("verified", "")).lower() in ["1", "true"]

    # --- HISTORY SIGNALS ---
    sender_previously_reported = False
    sender_reply_rate = 0.0
    sender_is_known = len(sender_history) > 0
    
    if sender_history:
        replied_count = 0
        total_events = 0
        for m in sender_history:
            m_id = m.get("message_id")
            if m_id:
                events = store.get_message_events(m_id) or []
                total_events += len(events)
                for e in events:
                    if e.get("event_type") == "reported":
                        sender_previously_reported = True
                    if e.get("event_type") == "replied":
                        replied_count += 1
        
        if total_events > 0:
            sender_reply_rate = replied_count / total_events

    # --- MEDIA SIGNALS ---
    has_text = msg_text is not None and len(msg_text) > 0
    is_voice_only = media_type == "voice" and not has_text
    is_image_message = media_type == "image"

    # --- COMPLEXITY SCORE ---
    ambiguity_score = 0.0
    if not sender_is_known:
        ambiguity_score += 0.15
    if not group_id and not business_id:
        ambiguity_score += 0.15
    
    # conflicting signals
    if is_verified_business and is_domain_mismatch:
        ambiguity_score += 0.15
        
    if user_reply_rate < 0.1 and not is_high_dismissal_user:
        ambiguity_score += 0.15
        
    if is_high_forward or forwarded_count > 2:
        ambiguity_score += 0.15
        
    ambiguity_score = min(ambiguity_score, 1.0)

    return {
        "is_otp_scam": is_otp_scam,
        "is_chain_forward": is_chain_forward,
        "is_high_forward": is_high_forward,
        "is_prompt_injection": is_prompt_injection,
        "is_domain_mismatch": is_domain_mismatch,
        "is_suspicious_business": is_suspicious_business,
        "is_quiet_hours": is_quiet_hours,
        "is_high_dismissal_user": is_high_dismissal_user,
        "user_reply_rate": user_reply_rate,
        "is_group_muted": is_group_muted,
        "is_user_group_admin": is_user_group_admin,
        "is_direct_mention": is_direct_mention,
        "is_opted_out": is_opted_out,
        "has_active_transaction": has_active_transaction,
        "is_verified_business": is_verified_business,
        "sender_previously_reported": sender_previously_reported,
        "sender_reply_rate": sender_reply_rate,
        "sender_is_known": sender_is_known,
        "is_voice_only": is_voice_only,
        "is_image_message": is_image_message,
        "has_text": has_text,
        "ambiguity_score": ambiguity_score
    }


def early_exit(signals: dict) -> dict | None:
    """Returns a complete result dict if a clear rule applies, else None."""
    
    if signals.get("is_prompt_injection"):
        return {
            "action": "mute",
            "message_type": "scam",
            "reason": "Message contains instructions attempting to override the routing system, consistent with a prompt injection attack.",
            "confidence": 0.91,
            "evidence_message_ids": "none"
        }
        
    if signals.get("is_otp_scam"):
        return {
            "action": "mute",
            "message_type": "scam",
            "reason": "Message requests OTP or verification code from the user, consistent with a phishing or scam pattern.",
            "confidence": 0.90,
            "evidence_message_ids": "none"
        }
        
    if signals.get("is_chain_forward"):
        return {
            "action": "mute",
            "message_type": "forward",
            "reason": "High-forward chain message with blessing or luck keywords has no personal relevance to the user.",
            "confidence": 0.88,
            "evidence_message_ids": "none"
        }
        
    if signals.get("is_high_forward"):
        return {
            "action": "mute",
            "message_type": "forward",
            "reason": "Message has been forwarded extensively and is unlikely to be personally relevant to the user.",
            "confidence": 0.85,
            "evidence_message_ids": "none"
        }
        
    if signals.get("is_suspicious_business"):
        return {
            "action": "mute",
            "message_type": "scam",
            "reason": "Unverified business with domain mismatch and high user report count indicates likely scam sender.",
            "confidence": 0.89,
            "evidence_message_ids": "none"
        }
        
    if signals.get("is_opted_out"):
        return {
            "action": "mute",
            "message_type": "promotion",
            "reason": "User has opted out of promotional messages from this business.",
            "confidence": 0.86,
            "evidence_message_ids": "none"
        }
        
    if signals.get("is_group_muted") and not signals.get("is_direct_mention"):
        return {
            "action": "mute",
            "message_type": "unknown",
            "reason": "User has muted this group and was not directly mentioned in the message.",
            "confidence": 0.83,
            "evidence_message_ids": "none"
        }
        
    return None
