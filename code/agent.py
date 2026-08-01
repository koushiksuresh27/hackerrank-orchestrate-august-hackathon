"""
Agent — the routing decision maker.

Single point where the LLM makes the routing decision.
Orchestrates: prompt building → LLM call → response parsing.
Also contains the rule-based fallback for when all LLM providers fail.
"""

import logging
from typing import Any

from config import VALID_ACTIONS, VALID_MESSAGE_TYPES, DEFAULT_CONFIDENCE
from context_builder import format_context_for_prompt
from evidence_selector import get_evidence_context, select_evidence
from media_handler import get_image_mime_type
from prompts.router_prompt import build_full_prompt
from provider_router import ProviderRouter
from signal_extractor import early_exit

logger = logging.getLogger(__name__)


class RoutingAgent:
    """Makes routing decisions using LLM + signals + evidence."""

    def __init__(self, provider_router: ProviderRouter):
        self.router = provider_router

    def route_message(
        self,
        message: dict,
        context: dict,
        signals: dict,
        media_result: dict,
        evidence_ids: str,
        store: Any,  # DataStore
    ) -> dict:
        """
        Route a single message through the pipeline.

        Args:
            message: Raw message row
            context: Context dict from context_builder
            signals: Signal extraction result
            media_result: Media processing result (augmented text, image data)
            evidence_ids: Pre-selected evidence IDs
            store: DataStore instance

        Returns:
            Dict with action, message_type, reason, confidence, evidence_message_ids
        """
        message_id = message.get("message_id", "")

        # 2. Build prompt with augmented text
        # Replace message text with augmented version (includes voice transcription)
        augmented_context = context.copy()
        if media_result.get("augmented_text"):
            augmented_context["message"] = context["message"].copy()
            augmented_context["message"]["message_text"] = media_result["augmented_text"]

        context_text = format_context_for_prompt(augmented_context)
        
        # Format signals
        active_signals = [k for k, v in signals.items() if v is True and k != "ambiguity_score"]
        if active_signals:
            signals_text = "=== PRE-ANALYSIS SIGNALS ===\n" + "\n".join(f"[!] {k}" for k in active_signals)
        else:
            signals_text = "=== PRE-ANALYSIS SIGNALS ===\nNo significant signals detected."

        evidence_text = get_evidence_context(evidence_ids, store)

        prompt = build_full_prompt(context_text, signals_text, evidence_text)

        # 3. Call LLM (with image if available)
        image_base64 = media_result.get("image_base64")
        image_mime = "image/jpeg"
        if media_result.get("image_path"):
            image_mime = get_image_mime_type(media_result["image_path"])

        llm_result = self.router.call_llm(prompt, image_base64, image_mime)

        # 4. If LLM succeeded, merge with evidence
        if llm_result:
            llm_result["evidence_message_ids"] = evidence_ids
            logger.info(
                f"[{message_id}] LLM decision: {llm_result.get('action')} / "
                f"{llm_result.get('message_type')} (conf: {llm_result.get('confidence')})"
            )
            return llm_result

        # 5. All LLM providers failed → rule-based fallback
        logger.warning(f"[{message_id}] All LLM providers failed, using rule-based fallback")
        fallback = self._rule_based_fallback(message, context, signals)
        fallback["evidence_message_ids"] = evidence_ids
        return fallback

    def _rule_based_fallback(
        self, message: dict, context: dict, signals: dict
    ) -> dict:
        """
        Rule-based fallback for when all LLM providers fail.
        Handles only clear-cut cases to avoid brittle heuristics.
        """
        msg = context.get("message", {})
        text = (msg.get("message_text", "") or "").lower()
        forwarded = msg.get("forwarded_count", 0)
        conv_type = msg.get("conversation_type", "")

        # Scam signals: domain mismatch + OTP/password keywords
        if signals.get("is_domain_mismatch") or signals.get("is_prompt_injection"):
            return {
                "action": "mute",
                "message_type": "scam",
                "reason": "The message contains suspicious signals that suggest it may be a scam.",
                "confidence": 0.82,
            }

        # High forward → mute as forward/spam
        if signals.get("is_chain_forward"):
            return {
                "action": "mute",
                "message_type": "forward",
                "reason": "The message has a high forward count suggesting it is a chain message.",
                "confidence": 0.83,
            }

        # Muted group without direct mention → mute
        if signals.get("is_group_muted") and not signals.get("is_direct_mention"):
            return {
                "action": "mute",
                "message_type": "unknown",
                "reason": "The message is from a group the user has muted.",
                "confidence": 0.80,
            }

        # OTP/password keywords from unknown sender
        scam_keywords = ["otp", "password", "verify now", "account blocked",
                         "payment failed", "click here", "act now"]
        if signals.get("sender_is_known") is False and any(kw in text for kw in scam_keywords):
            return {
                "action": "mute",
                "message_type": "scam",
                "reason": "This is the first message from the sender and it asks for sensitive verification or payment.",
                "confidence": 0.85,
            }

        # Direct mention → notify
        if signals.get("is_direct_mention"):
            return {
                "action": "notify",
                "message_type": "personal",
                "reason": "The message contains a direct mention of the user.",
                "confidence": 0.82,
            }

        # Opted-out promotions → mute
        if signals.get("is_opted_out"):
            return {
                "action": "mute",
                "message_type": "promotion",
                "reason": "The user has opted out of promotions from this business.",
                "confidence": 0.80,
            }

        # Default: digest with low confidence
        return {
            "action": "digest",
            "message_type": "unknown",
            "reason": "Unable to determine urgency. Routing to digest as a safe default.",
            "confidence": 0.78,
        }
