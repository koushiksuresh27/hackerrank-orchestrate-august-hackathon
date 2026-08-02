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

        # Route image messages directly to Claude for vision support
        if media_result.get("image_base64"):
            result = self.router._call_claude_fallback(
                prompt,
                image_base64,
                image_mime,
            )
            if result:
                result["evidence_message_ids"] = evidence_ids
                logger.info(
                    f"[{message_id}] Claude vision decision: {result.get('action')} / "
                    f"{result.get('message_type')} (conf: {result.get('confidence')})"
                )
                return result
            # If Gemini fails, fall through to normal provider chain

        # Normal provider chain for all other messages
        llm_result = self.router.call_llm(prompt, image_base64, image_mime)

        # 4. If LLM succeeded, check confidence for two-step loop
        if llm_result:
            confidence = float(llm_result.get("confidence", 0.0))
            if confidence >= 0.82:
                llm_result["evidence_message_ids"] = evidence_ids
                logger.info(
                    f"[{message_id}] LLM decision (Pass 1): {llm_result.get('action')} / "
                    f"{llm_result.get('message_type')} (conf: {confidence})"
                )
                return llm_result
                
            # Step 2: Confidence < 0.82 -> enrich and retry
            logger.info(f"[{message_id}] Confidence {confidence} < 0.82, enriching prompt for second pass")
            enriched_prompt = prompt + "\n\nSECOND PASS: Previous confidence was low. Focus especially on: sender history, user engagement pattern, and whether this message type has been dismissed before by this user."
            
            second_result = self.router.call_llm(enriched_prompt, image_base64, image_mime)
            if second_result:
                second_result["evidence_message_ids"] = evidence_ids
                logger.info(
                    f"[{message_id}] LLM decision (Pass 2): {second_result.get('action')} / "
                    f"{second_result.get('message_type')} (conf: {second_result.get('confidence')})"
                )
                return second_result
            else:
                llm_result["evidence_message_ids"] = evidence_ids
                return llm_result

        # 5. First LLM call failed entirely → rule-based fallback
        logger.warning(f"[{message_id}] First LLM call failed entirely, using rule-based fallback")
        fallback = self._rule_based_fallback(message, context, signals)
        fallback["evidence_message_ids"] = evidence_ids
        return fallback

    def _rule_based_fallback(self, message, context, signals):
        msg = context.get("message", {})
        text = (msg.get("message_text", "") or "").lower()
        conv_type = msg.get("conversation_type", "")
        business = context.get("business", {}) or {}
        group = context.get("group", {}) or {}
        membership = context.get("user_group_membership", {}) or {}
    
        if signals.get("is_prompt_injection"):
            return {
                "action": "mute",
                "message_type": "scam",
                "reason": "Message contains prompt injection attempt — instructions embedded in text to manipulate the routing system.",
                "confidence": 0.91,
            }
    
        if signals.get("is_domain_mismatch"):
            official = business.get("official_domain", "unknown")
            used = business.get("domain_used_by_sender", "unknown")
            return {
                "action": "mute",
                "message_type": "scam",
                "reason": f"Business domain mismatch: official domain is {official} but sender used {used}. High fraud signal.",
                "confidence": 0.85,
            }
    
        if signals.get("is_chain_forward"):
            count = msg.get("forwarded_count", 0)
            return {
                "action": "mute",
                "message_type": "forward",
                "reason": f"Forwarded {count} times with no actionable content — consistent with chain or mass-forward pattern.",
                "confidence": 0.83,
            }
    
        if signals.get("is_group_muted") and not signals.get("is_direct_mention"):
            group_name = group.get("group_name", "this group")
            dismissed = membership.get("notifications_dismissed_30d", 0)
            return {
                "action": "digest",
                "message_type": "unknown",
                "reason": f"User has muted {group_name} and dismissed {dismissed} notifications from it in the last 30 days. No direct mention found.",
                "confidence": 0.80,
            }
    
        if signals.get("is_opted_out"):
            return {
                "action": "mute",
                "message_type": "promotion",
                "reason": "User has explicitly opted out of promotions from this business.",
                "confidence": 0.82,
            }
    
        if signals.get("is_direct_mention"):
            return {
                "action": "notify",
                "message_type": "personal",
                "reason": "Message contains a direct @mention of the user in a group context.",
                "confidence": 0.82,
            }
    
        return {
            "action": "digest",
            "message_type": "unknown",
            "reason": "No strong urgency or risk signals detected. Routing to digest as a safe default.",
            "confidence": 0.78,
        }
