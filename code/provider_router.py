"""
Provider Router — multi-provider LLM fallback chain.

Routes LLM calls through: Gemini Flash → Claude → Groq → rule-based fallback.
Each provider has its own API format and error handling.
Handles multimodal input (images) for providers that support it.
"""

import json
import logging
import os
import time
from typing import Any

import requests

from config import (
    PROVIDER_ORDER,
    PROVIDER_CONFIG,
    MAX_RETRIES,
    RETRY_DELAY_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


class ProviderRouter:
    """Routes LLM requests through the provider fallback chain."""

    def __init__(self):
        self._provider_status: dict[str, bool] = {
            p: True for p in PROVIDER_ORDER if p != "rule_based"
        }

    def call_llm(
        self,
        prompt: str,
        image_base64: str | None = None,
        image_mime: str = "image/jpeg",
    ) -> dict | None:
        """
        Send a prompt to the LLM provider chain.

        Args:
            prompt: The full routing prompt
            image_base64: Optional base64-encoded image data
            image_mime: MIME type of the image

        Returns:
            Parsed JSON response dict, or None if all providers fail.
        """
        for provider_name in PROVIDER_ORDER:
            if provider_name == "rule_based":
                # Rule-based is not an LLM call, handled separately
                continue

            if not self._provider_status.get(provider_name, False):
                logger.debug(f"Skipping disabled provider: {provider_name}")
                continue

            # Skip image for non-vision providers
            img = image_base64 if PROVIDER_CONFIG.get(provider_name, {}).get("supports_vision") else None

            logger.info(f"[provider_router] Trying provider: {provider_name}")
            try:
                result = self._call_provider(provider_name, prompt, img, image_mime)
                if result is not None:
                    return result
            except Exception as e:
                logger.warning(f"[provider_router] {provider_name} failed: {type(e).__name__}: {str(e)[:100]}")
                continue

        logger.error("All LLM providers failed")
        return None

    def _call_provider(
        self,
        provider_name: str,
        prompt: str,
        image_base64: str | None,
        image_mime: str,
    ) -> dict | None:
        """Dispatch to the appropriate provider handler."""
        if provider_name == "gemini":
            return self._call_gemini(prompt, image_base64, image_mime)
        elif provider_name == "claude":
            return self._call_claude(prompt, image_base64, image_mime)
        elif provider_name == "groq":
            return self._call_groq(prompt)
        return None

    def _call_gemini(
        self, prompt: str, image_base64: str | None, image_mime: str
    ) -> dict | None:
        """Call Gemini Flash API."""
        keys = [k for k in [
            os.getenv("GEMINI_API_KEY"),
            os.getenv("GEMINI_API_KEY_2"),
        ] if k]
        
        if not keys:
            logger.debug("Gemini API key not set")
            return None

        config = PROVIDER_CONFIG["gemini"]

        # Build parts
        parts: list[dict] = []

        if image_base64:
            parts.append({
                "inline_data": {
                    "mime_type": image_mime,
                    "data": image_base64,
                }
            })

        parts.append({"text": prompt})

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": config["temperature"],
                "maxOutputTokens": config["max_tokens"],
            },
        }

        for api_key in keys:
            url = (
                f"{config['base_url']}/models/{config['model']}:generateContent"
                f"?key={api_key}"
            )
            for attempt in range(MAX_RETRIES):
                try:
                    resp = requests.post(
                        url,
                        json=payload,
                        timeout=REQUEST_TIMEOUT_SECONDS,
                    )

                    if resp.status_code == 200:
                        data = resp.json()
                        text = (
                            data.get("candidates", [{}])[0]
                            .get("content", {})
                            .get("parts", [{}])[0]
                            .get("text", "")
                        )
                        return _parse_json_response(text)

                    elif resp.status_code == 429:
                        logger.warning("Gemini rate limited, trying next key or falling back...")
                        break

                    else:
                        raise Exception(f"HTTP {resp.status_code} - {resp.text[:200]}")

                except requests.Timeout:
                    logger.warning(f"Gemini timeout (attempt {attempt + 1})")
                    continue

        return None

    def _call_claude(
        self, prompt: str, image_base64: str | None, image_mime: str
    ) -> dict | None:
        """Call Claude API."""
        keys = [k for k in [
            os.getenv("ANTHROPIC_API_KEY"),
            os.getenv("ANTHROPIC_API_KEY_2"),
        ] if k]
        
        if not keys:
            logger.debug("Claude API key not set")
            return None

        config = PROVIDER_CONFIG["claude"]

        # Build content blocks
        content: list[dict] = []

        if image_base64:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image_mime,
                    "data": image_base64,
                },
            })

        content.append({"type": "text", "text": prompt})

        payload = {
            "model": config["model"],
            "max_tokens": config["max_tokens"],
            "temperature": config["temperature"],
            "messages": [{"role": "user", "content": content}],
        }

        for api_key in keys:
            headers = {
                "x-api-key": api_key,
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
            }

            for attempt in range(MAX_RETRIES):
                try:
                    resp = requests.post(
                        f"{config['base_url']}/messages",
                        json=payload,
                        headers=headers,
                        timeout=REQUEST_TIMEOUT_SECONDS,
                    )

                    if resp.status_code == 200:
                        data = resp.json()
                        text = data.get("content", [{}])[0].get("text", "")
                        return _parse_json_response(text)

                    elif resp.status_code == 429:
                        logger.warning("Claude rate limited, trying next key or falling back...")
                        break

                    else:
                        raise Exception(f"HTTP {resp.status_code} - {resp.text[:200]}")

                except requests.Timeout:
                    logger.warning(f"Claude timeout (attempt {attempt + 1})")
                    continue

        return None

    def _call_groq(self, prompt: str) -> dict | None:
        """Call Groq API (OpenAI-compatible format, text only)."""
        keys = [k for k in [
            os.getenv("GROQ_API_KEY"),
            os.getenv("GROQ_API_KEY_2"),
            os.getenv("GROQ_API_KEY_3"),
        ] if k]
        
        if not keys:
            logger.debug("Groq API key not set")
            return None

        config = PROVIDER_CONFIG["groq"]

        payload = {
            "model": config["model"],
            "max_tokens": config["max_tokens"],
            "temperature": config["temperature"],
            "messages": [{"role": "user", "content": prompt}],
        }

        for api_key in keys:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            for attempt in range(MAX_RETRIES):
                try:
                    resp = requests.post(
                        f"{config['base_url']}/chat/completions",
                        json=payload,
                        headers=headers,
                        timeout=REQUEST_TIMEOUT_SECONDS,
                    )

                    if resp.status_code == 200:
                        data = resp.json()
                        text = (
                            data.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        )
                        return _parse_json_response(text)

                    elif resp.status_code == 429:
                        logger.warning("Groq rate limited, trying next key or falling back...")
                        break

                    else:
                        raise Exception(f"HTTP {resp.status_code} - {resp.text[:200]}")

                except requests.Timeout:
                    logger.warning(f"Groq timeout (attempt {attempt + 1})")
                    continue

        return None


def _parse_json_response(text: str) -> dict | None:
    """
    Parse JSON from LLM response text.
    Handles common issues: markdown code fences, extra whitespace, trailing text.
    """
    if not text:
        return None

    # Strip markdown code fences
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON object from the text
    import re
    json_match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    logger.warning(f"Failed to parse JSON from response: {text[:200]}")
    return None
