"""
Media Handler — processes images and voice notes for the routing pipeline.

Voice notes: Transcribed using OpenAI Whisper (local) with result prefixed
    as "[Audio Transcription]: ..." and appended to any existing message_text.

Images: Loaded as base64 for Gemini multimodal input, or described via
    vision model fallback.
"""

import base64
import logging
import subprocess
import json
import whisper
from pathlib import Path
from typing import Any

from data_loader import DataStore
from config import DATASET_DIR

logger = logging.getLogger(__name__)

# Cache transcriptions so we don't re-process
_transcription_cache: dict[str, str] = {}


def process_media(message: dict, store: DataStore) -> dict[str, Any]:
    """
    Process media attached to a message.

    Args:
        message: Raw message row from messages.csv
        store: DataStore instance

    Returns:
        Dict with:
          - "augmented_text": message_text with voice transcription appended
          - "image_base64": base64-encoded image data (if image message)
          - "image_path": filesystem path to image (if image message)
          - "has_voice": bool
          - "has_image": bool
    """
    media_type = (message.get("media_type", "") or "").strip().lower()
    media_id = (message.get("media_id", "") or "").strip()
    original_text = message.get("message_text", "") or ""

    result: dict[str, Any] = {
        "augmented_text": original_text,
        "image_base64": None,
        "image_path": None,
        "has_voice": False,
        "has_image": False,
    }

    if media_type == "voice" and media_id:
        result["has_voice"] = True
        transcription = transcribe_voice_note(media_id, store)
        if transcription:
            # Prefix transcription to help LLM distinguish typed vs spoken
            prefixed = f"[Audio Transcription]: {transcription}"
            if original_text.strip():
                result["augmented_text"] = f"{original_text}\n\n{prefixed}"
            else:
                result["augmented_text"] = prefixed

    elif media_type == "image" and media_id:
        result["has_image"] = True
        image_path = store.get_image_path(media_id)
        if image_path and image_path.exists():
            result["image_path"] = str(image_path)
            result["image_base64"] = _encode_image_base64(image_path)
        else:
            logger.warning(
                f"Image file not found for media_id={media_id}: {image_path}"
            )

    return result


def transcribe_voice_note(media_id: str, store: DataStore) -> str | None:
    """
    Transcribe a voice note using Whisper.

    Tries local Whisper first (via CLI), falls back to a simple
    ffmpeg + whisper approach.

    Args:
        media_id: The voice note ID (e.g., "vn_001")
        store: DataStore for path lookup

    Returns:
        Transcription text, or None if transcription failed.
    """
    try:
        if media_id in _transcription_cache:
            return _transcription_cache[media_id]

        audio_path = store.get_voice_note_path(media_id)
        if not audio_path or not audio_path.exists():
            logger.warning(f"Voice note file not found: {media_id}")
            return None

        logger.info(f"Transcribing voice note: {media_id} ({audio_path})")

        # Try local whisper python API
        transcription = _transcribe_with_whisper_local(audio_path)

        if transcription is None:
            # Fallback: try Groq's whisper API if available
            transcription = _transcribe_with_groq_whisper(audio_path)

        if transcription:
            _transcription_cache[media_id] = transcription
            logger.info(
                f"Transcribed {media_id}: {transcription[:80]}..."
                if len(transcription) > 80
                else f"Transcribed {media_id}: {transcription}"
            )
        else:
            logger.warning(f"Failed to transcribe {media_id}")

        return transcription
    except BaseException as e:
        logger.error(f"transcribe_voice_note completely failed: {type(e).__name__} - {e}")
        return None


_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("tiny")  # tiny is fastest
    return _whisper_model

def _transcribe_with_whisper_local(audio_path: Path) -> str | None:
    """Transcribe using the local Whisper Python API."""
    try:
        model = get_whisper_model()
        result = model.transcribe(str(audio_path), fp16=False)
        return result["text"].strip()
    except Exception as e:
        logger.debug(f"Whisper local API error: {e}")
        return None


def _transcribe_with_groq_whisper(audio_path: Path) -> str | None:
    """Transcribe using Groq's whisper-large-v3 API as fallback."""
    import os

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return None

    try:
        import requests

        with open(audio_path, "rb") as f:
            response = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (audio_path.name, f, "audio/mpeg")},
                data={
                    "model": "whisper-large-v3",
                    "language": "en",
                    "response_format": "json",
                },
                timeout=60,
            )

        if response.status_code == 200:
            return response.json().get("text", "").strip()
        else:
            logger.debug(
                f"Groq whisper API failed ({response.status_code}): {response.text[:200]}"
            )
            return None
    except Exception as e:
        logger.debug(f"Groq whisper API error: {e}")
        return None


def _encode_image_base64(image_path: Path) -> str | None:
    """Read an image file and return its base64-encoded content."""
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode("utf-8")
    except Exception as e:
        logger.warning(f"Failed to encode image {image_path}: {e}")
        return None


def get_image_mime_type(image_path: str) -> str:
    """Determine MIME type from file extension."""
    ext = Path(image_path).suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return mime_map.get(ext, "image/jpeg")
