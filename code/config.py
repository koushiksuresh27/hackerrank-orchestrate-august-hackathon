"""
Configuration for the Message Notification Router.
All paths, API settings, enums, and tunables live here.
Secrets are read from environment variables only.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
MEDIA_DIR = DATASET_DIR / "media"
IMAGES_DIR = MEDIA_DIR / "images"
AUDIO_DIR = MEDIA_DIR / "audio"
OUTPUT_PATH = DATASET_DIR / "output.csv"

# CSV file paths
CSV_FILES = {
    "messages": DATASET_DIR / "messages.csv",
    "sample_messages": DATASET_DIR / "sample_messages.csv",
    "users": DATASET_DIR / "users.csv",
    "groups": DATASET_DIR / "groups.csv",
    "group_members": DATASET_DIR / "group_members.csv",
    "business_accounts": DATASET_DIR / "business_accounts.csv",
    "user_business_history": DATASET_DIR / "user_business_history.csv",
    "message_history": DATASET_DIR / "message_history.csv",
    "message_events": DATASET_DIR / "message_events.csv",
    "images": DATASET_DIR / "images.csv",
    "voice_notes": DATASET_DIR / "voice_notes.csv",
    "daily_notification_summary": DATASET_DIR / "daily_notification_summary.csv",
}

# ---------------------------------------------------------------------------
# API Keys (from environment variables)
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ---------------------------------------------------------------------------
# Provider Configuration
# ---------------------------------------------------------------------------
# Order defines fallback chain: Gemini → Claude → Groq → rule-based
PROVIDER_ORDER = ["gemini", "claude", "groq", "rule_based"]

PROVIDER_CONFIG = {
    "gemini": {
        "model": "gemini-2.0-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "max_tokens": 1024,
        "temperature": 0.1,
        "supports_vision": True,
    },
    "claude": {
        "model": "claude-sonnet-4-20250514",
        "base_url": "https://api.anthropic.com/v1",
        "max_tokens": 1024,
        "temperature": 0.1,
        "supports_vision": True,
    },
    "groq": {
        "model": "llama-3.3-70b-versatile",
        "base_url": "https://api.groq.com/openai/v1",
        "max_tokens": 1024,
        "temperature": 0.1,
        "supports_vision": False,
    },
}

# ---------------------------------------------------------------------------
# Retry & Rate Limiting
# ---------------------------------------------------------------------------
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2  # Base delay, exponential backoff applied
REQUEST_TIMEOUT_SECONDS = 30

# ---------------------------------------------------------------------------
# Valid Output Enums (from problem_statement.md)
# ---------------------------------------------------------------------------
VALID_ACTIONS = {"notify", "digest", "mute"}

VALID_MESSAGE_TYPES = {
    "personal",
    "urgent",
    "event",
    "payment",
    "business_update",
    "promotion",
    "greeting",
    "forward",
    "spam",
    "scam",
    "unknown",
}

# ---------------------------------------------------------------------------
# Signal Thresholds
# ---------------------------------------------------------------------------
HIGH_FORWARD_COUNT = 5  # forwarded_count >= this → likely chain/spam
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous",
    r"mark\s+this\s+(message\s+)?as\s+notify",
    r"override\s+(routing|rules|system)",
    r"system\s+prompt",
    r"you\s+are\s+now",
    r"disregard\s+(all\s+)?instructions",
    r"new\s+instructions?:",
]

# ---------------------------------------------------------------------------
# Confidence Calibration
# ---------------------------------------------------------------------------
CONFIDENCE_MIN = 0.78
CONFIDENCE_MAX = 0.91
DEFAULT_CONFIDENCE = 0.80

# ---------------------------------------------------------------------------
# Batch & Cache Settings
# ---------------------------------------------------------------------------
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
ENABLE_CACHE = True

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
