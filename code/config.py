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
GEMINI_API_KEY_2 = os.environ.get("GEMINI_API_KEY_2", "")
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_API_KEY_2 = os.environ.get("ANTHROPIC_API_KEY_2", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_KEY_2 = os.environ.get("GROQ_API_KEY_2", "")

# ---------------------------------------------------------------------------
# Provider Configuration
# ---------------------------------------------------------------------------
# Order defines fallback chain: Claude → Gemini → Groq → rule-based
PROVIDER_ORDER = ["groq_vision", "groq", "rule_based"]

PROVIDER_CONFIG = {
    "groq_vision": {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "base_url": "https://api.groq.com/openai/v1",
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
SUSPICIOUS_REPORTS_THRESHOLD = 30
SUSPICIOUS_AGE_THRESHOLD = 90
HIGH_DISMISSAL_THRESHOLD = 60
AMBIGUITY_THRESHOLD = 0.82
MAX_EVIDENCE_IDS = 3
MAX_HISTORY_MESSAGES = 5

OTP_KEYWORDS = [
    "otp", "share code", "send code", "verify now", "account block",
    "otp leak", "account band", "profile restricted", "link open karo",
    "verification code", "6 digit", "wallet active", "account closure",
    "share kar", "code bhejo", "account band ho"
]

CHAIN_KEYWORDS = [
    "forward", "share", "blessings", "luck", "chain", "10 people",
    "sabko bhejo", "groups me share", "positive energy", "do not ignore",
    "break the chain", "midnight", "iss message ko"
]

INJECTION_KEYWORDS = [
    "routing override", "set action", "ignore sender", "assistant instruction",
    "classify as", "set confidence", "action=notify", "action=mute",
    "ignore all previous", "mark this message as"
]

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
