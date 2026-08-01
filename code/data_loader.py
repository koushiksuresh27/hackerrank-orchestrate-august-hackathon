"""
Data Loader — reads and normalizes all CSV files into indexed lookup structures.

All CSV files are loaded once and indexed by their primary keys for O(1) lookup.
This module is the single source of truth for raw data access.
"""

import csv
import logging
from pathlib import Path
from typing import Any

from config import CSV_FILES, DATASET_DIR

logger = logging.getLogger(__name__)


def _clean_value(val: str | None) -> str | None:
    """Convert empty strings to None."""
    return None if val == "" else val


def _read_csv(path: Path) -> list[dict[str, Any]]:
    """Read a CSV file and return a list of row dicts. Handles multiline fields."""
    if not path.exists():
        logger.error(f"CSV file not found: {path}")
        return []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = [{k: _clean_value(v) for k, v in row.items()} for row in reader]
    logger.info(f"Loaded {len(rows)} rows from {path.name}")
    return rows


def _index_by(rows: list[dict], key: str) -> dict[str, dict]:
    """Index a list of row dicts by a single key field. Last write wins for duplicates."""
    return {row[key]: row for row in rows if key in row and row[key]}


def _group_by(rows: list[dict], key: str) -> dict[str, list[dict]]:
    """Group rows by a key field, returning a dict of key → list of rows."""
    result: dict[str, list[dict]] = {}
    for row in rows:
        k = row.get(key, "")
        if k:
            result.setdefault(k, []).append(row)
    return result


class DataStore:
    """
    Central data store that loads all CSV files and provides indexed access.

    Usage:
        store = DataStore()
        store.load_all()
        user = store.get_user("u_001")
        group = store.get_group("group_002")
    """

    def __init__(self, dataset_dir: Path | str | None = None):
        self.dataset_dir = Path(dataset_dir) if dataset_dir else DATASET_DIR
        # Raw row lists
        self.messages_raw: list[dict] = []
        self.sample_messages_raw: list[dict] = []
        self.users_raw: list[dict] = []
        self.groups_raw: list[dict] = []
        self.group_members_raw: list[dict] = []
        self.business_accounts_raw: list[dict] = []
        self.user_business_history_raw: list[dict] = []
        self.message_history_raw: list[dict] = []
        self.message_events_raw: list[dict] = []
        self.images_raw: list[dict] = []
        self.voice_notes_raw: list[dict] = []
        self.daily_notification_summary_raw: list[dict] = []

        # Indexed lookups (populated by load_all)
        self.users: dict[str, dict] = {}
        self.groups: dict[str, dict] = {}
        self.business_accounts: dict[str, dict] = {}
        self.images: dict[str, dict] = {}
        self.voice_notes: dict[str, dict] = {}

        # Grouped lookups
        self.group_members_by_group: dict[str, list[dict]] = {}
        self.group_members_by_user: dict[str, list[dict]] = {}
        self.user_business_by_user: dict[str, list[dict]] = {}
        self.user_business_by_business: dict[str, list[dict]] = {}
        self.message_history_by_user: dict[str, list[dict]] = {}
        self.message_history_by_sender: dict[str, list[dict]] = {}
        self.message_events_by_user: dict[str, list[dict]] = {}
        self.message_events_by_message: dict[str, list[dict]] = {}
        self.daily_summary_by_user: dict[str, list[dict]] = {}

        # Composite lookups
        self.group_membership: dict[str, dict] = {}  # (group_id, user_id) → membership row
        self.user_business_relation: dict[str, dict] = {}  # (user_id, business_id) → relation row
        self.message_history_index: dict[str, dict] = {}  # message_id → history row

    def load_all(self) -> None:
        """Load all CSV files and build indices."""
        logger.info("Loading all dataset files...")

        # Load raw data
        self.messages_raw = _read_csv(self.dataset_dir / CSV_FILES["messages"].name)
        self.sample_messages_raw = _read_csv(self.dataset_dir / CSV_FILES["sample_messages"].name)
        self.users_raw = _read_csv(self.dataset_dir / CSV_FILES["users"].name)
        self.groups_raw = _read_csv(self.dataset_dir / CSV_FILES["groups"].name)
        self.group_members_raw = _read_csv(self.dataset_dir / CSV_FILES["group_members"].name)
        self.business_accounts_raw = _read_csv(self.dataset_dir / CSV_FILES["business_accounts"].name)
        self.user_business_history_raw = _read_csv(self.dataset_dir / CSV_FILES["user_business_history"].name)
        self.message_history_raw = _read_csv(self.dataset_dir / CSV_FILES["message_history"].name)
        self.message_events_raw = _read_csv(self.dataset_dir / CSV_FILES["message_events"].name)
        self.images_raw = _read_csv(self.dataset_dir / CSV_FILES["images"].name)
        self.voice_notes_raw = _read_csv(self.dataset_dir / CSV_FILES["voice_notes"].name)
        self.daily_notification_summary_raw = _read_csv(
            self.dataset_dir / CSV_FILES["daily_notification_summary"].name
        )

        # Build primary key indices
        self.users = _index_by(self.users_raw, "user_id")
        self.groups = _index_by(self.groups_raw, "group_id")
        self.business_accounts = _index_by(self.business_accounts_raw, "business_id")
        self.images = _index_by(self.images_raw, "image_id")
        self.voice_notes = _index_by(self.voice_notes_raw, "voice_note_id")
        self.message_history_index = _index_by(self.message_history_raw, "message_id")

        # Build grouped indices
        self.group_members_by_group = _group_by(self.group_members_raw, "group_id")
        self.group_members_by_user = _group_by(self.group_members_raw, "user_id")
        self.user_business_by_user = _group_by(
            self.user_business_history_raw, "user_id"
        )
        self.user_business_by_business = _group_by(
            self.user_business_history_raw, "business_id"
        )
        self.message_history_by_user = _group_by(self.message_history_raw, "user_id")
        self.message_history_by_sender = _group_by(
            self.message_history_raw, "sender_user_id"
        )
        self.message_events_by_user = _group_by(self.message_events_raw, "user_id")
        self.message_events_by_message = _group_by(
            self.message_events_raw, "message_id"
        )
        self.daily_summary_by_user = _group_by(
            self.daily_notification_summary_raw, "user_id"
        )

        # Build composite key lookups
        for row in self.group_members_raw:
            key = f"{row.get('group_id', '')}:{row.get('user_id', '')}"
            self.group_membership[key] = row

        for row in self.user_business_history_raw:
            key = f"{row.get('user_id', '')}:{row.get('business_id', '')}"
            self.user_business_relation[key] = row

        logger.info(
            f"Data loaded: {len(self.messages_raw)} messages to route, "
            f"{len(self.users)} users, {len(self.groups)} groups, "
            f"{len(self.business_accounts)} businesses, "
            f"{len(self.message_history_index)} history messages"
        )

    # ----- Convenience accessors -----

    def get_user(self, user_id: str) -> dict | None:
        """Get user profile by user_id."""
        return self.users.get(user_id)

    def get_group(self, group_id: str) -> dict | None:
        """Get group info by group_id."""
        return self.groups.get(group_id)

    def get_business(self, business_id: str) -> dict | None:
        """Get business account by business_id."""
        return self.business_accounts.get(business_id)

    def get_group_member(self, user_id: str, group_id: str) -> dict | None:
        """Get the membership record for a user in a specific group."""
        return self.group_membership.get(f"{group_id}:{user_id}")

    def get_user_business_relation(
        self, user_id: str, business_id: str
    ) -> dict | None:
        """Get the relationship between a user and a business."""
        return self.user_business_relation.get(f"{user_id}:{business_id}")

    def get_user_history(self, user_id: str) -> list[dict]:
        """Get all historical messages for a user."""
        return self.message_history_by_user.get(user_id, [])

    def get_sender_history(self, sender_user_id: str) -> list[dict]:
        """Get all historical messages from a specific sender."""
        return self.message_history_by_sender.get(sender_user_id, [])

    def get_message_events(self, message_id: str) -> list[dict]:
        """Get user reaction events for a specific historical message."""
        return self.message_events_by_message.get(message_id, [])

    def get_user_events(self, user_id: str) -> list[dict]:
        """Get all message events for a user."""
        return self.message_events_by_user.get(user_id, [])

    def get_daily_summary(self, user_id: str) -> list[dict]:
        """Get daily notification summaries for a user."""
        return self.daily_summary_by_user.get(user_id, [])

    def get_image_path(self, media_id: str) -> Path | None:
        """Get the full filesystem path for an image by its media_id."""
        img = self.images.get(media_id)
        if img and img.get("file_path"):
            return DATASET_DIR / img["file_path"]
        return None

    def get_voice_note_path(self, media_id: str) -> Path | None:
        """Get the full filesystem path for a voice note by its media_id."""
        vn = self.voice_notes.get(media_id)
        if vn and vn.get("file_path"):
            return DATASET_DIR / vn["file_path"]
        return None

    def get_history_message(self, message_id: str) -> dict | None:
        """Get a specific message from message_history by its message_id."""
        return self.message_history_index.get(message_id)

    def validate_evidence_id(self, evidence_id: str) -> bool:
        """Check whether an evidence message_id exists in message_history."""
        return evidence_id in self.message_history_index
