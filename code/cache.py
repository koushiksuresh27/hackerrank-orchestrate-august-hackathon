"""
Cache — LLM response and context caching.

Caches routing decisions keyed by message_id to:
1. Avoid re-calling APIs on reruns
2. Support incremental development (modify prompt, only re-process failures)

Uses simple JSON file storage in .cache/ directory.
"""

import json
import logging
from pathlib import Path
from typing import Any

from config import CACHE_DIR, ENABLE_CACHE

logger = logging.getLogger(__name__)


class ResponseCache:
    """Simple file-based cache for LLM routing decisions."""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.enabled = ENABLE_CACHE
        self._memory_cache: dict[str, dict] = {}
        self._loaded = False

    def _ensure_dir(self) -> None:
        """Create cache directory if it doesn't exist."""
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_file(self) -> Path:
        return self.cache_dir / "routing_cache.json"

    def _load_from_disk(self) -> None:
        """Load cache from disk into memory."""
        if self._loaded:
            return

        self._ensure_dir()
        cache_file = self._cache_file()

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    self._memory_cache = json.load(f)
                logger.info(f"Loaded {len(self._memory_cache)} cached decisions")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache: {e}")
                self._memory_cache = {}

        self._loaded = True

    def get(self, message_id: str) -> dict | None:
        """Get a cached decision for a message_id."""
        if not self.enabled:
            return None

        self._load_from_disk()
        return self._memory_cache.get(message_id)

    def put(self, message_id: str, decision: dict) -> None:
        """Cache a routing decision."""
        if not self.enabled:
            return

        self._load_from_disk()
        self._memory_cache[message_id] = decision

    def save_to_disk(self) -> None:
        """Persist the in-memory cache to disk."""
        if not self.enabled:
            return

        self._ensure_dir()

        try:
            with open(self._cache_file(), "w", encoding="utf-8") as f:
                json.dump(self._memory_cache, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self._memory_cache)} decisions to cache")
        except IOError as e:
            logger.warning(f"Failed to save cache: {e}")

    def clear(self) -> None:
        """Clear all cached decisions."""
        self._memory_cache = {}
        cache_file = self._cache_file()
        if cache_file.exists():
            cache_file.unlink()
        logger.info("Cache cleared")

    def size(self) -> int:
        """Return the number of cached decisions."""
        self._load_from_disk()
        return len(self._memory_cache)
