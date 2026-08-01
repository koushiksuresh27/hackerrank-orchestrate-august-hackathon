"""
Cache — per-message JSON file cache for the classification pipeline.

Saves each result as cache/{message_id}.json immediately after classification.
One file per message, so a crashed run loses only the current message —
all previously classified messages are safe on disk.
"""

import csv
import json
import logging
import os
from pathlib import Path

from config import CACHE_DIR

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
]


class ResponseCache:
    """
    Per-message file-based cache for LLM routing decisions.

    Each result is written to cache/{message_id}.json immediately after
    classification. On startup, all existing files are loaded into memory
    so is_cached() and load() are O(1) without hitting disk each time.

    Args:
        cache_dir: Directory to store cache files. Defaults to config.CACHE_DIR.
        force_rerun: List of message_ids to always re-classify, even if cached.
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        force_rerun: list[str] | None = None,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.force_rerun: set[str] = set(force_rerun or [])
        self._memory: dict[str, dict] = {}

        # Populate in-memory index from disk on startup
        self._load_all_from_disk()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path_for(self, message_id: str) -> Path:
        """Return the JSON file path for a given message_id."""
        return self.cache_dir / f"{message_id}.json"

    def _load_all_from_disk(self) -> None:
        """Load all existing cache files into _memory on startup."""
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            return

        loaded = 0
        for fpath in self.cache_dir.glob("*.json"):
            message_id = fpath.stem
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    self._memory[message_id] = json.load(f)
                loaded += 1
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache file {fpath.name}: {e}")

        if loaded:
            logger.info(f"Loaded {loaded} cached decisions from {self.cache_dir}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_cached(self, message_id: str) -> bool:
        """
        Return True if a valid cache entry exists for message_id.

        Always returns False for message_ids in the force_rerun list.
        """
        if message_id in self.force_rerun:
            return False
        return message_id in self._memory

    def load(self, message_id: str) -> dict | None:
        """Return the cached result for message_id, or None if not cached."""
        if message_id in self.force_rerun:
            return None
        return self._memory.get(message_id)

    def save(self, message_id: str, result: dict) -> None:
        """
        Write result to disk immediately as cache/{message_id}.json
        and update the in-memory index.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        fpath = self._path_for(message_id)
        try:
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            self._memory[message_id] = result
        except IOError as e:
            logger.warning(f"Failed to write cache for {message_id}: {e}")

    def load_all(self) -> dict:
        """Return a shallow copy of all in-memory cached decisions."""
        return dict(self._memory)

    def clear(self, message_id: str) -> None:
        """Delete the cache file for a single message_id."""
        self._memory.pop(message_id, None)
        fpath = self._path_for(message_id)
        if fpath.exists():
            fpath.unlink()
            logger.debug(f"Cleared cache for {message_id}")

    def clear_all(self) -> None:
        """Delete all cache files in the cache directory."""
        count = 0
        for fpath in self.cache_dir.glob("*.json"):
            try:
                fpath.unlink()
                count += 1
            except IOError as e:
                logger.warning(f"Failed to delete {fpath.name}: {e}")
        self._memory.clear()
        logger.info(f"Cleared {count} cache files from {self.cache_dir}")

    def size(self) -> int:
        """Return the number of cached decisions."""
        return len(self._memory)


# ------------------------------------------------------------------
# Standalone output function
# ------------------------------------------------------------------


def write_final_output(
    messages_df,
    cache: ResponseCache,
    fallback_fn,
    output_path: Path | str,
) -> None:
    """
    Write a complete output.csv from cached results + fallback for missing rows.

    Iterates every row in messages_df. For each message_id:
      - If cached → use cache.load()
      - Else → call fallback_fn(row) to get a result

    This guarantees output.csv is always complete even after an interrupted run.

    Args:
        messages_df: Iterable of row dicts (e.g. list[dict] from DataStore).
        cache: ResponseCache instance.
        fallback_fn: Callable(row: dict) -> dict. Must always return a valid
            result dict with all required keys.
        output_path: Path to write the output CSV file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    fallback_used = 0

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(OUTPUT_COLUMNS)

        for row in messages_df:
            message_id = (row.get("message_id") or "").strip()

            if cache.is_cached(message_id):
                result = cache.load(message_id)
            else:
                result = fallback_fn(row)
                fallback_used += 1

            writer.writerow([
                message_id,
                result.get("action", "digest"),
                result.get("message_type", "unknown"),
                result.get("reason", "No decision available."),
                str(result.get("confidence", 0.72)),
                result.get("evidence_message_ids", "none"),
            ])
            rows_written += 1

    logger.info(
        f"write_final_output: {rows_written} rows written to {output_path} "
        f"({fallback_used} via fallback)"
    )
