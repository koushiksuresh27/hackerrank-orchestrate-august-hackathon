"""
Main — orchestrator for the Message Notification Router.

Pipeline: Load Data → For each message: Build Context → Extract Signals →
Process Media → Select Evidence → Route (LLM) → Validate → Write Output.

Usage:
    python main.py                  # Full run
    python main.py --evaluate       # Run + evaluate output
    python main.py --clear-cache    # Clear cached decisions and re-run
    python main.py --dry-run        # Run pipeline without LLM calls (rule-based only)
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Load env variables before any other imports
load_dotenv()

# Setup logging before imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")

from agent import RoutingAgent
from cache import ResponseCache, write_final_output
from config import OUTPUT_PATH
from context_builder import build_context
from data_loader import DataStore
from evidence_selector import select_evidence
from evaluator import evaluate_output, print_report
from media_handler import process_media
from output_writer import write_output
from provider_router import ProviderRouter
from signal_extractor import extract_signals, early_exit
from validator import validate_and_repair


def main():
    """Main entry point for the routing pipeline."""
    args = parse_args()

    # Configure log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("Message Notification Router — Starting")
    logger.info("=" * 60)

    start_time = time.time()

    # Phase 1: Load all data
    logger.info("Phase 1: Loading data...")
    store = DataStore()
    store.load_all()

    total_messages = len(store.messages_raw)
    logger.info(f"Messages to route: {total_messages}")

    # Phase 2: Initialize components
    logger.info("Phase 2: Initializing components...")
    cache = ResponseCache()

    if args.clear_cache:
        cache.clear_all()
        logger.info("Cache cleared")

    provider_router = ProviderRouter()
    agent = RoutingAgent(provider_router)

    # Phase 3: Process each message
    logger.info("Phase 3: Processing messages...")
    results: dict[str, dict] = {}
    cache_hits = 0
    llm_calls = 0
    signal_skips = 0
    errors = 0

    for i, message in enumerate(store.messages_raw):
        message_id = message.get("message_id", f"unknown_{i}")

        try:
            # Check cache first
            if cache.is_cached(message_id):
                results[message_id] = cache.load(message_id)
                cache_hits += 1
                continue

            # Step 1: Build context
            context = build_context(message, store)

            # Step 2: Extract signals
            signals = extract_signals(message, store)

            # Step 3: Process media (voice transcription, image encoding)
            media_result = process_media(message, store)

            # Step 4: Select evidence
            evidence_ids = select_evidence(message, context, store)

            # Step 5: Route message
            early_decision = early_exit(signals)
            if early_decision:
                early_decision["evidence_message_ids"] = "none"
                decision = early_decision
                signal_skips += 1
            elif args.dry_run:
                # Force rule-based fallback
                decision = agent._rule_based_fallback(message, context, signals)
                decision["evidence_message_ids"] = evidence_ids
                llm_calls += 1
            else:
                decision = agent.route_message(
                    message, context, signals, media_result, evidence_ids, store
                )
                llm_calls += 1

            # Step 6: Validate and repair
            validated = validate_and_repair(decision, message_id, store)

            # Save result immediately to disk (crash-safe)
            cache.save(message_id, validated)
            results[message_id] = validated

            time.sleep(2)  # 2 seconds between messages = 30/min max

            # Progress logging
            if (i + 1) % 10 == 0 or (i + 1) == total_messages:
                logger.info(
                    f"Progress: {i + 1}/{total_messages} "
                    f"(cache: {cache_hits}, LLM: {llm_calls}, "
                    f"signals: {signal_skips}, errors: {errors})"
                )

        except Exception as e:
            logger.error(f"[{message_id}] Processing failed: {e}", exc_info=True)
            errors += 1
            # Default safe decision
            results[message_id] = {
                "action": "digest",
                "message_type": "unknown",
                "reason": "Processing error. Defaulting to digest.",
                "confidence": 0.72,
                "evidence_message_ids": "none",
            }

    # Phase 5: Write output
    logger.info("Phase 5: Writing output...")
    
    messages_df = store.messages_raw
    write_final_output(
        messages_df=messages_df,
        cache=cache,
        fallback_fn=lambda row: agent._rule_based_fallback(
            row, 
            build_context(row, store),
            extract_signals(row, store)
        ),
        output_path=OUTPUT_PATH
    )
    success = True

    elapsed = time.time() - start_time
    logger.info(f"\nPipeline completed in {elapsed:.1f}s")
    logger.info(f"  Total: {total_messages}")
    logger.info(f"  Cache hits: {cache_hits}")
    logger.info(f"  LLM calls: {llm_calls}")
    logger.info(f"  Signal-based: {signal_skips}")
    logger.info(f"  Errors: {errors}")
    logger.info(f"  Output: {'[OK] Written' if success else '[FAIL] Failed'}")

    # Phase 6: Evaluate (optional)
    if args.evaluate:
        logger.info("\nPhase 6: Evaluating output...")
        report = evaluate_output(OUTPUT_PATH, store)
        print_report(report)

    return 0 if success else 1


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Message Notification Router — routes WhatsApp messages to notify/digest/mute"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation after processing",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cached decisions before processing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with rule-based fallback only (no LLM calls)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
