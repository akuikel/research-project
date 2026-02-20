"""
Main Pipeline for Spanish Educational Chatbot Prompt Analyzer.

This orchestration script:
1. Loads all raw scraped/synthetic JSON files from data/raw/
2. Deduplicates prompts by text similarity (MD5 hashing)
3. Classifies each prompt across 6 research dimensions
4. Saves checkpoints every 10 prompts for crash recovery
5. Exports results as JSON and CSV

The pipeline is resumable — if interrupted, re-running it picks up from the
last saved checkpoint rather than re-classifying already-processed prompts.
This is critical because LLM classification has a real API cost.
"""

import os
import sys
import csv
import json
import glob
import logging
import hashlib
from datetime import datetime, timezone
from typing import Optional

# Make sure src/ is on the path when running from the project root
sys.path.insert(0, os.path.dirname(__file__))

from classify_prompt import classify_prompt

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path configuration — all paths relative to this file
# ---------------------------------------------------------------------------
BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
CHECKPOINT_FILE = os.path.join(PROCESSED_DIR, "checkpoint.json")
OUTPUT_JSON = os.path.join(PROCESSED_DIR, "classified_prompts.json")
OUTPUT_CSV = os.path.join(PROCESSED_DIR, "classified_prompts.csv")

# How often to write a checkpoint (every N prompts)
CHECKPOINT_INTERVAL = 10


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_prompts() -> list[dict]:
    """
    Load and combine all JSON files from the raw data directory.

    Each JSON file is expected to contain a list of prompt records in the
    project's standard schema. Files from different sources (github, reddit,
    synthetic) are merged into a single flat list.

    Returns:
        Combined list of raw prompt records.
    """
    all_prompts: list[dict] = []
    json_files = glob.glob(os.path.join(RAW_DIR, "*.json"))

    if not json_files:
        logger.warning(f"No JSON files found in {RAW_DIR}")
        return []

    for filepath in sorted(json_files):
        filename = os.path.basename(filepath)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                all_prompts.extend(data)
                logger.info(f"Loaded {len(data)} prompts from {filename}")
            else:
                logger.warning(f"Unexpected format in {filename} — expected list, got {type(data)}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error in {filename}: {e}")
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")

    logger.info(f"Total prompts loaded: {len(all_prompts)}")
    return all_prompts


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _text_hash(text: str) -> str:
    """
    Compute an MD5 hash of the prompt text for deduplication.

    We normalise whitespace before hashing so that minor formatting differences
    (e.g. trailing spaces, different line endings) don't create false duplicates.
    This gives us 'exact or near-exact' deduplication without requiring a
    computationally expensive fuzzy-similarity pass.

    Args:
        text: Raw prompt text.

    Returns:
        Hexadecimal MD5 hash string.
    """
    normalised = " ".join(text.lower().split())
    return hashlib.md5(normalised.encode("utf-8")).hexdigest()


def deduplicate(prompts: list[dict]) -> list[dict]:
    """
    Remove duplicate prompts based on text content similarity (MD5 hashing).

    Two prompts are considered duplicates if their normalised text produces
    the same MD5 hash — this catches 100% identical texts and near-identical
    texts that differ only in whitespace/capitalisation.

    Args:
        prompts: List of raw prompt records.

    Returns:
        Deduplicated list with one record per unique text.
    """
    seen_hashes: set[str] = set()
    unique: list[dict] = []

    for prompt in prompts:
        text = prompt.get("raw_text", "")
        if not text:
            continue
        h = _text_hash(text)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(prompt)
        else:
            logger.debug(f"Duplicate removed: {prompt.get('id', 'unknown')}")

    removed = len(prompts) - len(unique)
    if removed > 0:
        logger.info(f"Deduplication: removed {removed} duplicates, {len(unique)} unique prompts remain")
    return unique


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def load_checkpoint() -> dict[str, dict]:
    """
    Load previously processed results from a checkpoint file.

    The checkpoint maps prompt IDs to their classification results. On startup,
    the pipeline checks this file and skips any prompts already present,
    allowing it to resume after a crash or interruption without re-classifying
    and paying for LLM calls twice.

    Returns:
        Dictionary mapping prompt_id → classification result dict.
        Empty dict if no checkpoint exists.
    """
    if not os.path.exists(CHECKPOINT_FILE):
        return {}

    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        logger.info(f"Checkpoint loaded: {len(checkpoint)} already-processed prompts found")
        return checkpoint
    except Exception as e:
        logger.warning(f"Could not load checkpoint ({e}) — starting fresh")
        return {}


def save_checkpoint(processed: dict[str, dict]) -> None:
    """
    Write current processing progress to the checkpoint file.

    Called every CHECKPOINT_INTERVAL prompts. Uses atomic write pattern
    (write to temp file, then rename) to prevent corruption if the process
    is killed mid-write.

    Args:
        processed: Dictionary mapping prompt_id → classification result dict.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    temp_path = CHECKPOINT_FILE + ".tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, CHECKPOINT_FILE)
        logger.debug(f"Checkpoint saved: {len(processed)} prompts")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def save_json(results: list[dict], path: str) -> None:
    """
    Save the full classified dataset as a JSON file.

    Args:
        results: List of fully classified prompt records.
        path:    Output file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON saved: {path} ({len(results)} records)")


def save_csv(results: list[dict], path: str) -> None:
    """
    Save the classified dataset as a flat CSV file (one row per prompt).

    Metadata fields (nested dict) are serialised as JSON strings in the CSV
    so that the file remains rectangular without losing information.

    Args:
        results: List of fully classified prompt records.
        path:    Output file path.
    """
    if not results:
        logger.warning("No results to save to CSV")
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Flatten nested 'metadata' dict into JSON string for CSV compatibility
    flat_results = []
    for r in results:
        flat = dict(r)
        if isinstance(flat.get("metadata"), dict):
            flat["metadata"] = json.dumps(flat["metadata"], ensure_ascii=False)
        flat_results.append(flat)

    # Collect all unique column names (preserving insertion order)
    all_keys: list[str] = []
    for row in flat_results:
        for k in row:
            if k not in all_keys:
                all_keys.append(k)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_results)

    logger.info(f"CSV saved: {path} ({len(flat_results)} rows, {len(all_keys)} columns)")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    """
    Execute the full end-to-end classification pipeline.

    Steps:
        1. Load all raw prompt files from data/raw/
        2. Deduplicate by text hash
        3. Load checkpoint to resume if available
        4. Classify each unprocessed prompt
        5. Checkpoint progress every CHECKPOINT_INTERVAL prompts
        6. Save final JSON and CSV outputs
        7. Print summary report
    """
    logger.info("=" * 60)
    logger.info("Starting Spanish Chatbot Prompt Classification Pipeline")
    logger.info("=" * 60)

    # Step 1: Load
    prompts = load_raw_prompts()
    if not prompts:
        logger.error("No raw prompts found. Run the scrapers first.")
        return

    # Step 2: Deduplicate
    prompts = deduplicate(prompts)
    total = len(prompts)
    logger.info(f"Processing {total} unique prompts")

    # Step 3: Load checkpoint (resume support)
    processed: dict[str, dict] = load_checkpoint()  # {id: result_dict}
    already_done = len(processed)
    if already_done > 0:
        logger.info(f"Resuming: {already_done}/{total} already classified, {total - already_done} remaining")

    # Step 4 & 5: Classify with checkpointing
    failed = 0

    for i, prompt in enumerate(prompts):
        prompt_id = prompt.get("id", f"unknown_{i}")

        # Skip already-processed prompts (resume logic)
        if prompt_id in processed:
            continue

        # Log timestamped progress
        current_count = len(processed) + 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] Processing {current_count}/{total} prompts... (id: {prompt_id})")

        try:
            classification = classify_prompt(prompt.get("raw_text", ""), prompt_id=prompt_id)
            # Merge: original prompt data + classification results
            # Classification fields override prompt fields (e.g. 'id' stays from prompt)
            merged = {**prompt, **classification}
            processed[prompt_id] = merged

        except Exception as e:
            logger.error(f"Classification failed for {prompt_id}: {e}")
            # Store original record with null classification to preserve the data
            processed[prompt_id] = {**prompt, "classification_error": str(e)}
            failed += 1

        # Checkpoint every N prompts to limit data loss on crash
        if len(processed) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(processed)

    # Final checkpoint
    save_checkpoint(processed)

    # Step 6: Save outputs
    results = list(processed.values())
    save_json(results, OUTPUT_JSON)
    save_csv(results, OUTPUT_CSV)

    # Step 7: Summary report
    successful = len(results) - failed
    source_counts: dict[str, int] = {}
    for r in results:
        source = r.get("source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Total prompts processed : {len(results)}")
    print(f"Successfully classified : {successful}")
    print(f"Failed classifications  : {failed}")
    print(f"\nBreakdown by source:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source:15s}: {count}")
    print(f"\nOutputs saved:")
    print(f"  JSON: {OUTPUT_JSON}")
    print(f"  CSV:  {OUTPUT_CSV}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
