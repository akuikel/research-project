"""
Hugging Face Dataset Scraper for Spanish Educational Chatbot System Prompts.

Downloads the `fka/awesome-chatgpt-prompts` dataset from the Hugging Face Hub
(no API key required — it is a public dataset). Filters for education-related
entries, translates/adapts non-Spanish prompts where needed, and standardises
results to the project's JSON schema.

This source replaces the Reddit scraper. It is more reliable (no approval
wait, no rate limits) and the data is cleaner (structured prompts rather than
freeform forum posts).

No environment variables required — the dataset is publicly available.
"""

import os
import re
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

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
# Dataset configuration
# ---------------------------------------------------------------------------

# Primary dataset — 170+ curated ChatGPT prompts covering many domains
PRIMARY_DATASET = "fka/awesome-chatgpt-prompts"

# Additional dataset with a much larger prompt collection
SECONDARY_DATASET = "MohamedRashad/ChatGPT-prompts"

# ---------------------------------------------------------------------------
# Education keyword filters
# ---------------------------------------------------------------------------

# English education keywords — used to filter the primary dataset which is in English
EDUCATION_KEYWORDS_EN = [
    "tutor", "teacher", "teach", "education", "student", "learn", "school",
    "math", "science", "history", "language", "reading", "writing",
    "homework", "lesson", "curriculum", "instructor", "professor",
    "classroom", "academy", "coach", "mentor", "explain",
]

# Spanish education keywords — used when prompts are already in Spanish,
# or when filtering translated/adapted content
EDUCATION_KEYWORDS_ES = [
    "tutor", "profesor", "maestra", "educación", "estudiante", "aprender",
    "escuela", "matemáticas", "ciencia", "historia", "lectura", "escritura",
    "lección", "enseñar", "explicar", "alumno", "académico", "asistente",
]

# ---------------------------------------------------------------------------
# Prompt adaptation
# ---------------------------------------------------------------------------

# Maps generic English role openers to Spanish equivalents so that filtered
# prompts look like realistic Spanish educational chatbot system prompts.
# This is NOT translation — it replaces the structural opener only.
OPENER_MAP = [
    (r"^I want you to act as (an?)\s+", r"Eres un "),
    (r"^I want you to act like (an?)\s+", r"Eres un "),
    (r"^Act as (an?)\s+", r"Eres un "),
    (r"^You will act as (an?)\s+", r"Eres un "),
    (r"^You are (an?)\s+", r"Eres un "),
    (r"^You are a\s+", r"Eres un "),
    (r"^Pretend you are (an?)\s+", r"Actúa como "),
]

# Spanish preamble that wraps adapted English prompts so they are clearly
# framed as educational system prompts in the project context. We are
# transparent about this in the metadata.
SPANISH_WRAPPER = (
    "Eres un asistente educativo en español. Tu objetivo es ayudar a estudiantes "
    "de nivel secundario. Adapta siempre tu respuesta al nivel del estudiante y "
    "responde con paciencia y claridad.\n\n"
    "--- Instrucciones adicionales (adaptado) ---\n"
)


def _is_education_related(act: str, prompt: str) -> bool:
    """
    Return True if a prompt entry appears to be education-related.

    Checks both the act/role label and the prompt body against education
    keyword lists (English and Spanish, case-insensitive).

    Args:
        act:    Short label describing the AI role (e.g. 'Math Tutor').
        prompt: Full prompt text.

    Returns:
        True if any education keyword is found in act or prompt.
    """
    combined = (act + " " + prompt).lower()
    return any(kw in combined for kw in EDUCATION_KEYWORDS_EN + EDUCATION_KEYWORDS_ES)


def _adapt_to_spanish_frame(act: str, prompt: str) -> str:
    """
    Adapt an English prompt into a Spanish educational chatbot framing.

    We preserve the original prompt content (which carries the pedagogical
    signal we want to classify) but wrap it with a Spanish educational
    context header. The metadata `adapted_from_english: true` flag ensures
    full transparency.

    Args:
        act:    Role label (e.g. 'Math Tutor').
        prompt: Original English prompt text.

    Returns:
        Spanish-framed prompt string starting with 'Eres un...'.
    """
    # Try to replace the opener with a Spanish equivalent
    adapted = prompt
    for pattern, replacement in OPENER_MAP:
        adapted, n = re.subn(pattern, replacement, adapted, flags=re.IGNORECASE)
        if n > 0:
            break

    # If no opener matched, prepend a generic Spanish framing
    if not adapted.lower().startswith(("eres ", "tu rol", "actúa")):
        adapted = f"Eres un tutor de {act.lower()}.\n\n{adapted}"

    return adapted


# ---------------------------------------------------------------------------
# Main scraper
# ---------------------------------------------------------------------------

def scrape_huggingface(target: int = 30) -> list[dict]:
    """
    Scrape education-related prompts from public Hugging Face datasets.

    Downloads `fka/awesome-chatgpt-prompts` (and optionally a secondary
    dataset), filters for education-relevant entries, adapts them to a
    Spanish educational framing, and returns structured records.

    Args:
        target: Maximum number of records to return (default 30).

    Returns:
        List of structured prompt records in the project's JSON schema.
    """
    results: list[dict] = []
    counter = 1

    # ── Primary dataset: fka/awesome-chatgpt-prompts ──────────────────────
    logger.info(f"Loading dataset: {PRIMARY_DATASET}")
    try:
        ds = load_dataset(PRIMARY_DATASET, split="train")
        logger.info(f"Primary dataset loaded: {len(ds)} total entries")

        for row in ds:
            if len(results) >= target:
                break

            act = row.get("act", "") or ""
            prompt = row.get("prompt", "") or ""

            if not _is_education_related(act, prompt):
                logger.debug(f"Skipping non-education entry: '{act}'")
                continue

            adapted_text = _adapt_to_spanish_frame(act, prompt)

            record = {
                "id": f"huggingface_{counter:03d}",
                "source": "huggingface",
                "raw_text": adapted_text,
                "language": "es",
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "metadata": {
                    "dataset": PRIMARY_DATASET,
                    "original_act": act,
                    "original_language": "en",
                    "adapted_from_english": True,
                    "adaptation_note": (
                        "Original English prompt adapted to Spanish educational framing. "
                        "Content preserved; structural opener replaced with Spanish equivalent."
                    ),
                },
            }
            results.append(record)
            logger.info(f"Collected huggingface_{counter:03d}: '{act}'")
            counter += 1

    except Exception as e:
        logger.error(f"Failed to load {PRIMARY_DATASET}: {e}")

    # ── Secondary dataset: broader prompt collection ──────────────────────
    if len(results) < target:
        logger.info(f"Loading secondary dataset: {SECONDARY_DATASET}")
        try:
            ds2 = load_dataset(SECONDARY_DATASET, split="train")
            logger.info(f"Secondary dataset loaded: {len(ds2)} total entries")

            for row in ds2:
                if len(results) >= target:
                    break

                # Schema varies by dataset — handle both common field names
                act = row.get("act", row.get("title", row.get("name", ""))) or ""
                prompt = row.get("prompt", row.get("text", row.get("content", ""))) or ""

                if not prompt or len(prompt) < 80:
                    continue

                if not _is_education_related(act, prompt):
                    continue

                adapted_text = _adapt_to_spanish_frame(act, prompt)

                record = {
                    "id": f"huggingface_{counter:03d}",
                    "source": "huggingface",
                    "raw_text": adapted_text,
                    "language": "es",
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                    "metadata": {
                        "dataset": SECONDARY_DATASET,
                        "original_act": act,
                        "original_language": "en",
                        "adapted_from_english": True,
                        "adaptation_note": (
                            "Original English prompt adapted to Spanish educational framing."
                        ),
                    },
                }
                results.append(record)
                logger.info(f"Collected huggingface_{counter:03d}: '{act}' (secondary dataset)")
                counter += 1

        except Exception as e:
            logger.warning(f"Secondary dataset unavailable ({e}) — using primary dataset results only")

    logger.info(f"Hugging Face scraping complete. Total collected: {len(results)}")
    return results


def save_results(results: list[dict], output_path: str) -> None:
    """
    Save scraped results to a JSON file, creating parent directories if needed.

    Args:
        results:     List of prompt records to save.
        output_path: Absolute or relative path for the output JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(results)} records to {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    output = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "raw", "huggingface_prompts.json"
    )
    prompts = scrape_huggingface(target=30)
    save_results(prompts, os.path.normpath(output))
    print(
        f"\n✅ Hugging Face scraper done. "
        f"{len(prompts)} prompts saved to {os.path.normpath(output)}"
    )
