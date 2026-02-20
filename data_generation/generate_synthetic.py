"""
Synthetic Data Generator for Spanish Educational Chatbot System Prompts.

Uses the Google Gemini API (google-genai SDK, free via Google AI Studio) to generate
realistic Spanish educational chatbot system prompts across a controlled matrix
of subjects and pedagogical approaches.

Required environment variables:
    GOOGLE_API_KEY — Free key from https://aistudio.google.com/app/apikey
"""

import os
import json
import time
import logging
from datetime import datetime, timezone
from itertools import product
from typing import Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Generation matrix
# ---------------------------------------------------------------------------
SUBJECTS = [
    "matemáticas",
    "lectura y comprensión",
    "historia de España",
    "ciencias naturales",
    "lengua española",
]

APPROACHES = [
    "formal y estructurado",
    "cálido y motivador",
    "socrático (preguntas guiadas)",
    "culturalmente inclusivo (referencias a comunidades hispanohablantes)",
    "enfocado en scaffolding paso a paso",
    "orientado al crecimiento personal",
]

GENERATION_PROMPT_TEMPLATE = """\
Generate a Spanish-language system prompt for an educational AI chatbot designed \
to help middle school students in Spain learn {subject}. \
The prompt should reflect a {approach} pedagogical style.

Write ONLY the system prompt itself, starting with "Eres un..." or "Tu rol es...". \
Make it 150-300 words. Include specific instructions about:
- The chatbot's tone and personality
- How to handle student mistakes constructively
- How to encourage and motivate the student
- Specific techniques for teaching {subject}

Do not include any explanation, title, or commentary — just the system prompt text."""


def _get_client() -> genai.Client:
    """Return a configured Gemini API client."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY not found. Get a free key at https://aistudio.google.com/app/apikey"
        )
    return genai.Client(api_key=api_key)


def _generate_prompt(client: genai.Client, subject: str, approach: str, max_retries: int = 3) -> Optional[str]:
    """
    Generate a single synthetic Spanish educational chatbot system prompt.

    Args:
        client:      Configured Gemini client.
        subject:     Subject area for the chatbot tutor.
        approach:    Pedagogical approach to reflect.
        max_retries: Number of retry attempts on failure.

    Returns:
        Generated prompt text, or None if all attempts failed.
    """
    user_message = GENERATION_PROMPT_TEMPLATE.format(subject=subject, approach=approach)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_message,
                config=types.GenerateContentConfig(
                    temperature=0.9,
                    max_output_tokens=500,
                ),
            )
            text = response.text.strip()
            logger.info(f"Generated prompt for '{subject}' / '{approach}' ({len(text)} chars)")
            return text
        except Exception as e:
            wait = 2 ** attempt
            logger.warning(f"Attempt {attempt}/{max_retries} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)

    logger.error(f"All {max_retries} attempts failed for '{subject}' / '{approach}'")
    return None


def generate_synthetic_prompts(prompts_per_combination: int = 1) -> list[dict]:
    """
    Generate synthetic prompts for all subject × approach combinations.

    Default: 5 subjects × 6 approaches × 1 variant = 30 prompts.

    Args:
        prompts_per_combination: Variants per combination.

    Returns:
        List of structured prompt records in the project's JSON schema.
    """
    client = _get_client()
    results: list[dict] = []
    counter = 1
    combinations = list(product(SUBJECTS, APPROACHES))
    total = len(combinations) * prompts_per_combination
    logger.info(f"Generating {total} synthetic prompts ({len(combinations)} combinations × {prompts_per_combination} each)")

    for subject, approach in combinations:
        for variant in range(prompts_per_combination):
            logger.info(f"[{counter}/{total}] subject='{subject}', approach='{approach}'")
            text = _generate_prompt(client, subject, approach)

            if text is None:
                logger.warning(f"Skipping failed generation for '{subject}' / '{approach}'")
                counter += 1
                continue

            record = {
                "id": f"synthetic_{counter:03d}",
                "source": "synthetic",
                "raw_text": text,
                "language": "es",
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "metadata": {
                    "subject": subject,
                    "pedagogical_approach": approach,
                    "model": "gemini-2.5-flash",
                    "variant": variant + 1,
                    "generation_note": "AI-generated synthetic prompt for portfolio demonstration.",
                },
            }
            results.append(record)
            counter += 1
            
            # Rate limit is 15 requests per minute on the free tier (1 request every 4 seconds)
            # Sleep 4.1s to avoid 429 Too Many Requests errors
            time.sleep(4.1)

    logger.info(f"Synthetic generation complete. {len(results)}/{total} prompts generated successfully.")
    return results


def save_results(results: list[dict], output_path: str) -> None:
    """Save generated prompts to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(results)} synthetic prompts to {output_path}")


if __name__ == "__main__":
    output = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "synthetic_prompts.json")
    prompts = generate_synthetic_prompts(prompts_per_combination=1)
    save_results(prompts, os.path.normpath(output))
    print(f"\n✅ Synthetic generation done. {len(prompts)} prompts saved to {os.path.normpath(output)}")
