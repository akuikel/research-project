"""
Prompt Classifier for Spanish Educational Chatbot System Prompts.

Classifies prompts across 6 research dimensions using keyword matching
(subject domain, cultural references) and Gemini 2.0 Flash via google-genai
SDK (tone, scaffolding depth, motivational strategies). Complexity metrics
computed via textstat.

Required environment variables:
    GOOGLE_API_KEY — Free key from https://aistudio.google.com/app/apikey
"""

import os
import re
import json
import logging
import time
from typing import Optional

import textstat
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

_gemini_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    """Return a cached Gemini client, initialising on first use."""
    global _gemini_client
    if _gemini_client is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY not found.")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


# ---------------------------------------------------------------------------
# Dimension 1: Subject Domain
# ---------------------------------------------------------------------------
SUBJECT_KEYWORDS: dict[str, list[str]] = {
    "matemáticas": ["matemáticas", "matemática", "números", "ecuación", "cálculo",
                    "álgebra", "geometría", "aritmética", "suma", "resta",
                    "multiplicación", "división", "fracciones", "porcentaje"],
    "lectura":     ["lectura", "comprensión lectora", "leer", "texto", "libro",
                    "literatura", "narrativa", "poesía", "cuento", "novela",
                    "párrafo", "vocabulario"],
    "historia":    ["historia", "histórico", "época", "civilización", "guerra",
                    "revolución", "reinado", "siglo", "conquista", "independencia",
                    "rey", "reina", "españa"],
    "ciencias":    ["ciencias", "ciencia", "biología", "física", "química",
                    "naturaleza", "experimento", "célula", "átomo", "ecosistema",
                    "planta", "animal", "energía"],
    "lengua":      ["lengua", "gramática", "ortografía", "redacción", "escritura",
                    "español", "castellano", "conjugación", "verbo", "sustantivo",
                    "adjetivo", "puntuación"],
}


def _classify_subject_domain(text: str) -> str:
    """Classify subject domain using keyword matching; fall back to Gemini."""
    text_lower = text.lower()
    scores = {cat: sum(1 for kw in kws if kw in text_lower)
              for cat, kws in SUBJECT_KEYWORDS.items()}
    best = max(scores, key=lambda c: scores[c])
    if scores[best] > 0:
        return best
    return _classify_subject_llm(text)


def _classify_subject_llm(text: str) -> str:
    """Use Gemini to classify subject domain when keywords don't match."""
    valid = {"matemáticas", "lectura", "historia", "ciencias", "lengua", "otro"}
    try:
        client = _get_client()
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=(
                "Classify the subject domain of this Spanish educational chatbot prompt. "
                "Respond with ONE word only from: matemáticas, lectura, historia, ciencias, lengua, otro\n\n"
                f"Prompt:\n{text[:500]}"
            ),
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=10,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        raw = response.text
        if not raw:
            return "otro"
        result = raw.strip().lower()
        return result if result in valid else "otro"
    except Exception as e:
        logger.error(f"Gemini subject classification failed: {e}")
        return "otro"


# ---------------------------------------------------------------------------
# Dimension 5: Cultural References
# ---------------------------------------------------------------------------
CULTURAL_KEYWORDS = [
    "gitano", "gitana", "roma", "romaní", "cultura", "comunidad",
    "identidad", "diversidad", "inclusión", "inclusivo", "multicultural",
    "minoría", "etnias", "patrimonio", "hispano", "hispanohablante",
]


def _classify_cultural_references(text: str) -> str:
    """Detect cultural references via keyword matching."""
    text_lower = text.lower()
    return "presente" if any(kw in text_lower for kw in CULTURAL_KEYWORDS) else "ausente"


# ---------------------------------------------------------------------------
# Dimension 6: Complexity
# ---------------------------------------------------------------------------
def _compute_complexity_metrics(text: str) -> dict:
    """Compute Flesch reading ease and related metrics via textstat."""
    textstat.set_lang("es")
    word_count = len(text.split())
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    avg_sentence_length = round(word_count / max(len(sentences), 1), 1)
    return {
        "flesch_reading_ease": round(textstat.flesch_reading_ease(text), 1),
        "syllable_count": textstat.syllable_count(text),
        "word_count": word_count,
        "avg_sentence_length": avg_sentence_length,
    }


# ---------------------------------------------------------------------------
# LLM multi-dimension classifier (single Gemini call for 4 dimensions)
# ---------------------------------------------------------------------------
LLM_PROMPT_TEMPLATE = """\
Analyze this Spanish educational chatbot system prompt and respond with ONLY a raw JSON object. \
No explanation, no markdown, no code fences. Start your response with {{ and end with }}.

{{"tone": "<formal|calido|motivador|neutral>", "scaffolding_depth": "<bajo|medio|alto>", "motivational_strategies": "<ninguna|elogio|metas|crecimiento|multiple>", "cultural_references": "<presente|ausente>"}}

Replace each placeholder with the correct value for this prompt:
{prompt_text}"""

VALID_LLM_VALUES = {
    "tone": {"formal", "calido", "motivador", "neutral"},
    "scaffolding_depth": {"bajo", "medio", "alto"},
    "motivational_strategies": {"ninguna", "elogio", "metas", "crecimiento", "multiple"},
    "cultural_references": {"presente", "ausente"},
}


def _extract_json(raw: str) -> dict | None:
    """Extract a JSON object from a string using regex — handles markdown-wrapped responses."""
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw.strip(), flags=re.MULTILINE)
    # Try to find a {...} block
    match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _classify_with_llm(text: str, max_retries: int = 2) -> dict:
    """
    Classify tone, scaffolding, motivation, and cultural refs in one Gemini call.

    Uses a direct prompt that forbids preamble text. Regex extraction handles
    any residual markdown wrapping from the model.

    Args:
        text:        System prompt text.
        max_retries: Retry attempts on failure.

    Returns:
        Dict with all LLM-classified fields; values are None on failure.
    """
    null_result = {"tone": None, "scaffolding_depth": None,
                   "motivational_strategies": None, "cultural_references": None}

    prompt = LLM_PROMPT_TEMPLATE.format(prompt_text=text[:1200])

    for attempt in range(1, max_retries + 1):
        try:
            client = _get_client()
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=2048,
                ),
            )

            raw = response.text
            if not raw:
                logger.warning(f"Empty response from Gemini on attempt {attempt}/{max_retries}")
                time.sleep(1)
                continue

            # Try direct parse first, then regex extraction
            parsed = None
            try:
                parsed = json.loads(raw.strip())
            except json.JSONDecodeError:
                parsed = _extract_json(raw)
                if parsed:
                    logger.debug(f"Used regex fallback to extract JSON on attempt {attempt}")

            if parsed is None:
                logger.warning(f"JSON parse failed attempt {attempt}/{max_retries} — raw: {raw[:80]!r}")
                time.sleep(1)
                continue

            # Null out any values outside allowed categories
            for field, valid_set in VALID_LLM_VALUES.items():
                if parsed.get(field) not in valid_set:
                    logger.warning(f"Unexpected '{field}': {parsed.get(field)!r} — setting null")
                    parsed[field] = None

            return parsed

        except Exception as e:
            logger.warning(f"Gemini API error attempt {attempt}/{max_retries}: {e}")
            time.sleep(2 ** attempt)

    logger.error("All LLM classification retries failed.")
    return null_result


# ---------------------------------------------------------------------------
# Main classification function
# ---------------------------------------------------------------------------
def classify_prompt(text: str, prompt_id: str = "") -> dict:
    """
    Classify a Spanish educational chatbot system prompt across 6 dimensions.

    Args:
        text:      Raw system prompt text.
        prompt_id: Optional ID for logging.

    Returns:
        Dict with all classification fields merged.
    """
    log_prefix = f"[{prompt_id}] " if prompt_id else ""
    logger.info(f"{log_prefix}Classifying ({len(text)} chars)...")

    subject_domain = _classify_subject_domain(text)
    cultural_keyword = _classify_cultural_references(text)
    llm = _classify_with_llm(text)
    complexity = _compute_complexity_metrics(text)

    # Keyword wins if it found cultural refs that Gemini missed
    final_cultural = llm.get("cultural_references") or cultural_keyword
    if cultural_keyword == "presente" and llm.get("cultural_references") == "ausente":
        final_cultural = "presente"

    result = {
        "id": prompt_id,
        "subject_domain": subject_domain,
        "tone": llm.get("tone"),
        "scaffolding_depth": llm.get("scaffolding_depth"),
        "motivational_strategies": llm.get("motivational_strategies"),
        "cultural_references": final_cultural,
        **complexity,
    }
    logger.info(f"{log_prefix}Done — domain={subject_domain}, tone={result['tone']}, words={complexity['word_count']}")
    return result


if __name__ == "__main__":
    test_prompt = (
        "Eres un tutor de matemáticas amigable y paciente para estudiantes de secundaria. "
        "Ayuda con preguntas guiadas, celebra los logros y divide cada problema en pasos pequeños."
    )
    result = classify_prompt(test_prompt.strip(), prompt_id="test_001")
    print("\n=== Classification Result ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
