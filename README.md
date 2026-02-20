# Spanish Educational Chatbot Prompt Analyzer

##Hosted at: https://research-project-inky.vercel.app/ 

## Overview

This project builds a complete data pipeline for collecting, classifying, and analyzing Spanish-language educational chatbot system prompts. It scrapes prompts from GitHub and public Hugging Face datasets, supplements them with Gemini-generated synthetic examples, and classifies each prompt across six research-relevant dimensions using keyword matching and the free Gemini 1.5 Flash API. The result is a clean, structured dataset ready for quantitative analysis — demonstrating the exact skills required for academic NLP research on AI tutoring tools.

## Relevance to Educational Research

Teacher-designed chatbot system prompts encode explicit pedagogical choices: how to handle mistakes, how much scaffolding to provide, whether to use culturally inclusive language. In a Randomized Controlled Trial (RCT) like *"When Teachers Program the AI"*, classifying these prompts at scale reveals **what teachers actually do** when they configure AI tutors — and these choices likely explain variation in student outcomes. This pipeline mirrors that classification workflow, from weekly data collection to structured NLP output, making it directly applicable to the real research context.

## Repository Structure

```
spanish-chatbot-analyzer/
├── data/
│   ├── raw/                        # Raw collected prompts (JSON, one file per source)
│   └── processed/                  # Classified output (JSON + CSV) + checkpoint
├── src/
│   ├── scrapers/
│   │   ├── github_scraper.py       # GitHub Search API scraper (no auth needed)
│   │   └── huggingface_scraper.py  # Hugging Face public datasets scraper (no auth needed)
│   ├── classify_prompt.py          # 6-dimension classifier (keywords + Gemini 1.5 Flash)
│   ├── pipeline.py                 # Orchestration: load → deduplicate → classify → save
│   └── validate_data.py            # Post-classification data quality checks
├── notebooks/
│   ├── prompt_analysis.ipynb          # Exploratory analysis with 5 visualizations
│   └── interaction_log_analysis.ipynb # Student-chatbot interaction log analysis
├── data_generation/
│   └── generate_synthetic.py       # Gemini-generated synthetic prompts
├── .env.example                    # Template for required API keys
├── requirements.txt                # Pinned dependencies
└── README.md
```

## How to Run

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Copy the env template and fill in your keys
copy .env.example .env
```

Edit `.env`:
```
GITHUB_TOKEN=...    # Optional — raises rate limit from 60→5000 req/hr
GOOGLE_API_KEY=...  # Free — https://aistudio.google.com/app/apikey
# Hugging Face datasets are public — no key needed
```

### Collect Data

```bash
python src/scrapers/github_scraper.py       # → data/raw/github_prompts.json
python src/scrapers/huggingface_scraper.py  # → data/raw/huggingface_prompts.json
```

### Generate Synthetic Data

```bash
python data_generation/generate_synthetic.py   # → data/raw/synthetic_prompts.json
```

### Run Classification Pipeline

```bash
python src/pipeline.py    # → data/processed/classified_prompts.{json,csv}
```

The pipeline is **resumable** — if interrupted, re-running picks up from the last checkpoint automatically.

### Validate Data

```bash
python src/validate_data.py    # Prints a validation report; exits 1 if issues found
```

*Example Output (showing true positives on injected messy data):*
```text
==================================================
=== DATA VALIDATION REPORT ===
==================================================
Total prompts              : 38
Prompts with null fields   : 0
Invalid category values    : 8
Word count outliers        : 8
Flesch score outliers      : 0

Overall: 30/38 prompts passed all checks

--- Invalid Value Details ---
  [synthetic_messy_100] 'subject_domain'='desconocido' not in ['ciencias', 'historia', 'lectura', 'lengua', 'matemáticas', 'otro']
...
--- Word Count Outliers (outside 10–800) ---
  [synthetic_messy_102] word_count=6 (expected 10–800)
==================================================
```

### Launch Notebooks

```bash
jupyter notebook notebooks/prompt_analysis.ipynb
jupyter notebook notebooks/interaction_log_analysis.ipynb
```

## Classification Dimensions

| Dimension | What It Measures | Method | Possible Values |
|---|---|---|---|
| Subject Domain | Academic subject the chatbot tutors | Keyword matching → Gemini fallback | `matemáticas`, `lectura`, `historia`, `ciencias`, `lengua`, `otro` |
| Tone | Register and emotional warmth of the prompt | Gemini 1.5 Flash | `formal`, `cálido`, `motivador`, `neutral` |
| Scaffolding Depth | How explicitly the prompt instructs step-by-step guidance | Gemini 1.5 Flash | `bajo`, `medio`, `alto` |
| Motivational Strategies | Encouragement techniques built into the prompt | Gemini 1.5 Flash | `ninguna`, `elogio`, `orientación_a_metas`, `mentalidad_de_crecimiento`, `múltiple` |
| Cultural References | Presence of culturally inclusive or community-specific language | Keyword matching + Gemini | `presente`, `ausente` |
| Complexity | Linguistic reading difficulty of the prompt itself | `textstat` (Flesch Reading Ease) | Raw score 0–100 + word count, avg sentence length |

## Sample Output

```json
{
  "id": "synthetic_003",
  "source": "synthetic",
  "raw_text": "Eres un tutor de matemáticas paciente y motivador...",
  "language": "es",
  "collected_at": "2026-02-20T17:00:00+00:00",
  "metadata": {
    "subject": "matemáticas",
    "pedagogical_approach": "cálido y motivador",
    "model": "gemini-1.5-flash"
  },
  "subject_domain": "matemáticas",
  "tone": "motivador",
  "tone_justification": "El prompt usa lenguaje de aliento y celebración del esfuerzo.",
  "scaffolding_depth": "alto",
  "scaffolding_justification": "Instruye explícitamente dividir tareas en pasos pequeños.",
  "motivational_strategies": "mentalidad_de_crecimiento",
  "motivational_justification": "Enfatiza el esfuerzo sobre el resultado correcto.",
  "cultural_references": "ausente",
  "cultural_justification": "No contiene referencias culturales específicas.",
  "flesch_reading_ease": 58.3,
  "syllable_count": 312,
  "word_count": 198,
  "avg_sentence_length": 19.8
}
```

## Data Sources

| Source | File | Target Count | Auth Required | Notes |
|---|---|---|---|---|
| GitHub | `github_prompts.json` | 20–30 | None (optional token for rate limit) | Markdown files from public repos matching Spanish prompt keywords |
| Hugging Face | `huggingface_prompts.json` | 20–30 | **None** | `fka/awesome-chatgpt-prompts` + secondary dataset; education entries adapted to Spanish framing |
| Synthetic | `synthetic_prompts.json` | 30 | Free Gemini key | Gemini 1.5 Flash; 5 subjects × 6 pedagogical approaches |

Synthetic and Hugging Face records are clearly labelled in every record (`"source": "synthetic"` / `"huggingface"`) and in the metadata. Adapted English prompts carry an `adapted_from_english: true` flag for full transparency.

## Limitations & Future Work

- **LLM classification reliability**: Gemini outputs are consistent but not ground-truth labels. A real research context would require inter-rater reliability checks (e.g., Cohen's κ between two human coders and the LLM).
- **Synthetic data bias**: Gemini-generated prompts may reflect the model's own pedagogical assumptions rather than real teacher behavior.
- **Scraping coverage**: GitHub and Reddit are proxies for the real platform; actual RCT data would come directly from the chatbot configuration API on a weekly schedule.
- **Spanish variant sensitivity**: The keyword lists and Flesch scoring are calibrated for Castilian Spanish; prompts from Latin American teachers may use different vocabulary.

## Inter-Rater Reliability (IRR) Demonstration

We include an IRR validation notebook (`notebooks/irr_validation.ipynb`) simulating two independent human "coders" via two varied Gemini system prompts. Cohen's κ is computed across classifying 20 prompts for each dimension. This provides concrete evidence of reliability mapping directly to the terms required for acceptable automated social science coding (κ > 0.70).

## Longitudinal Tracking Example

In a real RCT, prompt evolution is tracked dynamically across weeks. `notebooks/longitudinal_tracking.ipynb` models how a theoretical teacher gradually modifies their prompt across 4 weeks (introducing praise, shifting toward direct scaffolding, adopting a growth mindset tone). By turning classified labels into ordinal measures, we demonstrate how temporal changes can be visualized and statistically proven.

## PlayLab Platform Integration

In a production scenario corresponding to the *When Teachers Program the AI* RCT, data is fundamentally gathered from the PlayLab platform where educators configure their tailored AI tutors.

Teachers interact with PlayLab through configuration fields defining the **system prompt** (the core instruction manual), **guardrails** (boundaries of what the bot can/cannot do), and **model parameters** (temperature/creativity settings). In our pipeline, the logic currently directed at raw strings ingested from GitHub/Synthetic arrays would be mapped directly onto PlayLab’s JSON schema exports — pulling specifically the user-configured system prompt strings and correlating them with the teacher IDs and timestamp markers. By doing this, the extraction runs unmodified via our `classify_prompt.py` module bridging raw EdTech configurations to quantifiable NLP outcomes.
