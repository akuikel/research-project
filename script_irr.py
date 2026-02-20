import os
import sys
import json
import re
import time
import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
from google import genai
from google.genai import types

from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
client = genai.Client(api_key=api_key)

DATA_PATH = Path('data/raw/synthetic_prompts.json')
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# Use all 30 prompts for better statistical significance
sample_prompts = raw_data[:30]

CODER_PROMPT = """\
Analyze this Spanish educational chatbot system prompt and respond with ONLY a raw JSON object. No explanation, no markdown, no code fences. Start your response with {{ and end with }}.

{{"tone": "<formal|calido|motivador|neutral>", "scaffolding_depth": "<bajo|medio|alto>", "motivational_strategies": "<ninguna|elogio|metas|crecimiento|multiple>", "cultural_references": "<presente|ausente>"}}

Replace each placeholder with the correct value for this prompt:
{prompt_text}"""

VALID_VALUES = {
    'tone': ['formal', 'calido', 'motivador', 'neutral', 'cálido'],
    'scaffolding_depth': ['bajo', 'medio', 'alto'],
    'motivational_strategies': ['ninguna', 'elogio', 'metas', 'crecimiento', 'multiple', 'mentalidad_de_crecimiento', 'múltiple'],
    'cultural_references': ['presente', 'ausente'],
}

def extract_json(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw.strip(), flags=re.MULTILINE)
    match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
    if match:
        try: return json.loads(match.group())
        except: pass
    return {}

def classify_prompt(text: str, temp: float) -> dict:
    prompt = CODER_PROMPT.format(prompt_text=text[:1200])
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=temp, max_output_tokens=2048)
        )
        parsed = extract_json(response.text)
        return {k: parsed.get(k, "unknown") for k in VALID_VALUES.keys()}
    except Exception as e:
        return {k: "unknown" for k in VALID_VALUES.keys()}

results = []
for i, p in enumerate(sample_prompts):
    text = p.get('raw_text', '')
    coder_a = classify_prompt(text, 0.0) # Coder A is deterministic
    time.sleep(1)
    coder_b = classify_prompt(text, 0.7) # Coder B has high variance
    time.sleep(1)
    row = {'id': p.get('id', f'prompt_{i}')}
    for dim in VALID_VALUES.keys():
        row[f'A_{dim}'] = coder_a.get(dim)
        row[f'B_{dim}'] = coder_b.get(dim)
    results.append(row)

df = pd.DataFrame(results)
kappa_scores = []
for dim in VALID_VALUES.keys():
    valid_mask = df[f'A_{dim}'].isin(VALID_VALUES[dim]) & df[f'B_{dim}'].isin(VALID_VALUES[dim])
    valid_df = df[valid_mask]
    if len(valid_df) > 0:
        kappa = cohen_kappa_score(valid_df[f'A_{dim}'], valid_df[f'B_{dim}'])
    else: kappa = None
    kappa_scores.append({
        'Dimension': dim,
        'Kappa (k)': round(kappa, 3) if kappa is not None else 'N/A',
        'Valid Samples': len(valid_df)
    })
print('\n--- IRR Cohen\'s Kappa Results ---')
print(pd.DataFrame(kappa_scores).to_string(index=False))
