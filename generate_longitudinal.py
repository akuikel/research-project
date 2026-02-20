import json
import os

cells = []

# Cell 1 (Markdown)
md1 = '''# Longitudinal Tracking Example

This notebook demonstrates how we track the evolution of a single teacher's chatbot system prompt over a 4-week period. 
In the RCT, teachers might refine their prompts in response to student interactions. We want to quantify these changes to test hypotheses about teacher learning and adaptation.

Here, we'll create a synthetic 4-week dataset for a single `chatbot_id` (a math tutor), classify the prompts, compute the deltas between weeks, and plot the changes.'''
cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': md1})

# Cell 2 (Code)
code1 = '''import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Make sure we can import from src
sys.path.append(os.path.abspath('..'))
from src.classify_prompt import classify_prompt

plt.rcParams.update({
    'figure.dpi': 120,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
})
'''
cells.append({'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source': code1})

# Cell 3 (Markdown)
md2 = '''## 1. Synthetic Longitudinal Dataset

We mock a single teacher adjusting their prompt over 4 weeks:
- **Week 1**: Basic, neutral tone, no specific scaffolding.
- **Week 2**: Adds praise and a warmer tone.
- **Week 3**: Adds explicit step-by-step scaffolding.
- **Week 4**: Adds growth mindset motivation and a cultural reference.
'''
cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': md2})

# Cell 4 (Code)
code2 = '''longitudinal_data = [
    {
        "chatbot_id": "bot_math_01",
        "week": 1,
        "prompt_text": "Eres un tutor de matemáticas. Ayuda a los estudiantes a resolver los problemas de fracciones. Dales la respuesta si no saben."
    },
    {
        "chatbot_id": "bot_math_01",
        "week": 2,
        "prompt_text": "Eres un tutor de matemáticas amigable. Ayuda a los estudiantes con fracciones. Usa un tono cálido y diles que hacen un buen trabajo cuando acierten."
    },
    {
        "chatbot_id": "bot_math_01",
        "week": 3,
        "prompt_text": "Eres un tutor de matemáticas amigable y paciente. Ayuda con fracciones. No les des la respuesta directa: divide el problema en pasos muy pequeños y hazles preguntas para guiarles."
    },
    {
        "chatbot_id": "bot_math_01",
        "week": 4,
        "prompt_text": "Eres un tutor de matemáticas cálido para estudiantes de la comunidad gitana. Ayuda con fracciones dividiendo el problema en pasos pequeños. Enfatiza que los errores son parte del aprendizaje y que el esfuerzo es lo más importante."
    }
]

print(f"Defined {len(longitudinal_data)} weeks of prompts.")
'''
cells.append({'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source': code2})

# Cell 5 (Markdown)
md3 = '''## 2. Classification Pipeline

We run the same Gemini 2.5 Flash Lite classification pipeline used in production to score these 4 prompts.
'''
cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': md3})

# Cell 6 (Code)
code3 = '''results = []
for entry in longitudinal_data:
    classification = classify_prompt(entry['prompt_text'], prompt_id=f"w{entry['week']}")
    
    # Merge input metadata with classification results
    row = {
        'week': entry['week'],
        'chatbot_id': entry['chatbot_id'],
        **classification
    }
    results.append(row)

df = pd.DataFrame(results)
display(df[['week', 'tone', 'scaffolding_depth', 'motivational_strategies', 'cultural_references', 'flesch_reading_ease']])
'''
cells.append({'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source': code3})

# Cell 7 (Markdown)
md4 = '''## 3. Difference (Delta) Tracking Across Weeks

We convert categorical dimensions to ordinal scales to visualize the trajectory.
'''
cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': md4})

# Cell 8 (Code)
code4 = '''# Mapping ordinal values for plotting
mapping_tone = {'neutral': 0, 'formal': 0, 'calido': 1, 'motivador': 2}
mapping_scaffolding = {'bajo': 0, 'medio': 1, 'alto': 2}
mapping_culture = {'ausente': 0, 'presente': 1}
mapping_motivation = {'ninguna': 0, 'elogio': 1, 'metas': 1, 'crecimiento': 2, 'multiple': 2}

df['tone_score'] = df['tone'].map(mapping_tone)
df['scaffolding_score'] = df['scaffolding_depth'].map(mapping_scaffolding)
df['culture_score'] = df['cultural_references'].map(mapping_culture)
df['motivation_score'] = df['motivational_strategies'].map(mapping_motivation)

# Calculate Deltas
df_numeric = df[['week', 'tone_score', 'scaffolding_score', 'motivation_score', 'culture_score']]
diffs = df_numeric.set_index('week').diff().fillna(0)

print("Week-over-Week Change (Delta):")
display(diffs)
'''
cells.append({'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source': code4})

# Cell 9 (Markdown)
md5 = '''## 4. Visualization of Prompt Evolution
'''
cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': md5})

# Cell 10 (Code)
code5 = '''fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(df['week'], df['tone_score'], marker='o', label='Tone (Warmth/Motivation)', linewidth=2)
ax.plot(df['week'], df['scaffolding_score'], marker='s', label='Scaffolding Depth', linewidth=2)
ax.plot(df['week'], df['motivation_score'], marker='^', label='Motivational Strategies', linewidth=2)
ax.plot(df['week'], df['culture_score'], marker='D', label='Cultural References', linewidth=2)

ax.set_xticks([1, 2, 3, 4])
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['Low / Neutral', 'Medium / Moderate', 'High / Explicit'])

ax.set_title("Evolution of Teacher Prompt Across 4 Weeks (bot_math_01)", fontweight='bold')
ax.set_xlabel("Week of RCT")
ax.set_ylabel("Dimension Intensity")
ax.legend(title="Prompt Dimensions")

plt.tight_layout()
plt.show()
'''
cells.append({'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source': code5})

# Cell 11 (Markdown)
md6 = '''**Interpretation:** This simple tracking logic forms the core of our temporal analysis. By mapping text modifications to scalable numerical changes over time, we can test whether specific teacher professional development interventions (e.g., a workshop on growth mindset in Week 3) cause observable shifts in their prompt designs in subsequent weeks.
'''
cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': md6})

notebook = {
    'metadata': {},
    'nbformat': 4,
    'nbformat_minor': 5,
    'cells': cells
}

with open('notebooks/longitudinal_tracking.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)
