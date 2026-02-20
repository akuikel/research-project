import os
import sys
import pandas as pd
from pathlib import Path

sys.path.append(os.path.abspath('src'))
from classify_prompt import classify_prompt

longitudinal_data = [
    {
        'chatbot_id': 'bot_math_01',
        'week': 1,
        'prompt_text': 'Eres un tutor de matemáticas. Ayuda a los estudiantes a resolver los problemas de fracciones. Dales la respuesta si no saben.'
    },
    {
        'chatbot_id': 'bot_math_01',
        'week': 2,
        'prompt_text': 'Eres un tutor de matemáticas amigable. Ayuda a los estudiantes con fracciones. Usa un tono cálido y diles que hacen un buen trabajo cuando acierten.'
    },
    {
        'chatbot_id': 'bot_math_01',
        'week': 3,
        'prompt_text': 'Eres un tutor de matemáticas amigable y paciente. Ayuda con fracciones. No les des la respuesta directa: divide el problema en pasos muy pequeños y hazles preguntas para guiarles.'
    },
    {
        'chatbot_id': 'bot_math_01',
        'week': 4,
        'prompt_text': 'Eres un tutor de matemáticas cálido para estudiantes de la comunidad gitana. Ayuda con fracciones dividiendo el problema en pasos pequeños. Enfatiza que los errores son parte del aprendizaje y que el esfuerzo es lo más importante.'
    }
]

results = []
for entry in longitudinal_data:
    classification = classify_prompt(entry['prompt_text'], prompt_id=f"w{entry['week']}")
    row = {'week': entry['week'], 'chatbot_id': entry['chatbot_id'], **classification}
    results.append(row)

df = pd.DataFrame(results)

mapping_tone = {'neutral': 0, 'formal': 0, 'calido': 1, 'cálido': 1, 'motivador': 2}
mapping_scaffolding = {'bajo': 0, 'medio': 1, 'alto': 2}
mapping_culture = {'ausente': 0, 'presente': 1}
mapping_motivation = {'ninguna': 0, 'elogio': 1, 'metas': 1, 'orientación_a_metas': 1, 'crecimiento': 2, 'mentalidad_de_crecimiento': 2, 'múltiple': 2, 'multiple': 2}

df['tone_score'] = df['tone'].map(mapping_tone)
df['scaffolding_score'] = df['scaffolding_depth'].map(mapping_scaffolding)
df['culture_score'] = df['cultural_references'].map(mapping_culture)
df['motivation_score'] = df['motivational_strategies'].map(mapping_motivation)

df_numeric = df[['week', 'tone_score', 'scaffolding_score', 'motivation_score', 'culture_score']]
diffs = df_numeric.set_index('week').diff().fillna(0)

print('--- Longitudinal Raw Classifications ---')
print(df[['week', 'tone', 'scaffolding_depth', 'motivational_strategies', 'cultural_references']].to_string(index=False))
print('\n--- Week-over-Week Change (Delta) ---')
print(diffs.to_string())
