# ==============================================================================
# BASELINE 4: HÍBRIDO
# Referência: Combinação para mitigar as falhas de Gašpar usando Keklik.
# O que faz: Traduz. Tenta extrair verbo gramaticalmente.
# Se falhar (título nominal), aplica o template de regras. Garante 100% de geração em inglês.
# ==============================================================================
!pip install spacy -q
!pip install deep-translator -q
!python -m spacy download en_core_web_sm -q

import pandas as pd
import spacy
import math
import random
from deep_translator import GoogleTranslator

nlp = spacy.load("en_core_web_sm")

# 1. Carregamento e Tradução
nome_arquivo = 'conteudos_com_bloom.xlsx'
try: df = pd.read_excel(nome_arquivo)
except: df = pd.DataFrame()

if not df.empty:
    df['final_title'] = df['final_title'].astype(str)
    if 'Tipo_Conhecimento_Inferido' not in df.columns: df['Tipo_Conhecimento_Inferido'] = 'Conceitual'
    if 'group_name' not in df.columns: df['group_name'] = 'Geral'

    print("Traduzindo para Inglês (Híbrido)...")
    translator = GoogleTranslator(source='auto', target='en')
    df['title_en'] = df['final_title'].apply(lambda x: translator.translate(x) if isinstance(x, str) else x)

    # 2. Heurística (Quantidade)
    bloom_map = {'Lembrar': 1, 'Entender': 2, 'Aplicar': 3, 'Analisar': 4, 'Avaliar': 5, 'Criar': 6}
    def get_bloom_number(text):
        for k, v in bloom_map.items():
            if k.lower() in str(text).lower(): return v
        return 2
    df['b_level'] = df['Nível_Bloom_Competência'].apply(get_bloom_number)
    def calcular_quantidade_skills(grupo):
        c = len(grupo); d = grupo['Tipo_Conhecimento_Inferido'].nunique(); g = grupo['group_name'].nunique()
        b = grupo['b_level'].iloc[0]
        delta_d = 1 if d > 1 else 0; delta_g = 1 if g > 1 else 0
        if b <= 2: delta_b = -1
        elif b == 3: delta_b = 0
        elif b == 4: delta_b = 1
        else: delta_b = 2
        return int(min(8, max(3, math.floor(c / 10) + delta_d + delta_g + delta_b)))

    # 3. Dicionários e Lógica
    bloom_matrix_fallback = {
        'Factual': {'verbs': ['Define', 'List', 'Identify', 'Recognize']},
        'Conceitual': {'verbs': ['Explain', 'Classify', 'Summarize', 'Interpret']},
        'Procedural': {'verbs': ['Apply', 'Execute', 'Use', 'Implement']},
        'Metacognitivo': {'verbs': ['Evaluate', 'Plan', 'Structure', 'Validate']}
    }

    def generate_hybrid_skill_en(row):
        text = row['title_en']
        know_type = row.get('Tipo_Conhecimento_Inferido', 'Conceitual')

        # TENTATIVA 1: SINTÁTICA
        doc = nlp(text)
        extracted = None
        for token in doc:
            if token.pos_ == "VERB" or (token.dep_ == "ROOT" and token.pos_ in ["VERB"]):
                children = list(token.children)
                objects = [child for child in children if child.dep_ in ["dobj", "pobj", "attr"]]
                if objects:
                    root = token.lemma_
                    span = doc[objects[0].left_edge.i : objects[0].right_edge.i + 1]
                    extracted = f"{root.capitalize()} {span.text}"
                    break
        if extracted: return extracted, "Syntactic"

        # TENTATIVA 2: REGRAS
        k_key = 'Conceitual'
        if 'factual' in str(know_type).lower(): k_key = 'Factual'
        elif 'procedural' in str(know_type).lower(): k_key = 'Procedural'
        elif 'meta' in str(know_type).lower(): k_key = 'Metacognitivo'

        verb = random.choice(bloom_matrix_fallback[k_key]['verbs'])
        clean_obj = text.replace('–', '-').strip()

        if k_key == 'Procedural': fallback = f"{verb} the technique of '{clean_obj}'"
        elif k_key == 'Factual': fallback = f"{verb} the elements of '{clean_obj}'"
        else: fallback = f"{verb} concepts regarding '{clean_obj}'"

        return fallback, "Rules Fallback"

    # 4. Execução
    results = []
    for competencia, grupo in df.groupby('competencia_1'):
        target_skills = calcular_quantidade_skills(grupo)
        if 'score_relevancia' in grupo.columns: # Prioriza se tiver score IA
            top_contents = grupo.sort_values(by='score_relevancia', ascending=False).head(target_skills)
        else:
            top_contents = grupo.head(target_skills)

        for idx, row in top_contents.iterrows():
            skill, method = generate_hybrid_skill_en(row)
            results.append({
                'Competency': competencia,
                'Content (Translated)': row['title_en'],
                'Generated Skill': skill,
                'Method': method
            })

    df_result = pd.DataFrame(results)
    df_result.to_excel('baseline_4_hybrid.xlsx', index=False)
    print("Baseline 4 Concluído. Arquivo salvo: baseline_4_hybrid_english.xlsx")
    print(df_result['Method'].value_counts(normalize=True))
