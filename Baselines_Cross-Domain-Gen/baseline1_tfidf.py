# ==============================================================================
# BASELINE 1: TF-IDF + REGRAS
# Referência: Baseado em Wijanarko et al. (2021) e Das et al. (2021).
# O que faz: Traduz, calcula quais palavras são estatisticamente relevantes (TF-IDF) e aplica um template rígido em inglês.
# ==============================================================================
!pip install spacy -q
!pip install deep-translator -q
!python -m spacy download en_core_web_sm -q

import pandas as pd
import numpy as np
import random
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from deep_translator import GoogleTranslator

# 1. Carregamento
nome_arquivo = 'conteudos_com_bloom.xlsx'
try:
    df = pd.read_excel(nome_arquivo)
except:
    print("ERRO: Faça upload do arquivo xlsx.")
    df = pd.DataFrame()

if not df.empty:
    df['final_title'] = df['final_title'].astype(str)
    if 'Tipo_Conhecimento_Inferido' not in df.columns: df['Tipo_Conhecimento_Inferido'] = 'Conceitual'
    if 'group_name' not in df.columns: df['group_name'] = 'Geral'

    # 2. Tradução para Inglês
    print("Traduzindo conteúdos para Inglês...")
    translator = GoogleTranslator(source='auto', target='en')
    def translate_safe(text):
        try: return translator.translate(text)
        except: return text
    df['title_en'] = df['final_title'].apply(translate_safe)

    # 3. TF-IDF (Score de Relevância)
    print("Calculando TF-IDF...")
    custom_stop_words = list(ENGLISH_STOP_WORDS) + ['process', 'business', 'management', 'model']
    vectorizer = TfidfVectorizer(stop_words=custom_stop_words)
    tfidf_matrix = vectorizer.fit_transform(df['title_en'])
    dense = tfidf_matrix.todense()
    df['score_tfidf'] = np.array(dense.sum(axis=1)).flatten()

    # 4. Heurística e Dicionários (English)
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

    bloom_matrix_en = {
        'Factual': {'verbs': ['Define', 'List', 'Identify', 'Recognize', 'Name']},
        'Conceitual': {'verbs': ['Explain', 'Classify', 'Summarize', 'Interpret', 'Distinguish']},
        'Procedural': {'verbs': ['Apply', 'Execute', 'Use', 'Implement', 'Perform']},
        'Metacognitivo': {'verbs': ['Evaluate', 'Plan', 'Structure', 'Monitor', 'Validate']}
    }

    # 5. Geração
    results = []
    for competencia, grupo in df.groupby('competencia_1'):
        target_skills = calcular_quantidade_skills(grupo)
        top_contents = grupo.sort_values(by='score_tfidf', ascending=False).head(target_skills)

        for idx, row in top_contents.iterrows():
            know_type = row.get('Tipo_Conhecimento_Inferido', 'Conceitual')
            k_key = 'Conceitual'
            if 'factual' in str(know_type).lower(): k_key = 'Factual'
            elif 'procedural' in str(know_type).lower(): k_key = 'Procedural'
            elif 'meta' in str(know_type).lower(): k_key = 'Metacognitivo'

            verb = random.choice(bloom_matrix_en[k_key]['verbs'])
            obj = str(row['title_en']).replace('–', '-').strip()

            if k_key == 'Procedural': skill_text = f"{verb} the technique of '{obj}'"
            elif k_key == 'Factual': skill_text = f"{verb} the elements of '{obj}'"
            else: skill_text = f"{verb} concepts regarding '{obj}'"

            results.append({
                'Competency': competencia,
                'Content (Translated)': obj,
                'Generated Skill': skill_text,
                'Method': 'TF-IDF + Rules'
            })

    df_result = pd.DataFrame(results)
    df_result.to_excel('baseline_1_tfidf.xlsx', index=False)
    print("Baseline 1 Concluído. Arquivo salvo: baseline_1_tfidf_english.xlsx")
    display(df_result.head())

