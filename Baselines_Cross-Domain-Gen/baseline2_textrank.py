# ==============================================================================
# BASELINE 2: TEXTRANK
# Referência: Abordagem Extrativa (Similar a Odilinye et al., 2015).
# O que faz: Traduz, monta um grafo de similaridade entre os títulos e escolhe os mais "centrais" (representativos).
# ==============================================================================
!pip install spacy -q
!pip install deep-translator -q

import pandas as pd
import numpy as np
import networkx as nx
import random
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator

# 1. Carregamento
nome_arquivo = 'conteudos_com_bloom.xlsx'
try: df = pd.read_excel(nome_arquivo)
except: df = pd.DataFrame()

if not df.empty:
    df['final_title'] = df['final_title'].astype(str)
    if 'Tipo_Conhecimento_Inferido' not in df.columns: df['Tipo_Conhecimento_Inferido'] = 'Conceitual'
    if 'group_name' not in df.columns: df['group_name'] = 'Geral'

    # 2. Tradução
    print("Traduzindo para Inglês (TextRank)...")
    translator = GoogleTranslator(source='auto', target='en')
    df['title_en'] = df['final_title'].apply(lambda x: translator.translate(x) if isinstance(x, str) else x)

    # 3. Função TextRank
    def calcular_textrank_scores(grupo_titulos):
        if len(grupo_titulos) < 2: return [1.0] * len(grupo_titulos)
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform(grupo_titulos)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores_dict = nx.pagerank(nx_graph, max_iter=100)
            return [scores_dict[i] for i in range(len(grupo_titulos))]
        except: return [0.0] * len(grupo_titulos)

    # 4. Heurística e Dicionários (English)
    # (Replicando lógica para script ser autônomo)
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
        'Factual': {'verbs': ['Define', 'List', 'Identify', 'Recognize']},
        'Conceitual': {'verbs': ['Explain', 'Classify', 'Summarize', 'Interpret']},
        'Procedural': {'verbs': ['Apply', 'Execute', 'Use', 'Implement']},
        'Metacognitivo': {'verbs': ['Evaluate', 'Plan', 'Structure', 'Validate']}
    }

    # 5. Execução
    results = []
    for competencia, grupo in df.groupby('competencia_1'):
        # Score TextRank
        titulos = grupo['title_en'].tolist()
        scores = calcular_textrank_scores(titulos)
        grupo = grupo.copy()
        grupo['score_textrank'] = scores

        target_skills = calcular_quantidade_skills(grupo)
        top_contents = grupo.sort_values(by='score_textrank', ascending=False).head(target_skills)

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
                'Score TextRank': row['score_textrank'],
                'Method': 'TextRank'
            })

    df_result = pd.DataFrame(results)
    df_result.to_excel('baseline_2_textrank.xlsx', index=False)
    print("Baseline 2 Concluído. Arquivo salvo: baseline_2_textrank_english.xlsx")
    display(df_result.head())

