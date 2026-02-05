# ==============================================================================
# BASELINE 5: SEMÂNTICO VETORIAL
# Referência: Le and Pinkwart (2015) e Shwe et al. (2024).
# ==============================================================================
!pip install spacy -q
!pip install deep-translator -q
!python -m spacy download en_core_web_lg -q # Modelo Large Inglês

import pandas as pd
import spacy
import math
import random
from deep_translator import GoogleTranslator

# Carregar modelo large
nlp = spacy.load("en_core_web_lg")

# 1. Carregamento e Tradução
nome_arquivo = 'conteudos_com_bloom.xlsx'
try: df = pd.read_excel(nome_arquivo)
except: df = pd.DataFrame()

if not df.empty:
    df['final_title'] = df['final_title'].astype(str)
    if 'Tipo_Conhecimento_Inferido' not in df.columns: df['Tipo_Conhecimento_Inferido'] = 'Conceitual'
    if 'group_name' not in df.columns: df['group_name'] = 'Geral'

    print("Traduzindo para Inglês (Semântico)...")
    translator = GoogleTranslator(source='auto', target='en')
    df['title_en'] = df['final_title'].apply(lambda x: translator.translate(x) if isinstance(x, str) else x)

    # 2. Configuração Semântica (Inglês)
    bloom_candidates_en = {
        'Factual': ['Define', 'List', 'Identify', 'Recognize', 'Describe', 'Name'],
        'Conceitual': ['Explain', 'Classify', 'Summarize', 'Interpret', 'Distinguish', 'Compare', 'Understand'],
        'Procedural': ['Apply', 'Execute', 'Use', 'Implement', 'Perform', 'Calculate', 'Design', 'Model'],
        'Metacognitivo': ['Evaluate', 'Plan', 'Structure', 'Monitor', 'Validate', 'Critique']
    }

    def get_best_verb_vector_en(title_text, candidate_verbs):
        doc_title = nlp(title_text)
        best_verb = random.choice(candidate_verbs)
        best_score = -1.0
        for verb in candidate_verbs:
            doc_verb = nlp(verb)
            score = doc_verb.similarity(doc_title)
            if score > best_score:
                best_score = score
                best_verb = verb
        return best_verb, best_score

    # 3. Heurística
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

    # 4. Execução
    results = []
    for competencia, grupo in df.groupby('competencia_1'):
        target_skills = calcular_quantidade_skills(grupo)
        if 'score_relevancia' in grupo.columns:
            top_contents = grupo.sort_values(by='score_relevancia', ascending=False).head(target_skills)
        else:
            top_contents = grupo.head(target_skills)

        for idx, row in top_contents.iterrows():
            text_en = row['title_en']
            know_type = row.get('Tipo_Conhecimento_Inferido', 'Conceitual')
            k_key = 'Conceitual'
            if 'factual' in str(know_type).lower(): k_key = 'Factual'
            elif 'procedural' in str(know_type).lower(): k_key = 'Procedural'
            elif 'meta' in str(know_type).lower(): k_key = 'Metacognitivo'

            candidates = bloom_candidates_en[k_key]
            chosen_verb, score = get_best_verb_vector_en(text_en, candidates)
            clean_obj = text_en.replace('–', '-').strip()

            if k_key == 'Procedural': skill_text = f"{chosen_verb} the technique of '{clean_obj}'"
            elif k_key == 'Factual': skill_text = f"{chosen_verb} the elements of '{clean_obj}'"
            else: skill_text = f"{chosen_verb} concepts regarding '{clean_obj}'"

            results.append({
                'Competency': competencia,
                'Content': clean_obj,
                'Generated Skill': skill_text,
                'Semantic Score': round(score, 3)
            })

    df_result = pd.DataFrame(results)
    df_result.to_excel('baseline_5_semantic.xlsx', index=False)
    print("Baseline 5 Concluído. Arquivo salvo: baseline_5_semantic_english.xlsx")
    display(df_result.head())

