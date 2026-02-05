
# ==============================================================================
# BASELINE 3: SINTÁTICO PURO
# Referência: Gašpar et al. (2023) e Lui et al. (2015).
# O que faz: Traduz para inglês, faz o parsing gramatical.
# Se achar "Verbo + Objeto", extrai. Se for só substantivo (título nominal), falha (mostrando a limitação).
# ==============================================================================
!pip install spacy -q
!pip install deep-translator -q
!python -m spacy download en_core_web_sm -q

import pandas as pd
import spacy
import math
from deep_translator import GoogleTranslator

# Carregar NLP Inglês
nlp = spacy.load("en_core_web_sm")

# 1. Carregamento
nome_arquivo = 'conteudos_com_bloom.xlsx'
try: df = pd.read_excel(nome_arquivo)
except: df = pd.DataFrame()

if not df.empty:
    df['final_title'] = df['final_title'].astype(str)
    if 'Tipo_Conhecimento_Inferido' not in df.columns: df['Tipo_Conhecimento_Inferido'] = 'Conceitual'
    if 'group_name' not in df.columns: df['group_name'] = 'Geral'

    # 2. Tradução
    print("Traduzindo para Inglês (AQG)...")
    translator = GoogleTranslator(source='auto', target='en')
    df['title_en'] = df['final_title'].apply(lambda x: translator.translate(x) if isinstance(x, str) else x)

    # 3. Extrator Sintático (Dependency Parsing)
    def extract_svo_skill_en(text):
        doc = nlp(text)
        root_verb = None
        target_object = None

        for token in doc:
            # Procura verbo raiz ou verbo principal
            if token.pos_ == "VERB" or (token.dep_ == "ROOT" and token.pos_ in ["VERB"]):
                children = list(token.children)
                objects = [child for child in children if child.dep_ in ["dobj", "pobj", "attr"]]
                if objects:
                    root_verb = token.lemma_
                    subtree_span = doc[objects[0].left_edge.i : objects[0].right_edge.i + 1]
                    target_object = subtree_span.text
                    break

        if root_verb and target_object:
            return f"{root_verb.capitalize()} {target_object}"
        return f"[SYNTAX FAIL] {text}" # Mantém tag de falha para análise

    # 4. Heurística (Omitida para economizar espaço, assumindo que as colunas já existem ou usando lógica simples)
    # (Inserir bloco de heurística igual aos anteriores se precisar recalcular quantidade)
    # Aqui vamos simplificar e pegar top 5 para demonstração, ou usar a lógica completa:
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

    # 5. Execução
    results = []
    for competencia, grupo in df.groupby('competencia_1'):
        target_skills = calcular_quantidade_skills(grupo)
        top_contents = grupo.head(target_skills) # Seleção simples para focar no teste sintático

        for idx, row in top_contents.iterrows():
            skill_text = extract_svo_skill_en(row['title_en'])
            results.append({
                'Competency': competencia,
                'Content (Translated)': row['title_en'],
                'Generated Skill': skill_text,
                'Status': 'Extracted' if '[SYNTAX FAIL]' not in skill_text else 'Failed (Noun)'
            })

    df_result = pd.DataFrame(results)
    df_result.to_excel('baseline_3_syntax.xlsx', index=False)
    print("Baseline 3 Concluído. Arquivo salvo: baseline_3_syntax_english.xlsx")
    display(df_result.head())

