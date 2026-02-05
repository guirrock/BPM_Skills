
# ==============================================================================
# Validation_Environmental_Education
# ==============================================================================
!pip install -U transformers sentence-transformers pdfplumber

import pdfplumber
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ==============================================================================
# 1. PREPARAÇÃO (MODELO E COMPETÊNCIAS)
# ==============================================================================
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

competencias_bncc = {
    # Sua competência alvo (ex: Vida e Cosmos)
    "COMP_02_VIDA_COSMOS": """
    Analisar e utilizar interpretações sobre a dinâmica da Vida, da Terra e do Cosmos
    para elaborar argumentos, realizar previsões sobre o funcionamento e a evolução
    dos seres vivos e do Universo.
    """,
     "COMP_03_INVESTIGACAO": """
    Investigar situações-problema e avaliar aplicações do conhecimento científico e
    tecnológico e suas implicações no mundo, utilizando procedimentos e linguagens
    próprios das Ciências da Natureza.
    """
}

# ==============================================================================
# 2. FUNÇÃO DE EXTRAÇÃO INTELIGENTE (CHUNK POR PARÁGRAFO)
# ==============================================================================
def extract_paragraphs_from_pdf(pdf_path):
    chunks = []
    print(f"Lendo PDF: {pdf_path}...")

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text: continue

            # Divide por quebras de parágrafo (padrão visual de PDF)
            # O PDFPlumber geralmente separa blocos visuais por \n
            raw_paragraphs = text.split('\n')

            buffer = ""
            for line in raw_paragraphs:
                # Se a linha for muito curta, pode ser título ou lixo, ou continuação
                if len(line) < 50:
                    buffer += " " + line
                else:
                    # Se o buffer já tem conteúdo, salva como um chunk
                    if len(buffer) > 100: # Só aceita parágrafos com substância
                        chunks.append({"pagina": i + 1, "texto": buffer.strip()})
                    buffer = line # Começa novo buffer

            # Adiciona o último buffer se houver
            if len(buffer) > 100:
                chunks.append({"pagina": i + 1, "texto": buffer.strip()})

    return pd.DataFrame(chunks)


df_chunks = extract_paragraphs_from_pdf("Guia_PNLD_2026.pdf")
print(f"Total de parágrafos extraídos: {len(df_chunks)}")

# ==============================================================================
# 3. BUSCA SEMÂNTICA FILTRADA
# ==============================================================================
print("Gerando vetores dos parágrafos...")
chunk_vectors = model.encode(df_chunks['texto'].tolist(), show_progress_bar=True)

results = []

# Palavras proibidas para limpar metadados (Heurística simples)
black_list = ["ISBN", "Código da Coleção", "Manual do Professor", "Dimensões", "Editora"]

for comp_name, comp_text in competencias_bncc.items():
    comp_vector = model.encode([comp_text])
    scores = cosine_similarity(comp_vector, chunk_vectors)[0]

    # Pega os TOP 20 parágrafos mais relevantes
    top_indices = np.argsort(scores)[-20:][::-1]

    for idx in top_indices:
        score = scores[idx]
        texto = df_chunks.iloc[idx]['texto']
        pag = df_chunks.iloc[idx]['pagina']

        # Filtro extra: Se tiver palavra proibida, ignora (limpa metadados)
        if any(bad_word in texto for bad_word in black_list):
            continue

        results.append({
            "Competencia_Alvo": comp_name,
            "Pagina_Origem": pag,
            "Score": round(score, 4),
            "Conteudo_Para_SkillDerive": texto
        })

# ==============================================================================
# 4. EXPORTAR PARA SEU PIPELINE
# ==============================================================================
df_final = pd.DataFrame(results)
# Exporta um JSONL ou CSV limpo para você injetar no seu script Llama-3
df_final.to_json("dataset_validacao_pnld.jsonl", orient="records", lines=True, force_ascii=False)

print("Arquivo 'dataset_validacao_pnld.jsonl' gerado.")
print("Exemplo de input extraído:\n")
print(df_final.iloc[0]['Conteudo_Para_SkillDerive'])

# ==============================================================================
# 4. EXPORTAR PARA CSV E EXCEL
# ==============================================================================
df_final = pd.DataFrame(results)

# OPÇÃO 1: EXCEL (.xlsx)
# Melhor para leitura humana e formatação. Requer: pip install openpyxl
try:
    df_final.to_excel("dataset_validacao_pnld.xlsx", index=False)
    print("Arquivo 'dataset_validacao_pnld.xlsx' gerado com sucesso.")
except ImportError:
    print("Para gerar em Excel, instale a biblioteca: pip install openpyxl")

# OPÇÃO 2: CSV (.csv)
# Melhor para importar em outros scripts.
# O encoding='utf-8-sig' garante que acentos (ç, ã, é) abram corretamente no Excel/Windows.
# O sep=';' é o padrão para CSVs no Brasil (já que usamos vírgula para decimais).
df_final.to_csv(
    "dataset_validacao_pnld.csv",
    index=False,
    encoding='utf-8-sig',
    sep=';',
    quoting=1 # Garante que textos com aspas não quebrem o CSV
)

print("Arquivo 'dataset_validacao_pnld.csv' gerado com sucesso.")

# Visualização rápida das 5 primeiras linhas
print("\n--- AMOSTRA DOS DADOS ---")
print(df_final[['Competencia_Alvo', 'Score', 'Conteudo_Para_SkillDerive']].head())

!pip install sentence-transformers openpyxl

# ==============================================================================
# COMPARAÇÃO HABILIDADES GERADAS PELO MÉTODO E HABILIDADES BNCC EDUCAÇÃO AMBIENTAL
# ==============================================================================
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================
# Caminhos dos arquivos (ajuste se necessário)
file_bncc = "BNCC_Ambiente.xlsx"
file_user = "habilidades_unicas_por_competencia_ambiental.xlsx"

# Abas para processar
sheets_to_process = ["C2", "C3"]

# Modelo de Embeddings
# O 'paraphrase-multilingual-MiniLM-L12-v2' é excelente para similaridade em PT-BR
model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

print(f"Carregando modelo: {model_name}...")
model = SentenceTransformer(model_name)

# Lista para guardar os resultados finais de todas as abas
all_results = []

# ==============================================================================
# PROCESSAMENTO
# ==============================================================================
for sheet in sheets_to_process:
    print(f"\n--- Processando Aba: {sheet} ---")

    # 1. Carregar Dados
    try:
        df_bncc = pd.read_excel(file_bncc, sheet_name=sheet)
        df_user = pd.read_excel(file_user, sheet_name=sheet)
    except Exception as e:
        print(f"Erro ao ler a aba {sheet}: {e}")
        continue

    # Validação básica de colunas
    if 'Habilidade' not in df_bncc.columns:
        print(f"Aviso: Coluna 'Habilidade' não encontrada na BNCC aba {sheet}")
        continue
    if 'Habilidades_Agrupadas' not in df_user.columns:
        print(f"Aviso: Coluna 'Habilidades_Agrupadas' não encontrada no arquivo do usuário aba {sheet}")
        continue

    # Limpeza básica (remover vazios e converter para string)
    bncc_skills = df_bncc['Habilidade'].dropna().astype(str).tolist()
    user_skills = df_user['Habilidades_Agrupadas'].dropna().astype(str).tolist()

    print(f"Habilidades BNCC: {len(bncc_skills)} | Habilidades Geradas: {len(user_skills)}")

    # 2. Gerar Embeddings
    # Convertemos as frases para vetores
    embeddings_bncc = model.encode(bncc_skills, convert_to_tensor=True)
    embeddings_user = model.encode(user_skills, convert_to_tensor=True)

    # 3. Calcular Similaridade Semântica (Cosseno)
    # Resultado é uma matriz onde linhas = user_skills, colunas = bncc_skills
    cosine_scores = util.cos_sim(embeddings_user, embeddings_bncc)

    # 4. Encontrar a melhor correspondência para cada habilidade gerada
    for i, user_skill in enumerate(user_skills):
        # Encontra o índice da habilidade BNCC com maior score para esta habilidade do usuário
        best_match_idx = torch.argmax(cosine_scores[i]).item()
        best_score = cosine_scores[i][best_match_idx].item()

        best_bncc_skill = bncc_skills[best_match_idx]

        # Salva o resultado
        all_results.append({
            'Competencia': sheet,
            'Habilidade_Gerada_Modelo': user_skill,
            'Melhor_Match_BNCC': best_bncc_skill,
            'Score_Similaridade': round(best_score, 4)
        })

# ==============================================================================
# EXPORTAÇÃO
# ==============================================================================
if all_results:
    df_final = pd.DataFrame(all_results)

    # Ordenar por Score (opcional, para ver os piores/melhores matches primeiro)
    df_final = df_final.sort_values(by='Score_Similaridade', ascending=False)

    output_filename = "relatorio_comparacao_semantica.xlsx"
    df_final.to_excel(output_filename, index=False)

    print(f"\n✅ Análise concluída! Relatório salvo em: {output_filename}")

    # Exibir um resumo estatístico
    print("\nResumo das Similaridades:")
    print(df_final['Score_Similaridade'].describe())
else:
    print("\nNenhum resultado foi gerado. Verifique os nomes dos arquivos e colunas.")