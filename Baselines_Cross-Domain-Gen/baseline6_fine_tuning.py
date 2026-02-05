
# ==============================================================================
# 1. Fine_Tuning
# ==============================================================================
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes

# ==============================================================================
# 2. TREINAMENTO LLAMA 3 GERAR HABILIDADES
# ==============================================================================
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# --- CONFIGURAÇÃO ---
max_seq_length = 2048
dtype = None
load_in_4bit = True

print("🔄 Carregando Llama-3-8b...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Adiciona Adaptadores LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# --- PREPARAÇÃO DO DATASET ---
alpaca_prompt = """Abaixo está uma instrução que descreve uma tarefa, emparelhada com uma entrada que fornece mais contexto. Escreva uma resposta que complete adequadamente o pedido.

### Instrução:
{}

### Entrada:
{}

### Resposta:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

print("📂 Carregando dataset BNCC...")
try:
    dataset = load_dataset("json", data_files="train_bncc_api.jsonl", split="train")
    dataset = dataset.map(formatting_prompts_func, batched = True)
except Exception as e:
    print(f"ERRO: Não encontrei o arquivo 'train_bncc_api.jsonl'. Verifique o upload. Detalhe: {e}")

# --- O TREINAMENTO ---
print("🚀 Iniciando treinamento... (Aguarde a barra de progresso abaixo)")

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # <--- ISSO É CRUCIAL: Impede que ele trave pedindo login do WandB
    ),
)

trainer_stats = trainer.train()

print("✅ Treinamento concluído!")

from google.colab import userdata
from huggingface_hub import HfApi, login

# 1. Login e Configuração
hf_token = userdata.get('HF_TOKEN')
login(token=hf_token)
repo_name = "guirrock/llama3-8b-bncc-pedagogy"

# 2. Salvar LOCALMENTE primeiro (Garante que não perdemos o treino)
print("💾 Salvando arquivos localmente na pasta 'meu_modelo_lora'...")
try:
    # Tenta salvar pelo model (o método save_pretrained costuma ser mais robusto que push)
    model.save_pretrained("meu_modelo_lora")
    tokenizer.save_pretrained("meu_modelo_lora")
    print("✅ Arquivos salvos no disco do Colab!")
except AttributeError:
    # Se falhar, tenta salvar pelo trainer (plano B)
    print("⚠️ Tentando salvar via Trainer...")
    trainer.save_model("meu_modelo_lora")
    print("✅ Arquivos salvos via Trainer!")

# 3. Enviar a PASTA para o Hugging Face (Upload manual)
print(f"☁️ Enviando pasta para o repositório {repo_name}...")
api = HfApi()

try:
    api.create_repo(repo_id=repo_name, exist_ok=True, token=hf_token)

    api.upload_folder(
        folder_path="meu_modelo_lora",
        repo_id=repo_name,
        repo_type="model",
        token=hf_token
    )
    print(f"🚀 SUCESSO TOTAL! Seu modelo está salvo em: https://huggingface.co/{repo_name}")

except Exception as e:
    print(f"❌ Erro no upload: {e}")

# ==============================================================================
# SCRIPT GERAÇÃO DE HABILIDADES (USANDO MODELO FINE-TUNED)
# ==============================================================================

# --- 1. Instalar Bibliotecas (Atualizado para Unsloth) ---
print("1. Instalando bibliotecas...")
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes
!pip install pandas openpyxl sentence-transformers -qqq

import pandas as pd
import torch
from unsloth import FastLanguageModel # <--- MUDANÇA PRINCIPAL
from google.colab import userdata
from sentence_transformers import SentenceTransformer, util
import os
import re
from tqdm.notebook import tqdm

print("Bibliotecas prontas.")

# --- 2. Configuração e Autenticação ---
# (Seu modelo é público ou privado? Se for privado, precisa do token)
try:
    hf_token = userdata.get('HF_TOKEN')
except:
    hf_token = "SEU_TOKEN_AQUI_SE_PRECISAR"

# --- 3. Carregar Dados do Excel ---
print("\n3. Carregando dados do Excel...")
file_path_input = 'conteudos_com_bloom.xlsx'

if not os.path.exists(file_path_input):
    # Cria um dummy para teste se não existir arquivo
    print(f"⚠️ Aviso: '{file_path_input}' não encontrado. Certifique-se de fazer o upload.")
    df_conteudos = pd.DataFrame()
else:
    df_conteudos = pd.read_excel(file_path_input)
    df_conteudos.rename(columns={
        "competencia_1": "Competência",
        "final_title": "conteudo_curso",
        "Tipo_Conhecimento_Inferido": "Tipo_Mais_Frequente",
    }, inplace=True, errors='ignore')
    print(f"Dados carregados: {len(df_conteudos)} linhas.")

# --- 4. Carregar SEU Modelo Fine-Tuned (Unsloth) ---
print("\n4. Carregando seu modelo Fine-Tuned...")

# >>> AQUI ESTÁ A MUDANÇA DO CAMINHO <<<
model_name = "guirrock/llama3-8b-bncc-pedagogy"

max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = hf_token # Descomente se seu repo for privado
)

# Habilita modo de inferência (super rápido)
FastLanguageModel.for_inference(model)
print("✅ Seu modelo pedagogo foi carregado com sucesso!")

# Carrega o Sentence Transformer (Mantido igual)
embedding_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
print("Modelo SentenceTransformer carregado.")

# --- 5. Funções de Apoio e Prompt (ADAPTADO PARA ALPACA) ---

BLOOM_HIERARCHY = {
    "Lembrar": 1, "Compreender": 2, "Aplicar": 3,
    "Analisar": 4, "Avaliar": 5, "Criar": 6
}

def selecionar_conteudos_diversos(conteudos_list, embedding_model, similarity_threshold=0.7):
    # (Mantive sua função original intacta)
    if not conteudos_list: return []
    conteudos_list = [str(c) for c in conteudos_list if pd.notna(c)]
    if not conteudos_list: return []
    embeddings = embedding_model.encode(conteudos_list, convert_to_tensor=True)
    selected_contents = []
    processed_indices = set()
    for i in range(len(conteudos_list)):
        if i in processed_indices: continue
        similar_indices = {i}
        for j in range(i + 1, len(conteudos_list)):
            if j in processed_indices: continue
            if util.cos_sim(embeddings[i], embeddings[j]).item() >= similarity_threshold:
                similar_indices.add(j)
        best_content_in_group = max((conteudos_list[idx] for idx in similar_indices), key=len)
        selected_contents.append(best_content_in_group)
        processed_indices.update(similar_indices)
    return selected_contents

def calcular_numero_ideal_objetivos(n_conteudos, n_dimensoes, n_grupos, nivel_bloom):
    # (Mantive sua heurística intacta)
    base_obj = min(8, max(3, n_conteudos // 10))
    if n_dimensoes > 1: base_obj += 1
    if n_grupos > 1: base_obj += 1
    nivel_bloom_valor = BLOOM_HIERARCHY.get(nivel_bloom, 1)
    if nivel_bloom_valor <= 2: base_obj = max(3, base_obj - 1)
    if nivel_bloom_valor == 3: base_obj = min(8, base_obj)
    if nivel_bloom_valor == 4: base_obj = min(8, base_obj + 1)
    if nivel_bloom_valor > 4: base_obj = min(8, base_obj + 2)
    return max(3, min(8, base_obj))


def gerar_objetivos_com_unsloth(competencia, nivel_bloom, conteudos, max_objetivos):

    conteudos_str = ", ".join(conteudos)

    # Template Alpaca (O mesmo usado no treino da BNCC)
    # Adaptamos a instrução para pedir uma LISTA, mas mantendo o estilo pedagógico
    alpaca_prompt = """Abaixo está uma instrução que descreve uma tarefa, emparelhada com uma entrada que fornece mais contexto. Escreva uma resposta que complete adequadamente o pedido.

### Instrução:
Você é um especialista em currículo. Com base na Competência e nos Conteúdos fornecidos, elabore uma lista com exatamente {} habilidades educacionais técnicas e observáveis. As habilidades devem respeitar o nível cognitivo "{}".

### Entrada:
Competência: {}.
Conteúdos de Referência: {}.

### Resposta:
"""

    # Formata o prompt
    prompt_text = alpaca_prompt.format(max_objetivos, nivel_bloom, competencia, conteudos_str)

    # Tokeniza e envia para GPU
    inputs = tokenizer([prompt_text], return_tensors = "pt").to("cuda")

    # Gera
    outputs = model.generate(
        **inputs,
        max_new_tokens = 512, # Aumentei um pouco pois é uma lista
        use_cache = True,
        temperature = 0.3, # Baixo para manter a formalidade da BNCC
        top_p = 0.9
    )

    # Decodifica
    generated_text = tokenizer.batch_decode(outputs)[0]

    # Limpeza para pegar só a resposta (depois do ### Resposta:)
    resposta_limpa = generated_text.split("### Resposta:")[-1].replace(tokenizer.eos_token, "").strip()

    return resposta_limpa

# --- 6. Processamento Principal ---
print("\n6. Processando competências...")

if not df_conteudos.empty:
    resultados_objetivos = []
    grouped = df_conteudos.groupby(["Competência", "Nível_Bloom_Competência"])

    for (competencia, nivel_bloom), group in tqdm(grouped, desc="Gerando Habilidades"):

        # 1. Seleção de Conteúdos (Heurística)
        todos_conteudos = group['conteudo_curso'].dropna().tolist()
        conteudos_diversos = selecionar_conteudos_diversos(todos_conteudos, embedding_model)

        # 2. Cálculo da Quantidade (Heurística)
        n_dimensoes = group['Tipo_Mais_Frequente'].nunique()
        n_grupos = group['group_name'].nunique() if 'group_name' in group.columns else 1

        qtd_ideal = calcular_numero_ideal_objetivos(len(conteudos_diversos), n_dimensoes, n_grupos, nivel_bloom)

        # 3. Geração com IA (Seu Modelo)
        # print(f"Gerando para: {competencia[:30]}... ({qtd_ideal} habilidades)")

        texto_habilidades = gerar_objetivos_com_unsloth(
            competencia,
            nivel_bloom,
            conteudos_diversos,
            qtd_ideal
        )

        resultados_objetivos.append({
            'Competência': competencia,
            'Nível_Bloom': nivel_bloom,
            'Qtd_Calculada': qtd_ideal,
            'Conteúdos_Usados': ", ".join(conteudos_diversos),
            'Habilidades_Geradas_IA': texto_habilidades
        })

    # --- 7. Salvar ---
    df_final = pd.DataFrame(resultados_objetivos)
    df_final.to_excel("habilidades_geradas_fine_tuned.xlsx", index=False)
    print("\n✅ Concluído! Arquivo salvo: habilidades_geradas_fine_tuned.xlsx")

    # Mostra um exemplo
    print("\n--- Exemplo de Geração ---")
    print(df_final.iloc[0]['Habilidades_Geradas_IA'])

else:
    print("DataFrame vazio. Verifique o arquivo de entrada.")

