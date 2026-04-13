"""
=============================================================================
PROJETO: Sistema de Predição de Risco Clínico
ARQUIVO: 01_gerador_dataset.py
DESCRIÇÃO: Gera um dataset sintético com 2000 pacientes e dados biomédicos
           com regras realistas para classificação de risco clínico.
AUTOR: Projeto Acadêmico - Machine Learning Aplicado à Saúde
=============================================================================
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO INICIAL
# ─────────────────────────────────────────────────────────────────────────────

# Semente aleatória para reprodutibilidade: garante que cada execução
# produza os mesmos dados, fundamental em ambiente acadêmico
np.random.seed(42)

# Número total de registros a serem gerados
N = 2000

# ─────────────────────────────────────────────────────────────────────────────
# BLOCO 1: LISTA DE NOMES FICTÍCIOS
# Nomes simples, sem sobrenomes, para simular diversidade de pacientes
# ─────────────────────────────────────────────────────────────────────────────

nomes_masculinos = [
    "Carlos", "Pedro", "João", "Lucas", "Mateus", "Rafael", "Bruno",
    "André", "Felipe", "Gustavo", "Thiago", "Diego", "Eduardo", "Rodrigo",
    "Marcelo", "Leandro", "Vinícius", "Igor", "Renato", "Samuel"
]

nomes_femininos = [
    "Ana", "Maria", "Fernanda", "Juliana", "Beatriz", "Camila", "Larissa",
    "Patrícia", "Vanessa", "Letícia", "Aline", "Priscila", "Mariana",
    "Gabriela", "Isabela", "Cristina", "Luciana", "Natália", "Tatiane", "Bruna"
]

# Combina os dois grupos em uma lista única de nomes
todos_nomes = nomes_masculinos + nomes_femininos

# Sorteia aleatoriamente 2000 nomes com reposição (replace=True)
nomes = np.random.choice(todos_nomes, size=N, replace=True)

# ─────────────────────────────────────────────────────────────────────────────
# BLOCO 2: GERAÇÃO DAS VARIÁVEIS BIOMÉDICAS
# Cada variável segue uma distribuição normal (gaussiana) com parâmetros
# baseados em referências clínicas reais.
# ─────────────────────────────────────────────────────────────────────────────

# IDADE: distribuição uniforme entre 18 e 99 anos (adultos)
idade = np.random.randint(18, 100, size=N)

# GLICOSE (mg/dL):
#   - Normal em jejum: 70–99 mg/dL
#   - Pré-diabetes: 100–125 mg/dL
#   - Diabetes: ≥ 126 mg/dL
# Média 110, desvio 30 → cobre faixa normal a diabética
glicose = np.random.normal(loc=110, scale=30, size=N)
glicose = np.clip(glicose, 60, 300)          # limita valores extremos irreais
glicose = np.round(glicose, 1)               # arredonda para 1 casa decimal

# PRESSÃO ARTERIAL SISTÓLICA (mmHg):
#   - Normal: < 120 mmHg
#   - Elevada: 120–129 mmHg
#   - Hipertensão estágio 1: 130–139 mmHg
#   - Hipertensão estágio 2: ≥ 140 mmHg
# Média 125, desvio 20
pressao_arterial = np.random.normal(loc=125, scale=20, size=N)
pressao_arterial = np.clip(pressao_arterial, 60, 220)
pressao_arterial = np.round(pressao_arterial, 1)

# IMC - Índice de Massa Corporal (kg/m²):
#   - Abaixo do peso: < 18.5
#   - Normal: 18.5–24.9
#   - Sobrepeso: 25–29.9
#   - Obesidade grau I: 30–34.9
#   - Obesidade grau II+: ≥ 35
# Média 27, desvio 5
imc = np.random.normal(loc=27, scale=5, size=N)
imc = np.clip(imc, 15, 55)
imc = np.round(imc, 1)

# COLESTEROL TOTAL (mg/dL):
#   - Desejável: < 200 mg/dL
#   - Limítrofe: 200–239 mg/dL
#   - Alto: ≥ 240 mg/dL
# Média 210, desvio 40
colesterol = np.random.normal(loc=210, scale=40, size=N)
colesterol = np.clip(colesterol, 100, 400)
colesterol = np.round(colesterol, 1)

# ─────────────────────────────────────────────────────────────────────────────
# BLOCO 3: DEFINIÇÃO DO RISCO CLÍNICO
# Regra baseada em critérios clínicos reais:
#   - Cada variável fora da faixa ideal pontua positivamente
#   - O total de pontos determina a classe de risco
#
# CLASSES:
#   0 = Baixo Risco    (0–1 fatores de risco presentes)
#   1 = Risco Médio    (2 fatores de risco presentes)
#   2 = Alto Risco     (3 ou mais fatores de risco)
# ─────────────────────────────────────────────────────────────────────────────

def calcular_risco(idade, glicose, pressao, imc, colesterol):
    """
    Calcula o nível de risco clínico com base em fatores biomédicos.
    Retorna: 0 (baixo), 1 (médio) ou 2 (alto)
    """
    pontos = 0  # acumulador de fatores de risco

    # Fator 1: Idade avançada aumenta risco cardiovascular
    if idade >= 60:
        pontos += 1

    # Fator 2: Glicose elevada indica risco de diabetes ou pré-diabetes
    if glicose >= 126:
        pontos += 1
    elif glicose >= 100:
        pontos += 0.5  # pré-diabetes conta como meio fator

    # Fator 3: Pressão arterial elevada (hipertensão)
    if pressao >= 140:
        pontos += 1
    elif pressao >= 130:
        pontos += 0.5  # hipertensão estágio 1 conta como meio fator

    # Fator 4: IMC indica obesidade
    if imc >= 30:
        pontos += 1
    elif imc >= 25:
        pontos += 0.5  # sobrepeso conta como meio fator

    # Fator 5: Colesterol elevado
    if colesterol >= 240:
        pontos += 1
    elif colesterol >= 200:
        pontos += 0.5  # limítrofe conta como meio fator

    # Classificação final baseada no total de pontos acumulados
    if pontos <= 1:
        return 0   # Baixo Risco
    elif pontos <= 2.5:
        return 1   # Risco Médio
    else:
        return 2   # Alto Risco

# Aplica a função vetorialmente a todos os 2000 pacientes
# np.vectorize transforma a função escalar em uma função que aceita arrays
calcular_risco_vec = np.vectorize(calcular_risco)

risco = calcular_risco_vec(idade, glicose, pressao_arterial, imc, colesterol)

# ─────────────────────────────────────────────────────────────────────────────
# BLOCO 4: MONTAGEM DO DATAFRAME
# Organiza todas as variáveis em um DataFrame Pandas
# ─────────────────────────────────────────────────────────────────────────────

df = pd.DataFrame({
    'nome':            nomes,
    'idade':           idade,
    'glicose':         glicose,
    'pressao_arterial': pressao_arterial,
    'imc':             imc,
    'colesterol':      colesterol,
    'risco':           risco
})

# ─────────────────────────────────────────────────────────────────────────────
# BLOCO 5: VALIDAÇÃO E SALVAMENTO
# ─────────────────────────────────────────────────────────────────────────────

# Exibe as primeiras linhas para conferência visual
print("=" * 60)
print("  GERADOR DE DATASET - PREDIÇÃO DE RISCO CLÍNICO")
print("=" * 60)
print(f"\nTotal de registros gerados: {len(df)}")
print("\nPrimeiros 5 registros:")
print(df.head())

# Estatísticas descritivas das variáveis numéricas
print("\nEstatísticas descritivas:")
print(df.describe().round(2))

# Distribuição das classes de risco — importante verificar o balanceamento
print("\nDistribuição das classes de risco:")
contagem = df['risco'].value_counts().sort_index()
labels = {0: 'Baixo Risco', 1: 'Risco Médio', 2: 'Alto Risco'}
for classe, qtd in contagem.items():
    pct = (qtd / N) * 100
    print(f"  Classe {classe} ({labels[classe]}): {qtd} pacientes ({pct:.1f}%)")

# Salva o dataset em formato CSV
caminho_saida = 'pacientes.csv'
df.to_csv(caminho_saida, index=False, encoding='utf-8')
print(f"\nDataset salvo com sucesso em: '{caminho_saida}'")
print("=" * 60)
