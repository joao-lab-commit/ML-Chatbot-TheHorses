"""
=============================================================================
PROJETO: Sistema de Predição de Risco Clínico
ARQUIVO: 02_pipeline_ml.py
DESCRIÇÃO: Pipeline completo de Machine Learning — leitura, pré-processamento,
           treinamento de múltiplos modelos, avaliação, visualização e predição.
DEPENDÊNCIA: Execute primeiro o arquivo 01_gerador_dataset.py para gerar
             o arquivo pacientes.csv antes de rodar este script.
AUTOR: Projeto Acadêmico - Machine Learning Aplicado à Saúde
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTAÇÕES
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')   # Suprime avisos de convergência em logs

# Utilitários de pré-processamento e divisão de dados
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize

# Modelos de Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Métricas de avaliação
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 1: LEITURA E INSPEÇÃO DO DATASET
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  PIPELINE DE MACHINE LEARNING — PREDIÇÃO DE RISCO CLÍNICO")
print("=" * 65)

# Carrega o CSV gerado pelo script anterior
df = pd.read_csv('pacientes.csv')

print("\n[ ETAPA 1 ] Carregamento e inspeção do dataset")
print(f"  Dimensões: {df.shape[0]} linhas × {df.shape[1]} colunas")
print(f"  Colunas  : {list(df.columns)}")
print(f"\n  Primeiros registros:")
print(df.head(3).to_string(index=False))

# Verificação de valores nulos — essencial antes de qualquer processamento
nulos = df.isnull().sum()
print(f"\n  Valores nulos por coluna:\n{nulos}")
print(f"  → Nenhum valor nulo encontrado." if nulos.sum() == 0 else "  → ATENÇÃO: valores nulos detectados!")

# Distribuição das classes — verificar balanceamento do dataset
print("\n  Distribuição da variável alvo (risco):")
dist = df['risco'].value_counts().sort_index()
for cls, cnt in dist.items():
    label = ["Baixo Risco", "Risco Médio", "Alto Risco"][cls]
    print(f"    Classe {cls} — {label}: {cnt} pacientes ({cnt/len(df)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 2: SEPARAÇÃO DE FEATURES E TARGET
# ─────────────────────────────────────────────────────────────────────────────

print("\n[ ETAPA 2 ] Separação de features (X) e target (y)")

# Features: variáveis preditoras (excluímos 'nome' pois é identificador textual)
# e 'risco' pois é a variável alvo
X = df[['idade', 'glicose', 'pressao_arterial', 'imc', 'colesterol']]

# Target: variável que queremos prever
y = df['risco']

print(f"  Features selecionadas : {list(X.columns)}")
print(f"  Formato de X          : {X.shape}")
print(f"  Formato de y          : {y.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 3: DIVISÃO TREINO / TESTE
# ─────────────────────────────────────────────────────────────────────────────

print("\n[ ETAPA 3 ] Divisão treino/teste (80% / 20%)")

# stratify=y garante que a proporção das classes seja mantida
# tanto no conjunto de treino quanto no de teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% para teste → 400 amostras
    random_state=42,    # reprodutibilidade
    stratify=y          # divisão estratificada por classe
)

print(f"  Total de amostras  : {len(X)}")
print(f"  Treino             : {len(X_train)} amostras ({len(X_train)/len(X)*100:.0f}%)")
print(f"  Teste              : {len(X_test)} amostras ({len(X_test)/len(X)*100:.0f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 4: NORMALIZAÇÃO DOS DADOS (StandardScaler)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[ ETAPA 4 ] Normalização com StandardScaler")

# StandardScaler transforma os dados para média=0 e desvio padrão=1
# REGRA CRÍTICA: fit() somente nos dados de treino para evitar data leakage
# (vazamento de informação do conjunto de teste para o modelo)
scaler = StandardScaler()

# fit_transform: aprende a média/desvio do treino e já transforma
X_train_scaled = scaler.fit_transform(X_train)

# transform: usa os parâmetros aprendidos no treino para transformar o teste
# (o scaler NÃO vê os dados de teste durante o ajuste)
X_test_scaled = scaler.transform(X_test)

print("  Média das features antes da escala (treino):")
print("  ", dict(zip(X.columns, X_train.mean().round(2).values)))
print("  Média após escala (deve ser ≈ 0 para cada feature):")
print("  ", dict(zip(X.columns, X_train_scaled.mean(axis=0).round(4))))

# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 5: DEFINIÇÃO E TREINAMENTO DOS MODELOS
# ─────────────────────────────────────────────────────────────────────────────

print("\n[ ETAPA 5 ] Treinamento dos modelos")

# Dicionário com os três modelos que serão treinados e comparados
modelos = {
    # Regressão Logística: modelo linear, interpretável, bom baseline
    # max_iter=1000 garante convergência em datasets com mais features
    # solver='lbfgs' suporta classificação multiclasse nativamente
    'Regressão Logística': LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    ),

    # Random Forest: ensemble de árvores de decisão, robusto e preciso
    # n_estimators=100: usa 100 árvores na floresta
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1          # usa todos os núcleos disponíveis
    ),

    # KNN (K-Nearest Neighbors): classifica com base nos k vizinhos mais próximos
    # k=5 é um valor padrão equilibrado (reduz ruído sem perder fronteiras)
    'KNN (k=5)': KNeighborsClassifier(
        n_neighbors=5
    )
}

# Dicionários para armazenar resultados de cada modelo
resultados = {}          # métricas no conjunto de teste
scores_cv   = {}         # scores da validação cruzada

# Configuração da validação cruzada estratificada
# k=5 folds: garante que cada fold tenha representação proporcional de classes
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Loop de treinamento e avaliação
for nome, modelo in modelos.items():
    print(f"\n  Treinando: {nome}...")

    # ── Treinamento no conjunto de treino normalizado
    modelo.fit(X_train_scaled, y_train)

    # ── Predição no conjunto de teste
    y_pred = modelo.predict(X_test_scaled)

    # ── Cálculo das métricas de avaliação
    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1        = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # ── Validação cruzada: avalia o modelo em 5 partições diferentes
    # Importante: usa os dados de TREINO completos (sem o conjunto de teste)
    cv_scores = cross_val_score(
        modelo, X_train_scaled, y_train,
        cv=kfold,
        scoring='accuracy'
    )

    # Armazena todos os resultados
    resultados[nome] = {
        'modelo':     modelo,
        'y_pred':     y_pred,
        'acuracia':   acc,
        'precision':  precision,
        'recall':     recall,
        'f1':         f1
    }

    scores_cv[nome] = cv_scores

    # Exibe sumário do modelo atual
    print(f"    Acurácia (teste)       : {acc:.4f} ({acc*100:.2f}%)")
    print(f"    Precisão (weighted)    : {precision:.4f}")
    print(f"    Recall (weighted)      : {recall:.4f}")
    print(f"    F1-Score (weighted)    : {f1:.4f}")
    print(f"    CV Accuracy (5-fold)   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 6: COMPARAÇÃO E IDENTIFICAÇÃO DO MELHOR MODELO
# ─────────────────────────────────────────────────────────────────────────────

print("\n[ ETAPA 6 ] Comparação dos modelos")
print("-" * 65)
print(f"  {'Modelo':<25} {'Acurácia':>10} {'Precisão':>10} {'Recall':>10} {'F1':>10}")
print("-" * 65)

for nome, res in resultados.items():
    print(f"  {nome:<25} {res['acuracia']:>10.4f} {res['precision']:>10.4f} "
          f"{res['recall']:>10.4f} {res['f1']:>10.4f}")

print("-" * 65)

# Identifica o melhor modelo pelo F1-Score (mais robusto que acurácia isolada
# em datasets com possível desequilíbrio entre classes)
melhor_nome = max(resultados, key=lambda k: resultados[k]['f1'])
melhor_res  = resultados[melhor_nome]
melhor_modelo = melhor_res['modelo']

print(f"\n  Melhor modelo: {melhor_nome}")
print(f"  F1-Score     : {melhor_res['f1']:.4f}")

# Relatório detalhado do melhor modelo (precision/recall/f1 por classe)
print(f"\n  Relatório de classificação completo — {melhor_nome}:")
labels_nomes = ['Baixo Risco', 'Risco Médio', 'Alto Risco']
print(classification_report(y_test, melhor_res['y_pred'],
                             target_names=labels_nomes))

# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 7: VISUALIZAÇÕES
# ─────────────────────────────────────────────────────────────────────────────

print("\n[ ETAPA 7 ] Gerando visualizações...")

# Paleta de cores consistente para todos os gráficos
CORES = ['#2196F3', '#4CAF50', '#FF9800']   # azul, verde, laranja

# ── FIGURA 1: Comparação de Acurácia dos Modelos ─────────────────────────────
fig1, ax1 = plt.subplots(figsize=(9, 5))

nomes_modelos = list(resultados.keys())
acuracias     = [resultados[n]['acuracia'] for n in nomes_modelos]
cv_medias     = [scores_cv[n].mean() for n in nomes_modelos]
cv_stds       = [scores_cv[n].std() for n in nomes_modelos]

x = np.arange(len(nomes_modelos))
largura = 0.35

# Barras: acurácia no conjunto de teste
barras1 = ax1.bar(x - largura/2, acuracias, largura,
                  label='Acurácia (Teste)', color='#2196F3', alpha=0.85, edgecolor='white')

# Barras: acurácia média na validação cruzada com barra de erro (±1 desvio)
barras2 = ax1.bar(x + largura/2, cv_medias, largura,
                  label='Acurácia Média CV (k=5)', color='#FF9800',
                  alpha=0.85, edgecolor='white',
                  yerr=cv_stds, capsize=6, error_kw={'linewidth': 2})

# Anotação dos valores em cima de cada barra
for barra in barras1:
    altura = barra.get_height()
    ax1.annotate(f'{altura:.3f}',
                 xy=(barra.get_x() + barra.get_width() / 2, altura),
                 xytext=(0, 5), textcoords="offset points",
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

for barra in barras2:
    altura = barra.get_height()
    ax1.annotate(f'{altura:.3f}',
                 xy=(barra.get_x() + barra.get_width() / 2, altura),
                 xytext=(0, 5), textcoords="offset points",
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.set_xlabel('Modelo', fontsize=12)
ax1.set_ylabel('Acurácia', fontsize=12)
ax1.set_title('Comparação de Acurácia: Teste vs Validação Cruzada', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(nomes_modelos, fontsize=11)
ax1.set_ylim(0, 1.1)
ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4)
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)
fig1.tight_layout()
fig1.savefig('grafico_comparacao_modelos.png', dpi=150, bbox_inches='tight')
print("  Salvo: grafico_comparacao_modelos.png")

# ── FIGURA 2: Matriz de Confusão do Melhor Modelo ────────────────────────────
fig2, ax2 = plt.subplots(figsize=(7, 6))

cm = confusion_matrix(y_test, melhor_res['y_pred'])

# Exibe a matriz com gradiente de cor para identificar padrões visualmente
im = ax2.imshow(cm, interpolation='nearest', cmap='Blues')
fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

# Rótulos dos eixos
ax2.set_xticks(range(3))
ax2.set_yticks(range(3))
ax2.set_xticklabels(labels_nomes, rotation=20, ha='right', fontsize=10)
ax2.set_yticklabels(labels_nomes, fontsize=10)
ax2.set_xlabel('Classe Predita', fontsize=12)
ax2.set_ylabel('Classe Real', fontsize=12)
ax2.set_title(f'Matriz de Confusão — {melhor_nome}', fontsize=13, fontweight='bold')

# Anota cada célula com o número de predições
limiar = cm.max() / 2  # limiar para cor do texto (branco/preto)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        cor_texto = 'white' if cm[i, j] > limiar else 'black'
        ax2.text(j, i, str(cm[i, j]),
                 ha='center', va='center',
                 fontsize=14, fontweight='bold', color=cor_texto)

fig2.tight_layout()
fig2.savefig('matriz_confusao.png', dpi=150, bbox_inches='tight')
print("  Salvo: matriz_confusao.png")

# ── FIGURA 3: Curva ROC (One-vs-Rest para multiclasse) ───────────────────────
fig3, ax3 = plt.subplots(figsize=(8, 6))

# Binariza o target para o cálculo ROC no formato One-vs-Rest
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

# Obtém probabilidades de pertencer a cada classe
# (disponível em todos os três modelos)
y_prob = melhor_modelo.predict_proba(X_test_scaled)

cores_roc = ['#2196F3', '#4CAF50', '#F44336']
classes_labels = ['Baixo Risco (0)', 'Risco Médio (1)', 'Alto Risco (2)']

# Plota a curva ROC para cada classe separadamente
for i, (label, cor) in enumerate(zip(classes_labels, cores_roc)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    area = auc(fpr, tpr)
    ax3.plot(fpr, tpr, lw=2, color=cor,
             label=f'{label} (AUC = {area:.3f})')

# Linha diagonal de referência (classificador aleatório)
ax3.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6, label='Aleatório (AUC = 0.5)')

ax3.set_xlim([0.0, 1.0])
ax3.set_ylim([0.0, 1.05])
ax3.set_xlabel('Taxa de Falso Positivo (FPR)', fontsize=12)
ax3.set_ylabel('Taxa de Verdadeiro Positivo (TPR)', fontsize=12)
ax3.set_title(f'Curva ROC (One-vs-Rest) — {melhor_nome}', fontsize=13, fontweight='bold')
ax3.legend(loc='lower right', fontsize=10)
ax3.grid(alpha=0.3)
fig3.tight_layout()
fig3.savefig('curva_roc.png', dpi=150, bbox_inches='tight')
print("  Salvo: curva_roc.png")

# ── FIGURA 4: Importância das Features (Random Forest) ───────────────────────
# Apenas modelos baseados em árvore fornecem importância diretamente
if 'Random Forest' in resultados:
    fig4, ax4 = plt.subplots(figsize=(7, 4))

    rf_model = resultados['Random Forest']['modelo']
    importancias = rf_model.feature_importances_
    features = X.columns.tolist()

    # Ordena por importância decrescente para visualização clara
    indices = np.argsort(importancias)[::-1]
    features_ord = [features[i] for i in indices]
    imp_ord = importancias[indices]

    barras = ax4.bar(range(len(features_ord)), imp_ord,
                     color=['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336'],
                     alpha=0.85, edgecolor='white')
    ax4.set_xticks(range(len(features_ord)))
    ax4.set_xticklabels(features_ord, rotation=15, ha='right', fontsize=11)
    ax4.set_ylabel('Importância Relativa', fontsize=12)
    ax4.set_title('Importância das Features — Random Forest', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    # Anota o percentual em cada barra
    for b, imp in zip(barras, imp_ord):
        ax4.annotate(f'{imp*100:.1f}%',
                     xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    fig4.tight_layout()
    fig4.savefig('importancia_features.png', dpi=150, bbox_inches='tight')
    print("  Salvo: importancia_features.png")

plt.show()
print("  Todas as visualizações foram geradas com sucesso.")

# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 8: PREDIÇÃO PARA UM NOVO PACIENTE
# Simula um caso clínico real e utiliza o melhor modelo para classificar
# ─────────────────────────────────────────────────────────────────────────────

print("\n[ ETAPA 8 ] Predição para novos pacientes")
print("=" * 65)

# Definição dos novos casos clínicos (podem ser alterados para testes)
novos_pacientes = [
    {
        'nome': 'Roberto',
        'idade': 58,
        'glicose': 145.0,        # pré-diabético
        'pressao_arterial': 148.0,  # hipertensão estágio 2
        'imc': 33.5,             # obesidade grau I
        'colesterol': 252.0      # colesterol alto
    },
    {
        'nome': 'Carla',
        'idade': 28,
        'glicose': 88.0,         # normal
        'pressao_arterial': 112.0,  # normal
        'imc': 22.0,             # normal
        'colesterol': 175.0      # desejável
    },
    {
        'nome': 'Henrique',
        'idade': 45,
        'glicose': 118.0,        # pré-diabetes
        'pressao_arterial': 132.0,  # hipertensão leve
        'imc': 27.8,             # sobrepeso
        'colesterol': 215.0      # limítrofe
    }
]

# Mapeamento das classes para rótulos descritivos
LABELS_RISCO  = {0: 'BAIXO RISCO ✓', 1: 'RISCO MÉDIO ⚠', 2: 'ALTO RISCO ✗'}
CORES_RISCO   = {0: '🟢', 1: '🟡', 2: '🔴'}

for caso in novos_pacientes:

    # Extrai apenas as features (exclui o nome)
    features_caso = {k: v for k, v in caso.items() if k != 'nome'}

    # Cria DataFrame com a mesma estrutura do dataset de treino
    novo_df = pd.DataFrame([features_caso])

    # Aplica a MESMA normalização aprendida nos dados de treino
    # (usa o scaler já ajustado, não um novo scaler)
    novo_scaled = scaler.transform(novo_df)

    # Predição da classe
    classe_predita = melhor_modelo.predict(novo_scaled)[0]

    # Probabilidades para cada classe (confiança do modelo)
    probs = melhor_modelo.predict_proba(novo_scaled)[0]

    # Exibição dos resultados
    print(f"\n  Paciente: {caso['nome']}")
    print(f"  {'─' * 50}")
    print(f"  Dados informados:")
    for feat, val in features_caso.items():
        print(f"    {feat:<20}: {val}")

    print(f"\n  Resultado do modelo ({melhor_nome}):")
    print(f"  {CORES_RISCO[classe_predita]}  Classificação: {LABELS_RISCO[classe_predita]}")
    print(f"\n  Probabilidades por classe:")
    for i, (label, prob) in enumerate(zip(LABELS_RISCO.values(), probs)):
        barra = '█' * int(prob * 30)
        print(f"    Classe {i} — {label:<20}: {prob*100:5.1f}%  {barra}")

print("\n" + "=" * 65)
print("  PIPELINE CONCLUÍDO COM SUCESSO")
print("  Arquivos gerados:")
print("    → grafico_comparacao_modelos.png")
print("    → matriz_confusao.png")
print("    → curva_roc.png")
print("    → importancia_features.png")
print("=" * 65)
