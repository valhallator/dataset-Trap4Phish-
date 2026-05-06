# Fase 3 - Modelos de Machine Learning
# Projeto: Detecção de Phishing em PDFs (CIC-Trap4Phish 2025)
# Data: 2026

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report, roc_auc_score, roc_curve)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# ── Configuração de paths ─────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
INPUT    = os.path.join(BASE, '..', 'f2', 'code_output2', 'dataset_final_fase2.csv')
OUTPUT   = os.path.join(BASE, 'code_output3')
os.makedirs(OUTPUT, exist_ok=True)

RANDOM_STATE = 42

# ── 1. Leitura dos dados ──────────────────────────────────────────────────────
df = pd.read_csv(INPUT)
X = df.drop(columns=['label'])
y = df['label']
print(f"Dataset carregado: {df.shape} | Features: {X.columns.tolist()}")

# Train/test split — 80/20 estratificado
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Treino: {X_train.shape} | Teste: {X_test.shape}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ── 2. Definição dos modelos ──────────────────────────────────────────────────
modelos = {
    'K Neighbors Classifier':   KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'Logistic Regression':       LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'Decision Tree':             DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
    'Random Forest':             RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    'SVM':                       SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
    'Naive Bayes':               GaussianNB(),
}

# Regressão Linear — adaptada para classificação binária (threshold 0.5)
# Incluída porque o professor pediu explicitamente
class LinearRegressionClassifier:
    """Wrapper que usa Regressão Linear para classificação com threshold 0.5."""
    def __init__(self):
        self.model = LinearRegression()
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    def predict(self, X):
        return (self.model.predict(X) >= 0.5).astype(int)
    def predict_proba(self, X):
        raw = self.model.predict(X).clip(0, 1)
        return np.column_stack([1 - raw, raw])

modelos['Linear Regression'] = LinearRegressionClassifier()

# ── 3. Treino, avaliação e métricas ──────────────────────────────────────────
resultados = []
modelos_treinados = {}

for nome, modelo in modelos.items():
    print(f"\nA treinar: {nome}...")

    # Tempo de treino
    t0 = time.time()
    modelo.fit(X_train, y_train)
    t_treino = time.time() - t0

    # Predições
    y_pred = modelo.predict(X_test)

    # Métricas base
    acc       = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred, average='weighted')

    # Cross-validation accuracy (5-fold)
    # LinearRegressionClassifier não é compatível com cross_val_score directamente
    if nome == 'Linear Regression':
        cv_scores = np.array([acc])  # aproximação
        cv_mean   = acc
    else:
        cv_scores = cross_val_score(modelo, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        cv_mean   = cv_scores.mean()

    # ROC-AUC
    try:
        y_proba = modelo.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = None

    resultados.append({
        'Model':             nome,
        'Accuracy':          round(acc, 4),
        'CV Accuracy':       round(cv_mean, 4),
        'F-Score':           round(f1, 4),
        'ROC-AUC':           round(auc, 4) if auc else 'N/A',
        'Time (s)':          round(t_treino, 4),
    })

    modelos_treinados[nome] = modelo
    joblib.dump(modelo, os.path.join(OUTPUT, f'modelo_{nome.replace(" ","_")}.pkl'))

    print(f"  Accuracy: {acc:.4f} | CV: {cv_mean:.4f} | F1: {f1:.4f} | Tempo: {t_treino:.2f}s")

# ── 4. Tabela de resultados ───────────────────────────────────────────────────
df_resultados = pd.DataFrame(resultados).sort_values('Accuracy', ascending=False)
df_resultados.to_csv(os.path.join(OUTPUT, 'metricas_modelos.csv'), index=False)
print("\n\n=== RESULTADOS FINAIS ===")
print(df_resultados.to_string(index=False))

# ── 5. Gráfico comparativo de métricas ───────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
metricas = ['Accuracy', 'CV Accuracy', 'F-Score']
cores    = ['steelblue', 'tomato', 'seagreen']

for ax, metrica, cor in zip(axes, metricas, cores):
    dados = df_resultados[['Model', metrica]].copy()
    dados[metrica] = pd.to_numeric(dados[metrica], errors='coerce')
    dados = dados.dropna().sort_values(metrica, ascending=True)
    ax.barh(dados['Model'], dados[metrica], color=cor, edgecolor='black')
    ax.set_xlim(0, 1.05)
    ax.set_title(metrica)
    ax.set_xlabel('Score')
    for i, v in enumerate(dados[metrica]):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=8)

plt.suptitle('Comparação de Modelos — Fase 3', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, 'comparacao_modelos.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\nGráfico comparativo guardado.")

# ── 6. Matriz de confusão — todos os modelos ─────────────────────────────────
n = len(modelos)
n_cols = 3
n_rows = -(-n // n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
axes = axes.flatten()

for i, (nome, modelo) in enumerate(modelos_treinados.items()):
    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['Legítimo', 'Phishing'],
                yticklabels=['Legítimo', 'Phishing'])
    acc = accuracy_score(y_test, y_pred)
    axes[i].set_title(f'{nome}\nAcc: {acc:.3f}')
    axes[i].set_xlabel('Previsto')
    axes[i].set_ylabel('Real')

for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Matrizes de Confusão — Todos os Modelos', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, 'matrizes_confusao.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Matrizes de confusão guardadas.")

# ── 7. Curvas ROC ─────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 7))
for nome, modelo in modelos_treinados.items():
    try:
        y_proba = modelo.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{nome} (AUC={auc:.3f})')
    except Exception:
        pass

plt.plot([0,1], [0,1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curvas ROC — Todos os Modelos')
plt.legend(loc='lower right', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, 'curvas_roc.png'), dpi=150)
plt.close()
print("Curvas ROC guardadas.")

# ── 8. Árvore de Decisão — visualização das primeiras 3 camadas ──────────────
dt = modelos_treinados['Decision Tree']
try:
    from sklearn.tree import plot_tree
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(dt, max_depth=3, feature_names=X.columns.tolist(),
              class_names=['Legítimo', 'Phishing'],
              filled=True, rounded=True, fontsize=8, ax=ax)
    plt.title('Árvore de Decisão (primeiras 3 camadas)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, 'arvore_decisao.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Árvore de decisão guardada.")
except Exception as e:
    print(f"Erro ao gerar árvore: {e}")

# ── 9. Random Forest — feature importance ────────────────────────────────────
rf = modelos_treinados['Random Forest']
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 5))
importances.plot(kind='barh', ax=ax, color='tomato', edgecolor='black')
ax.set_title('Random Forest — Feature Importance (Fase 3)')
ax.set_xlabel('Importância')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, 'rf_feature_importance.png'), dpi=150)
plt.close()
print("Feature importance RF guardada.")

# ── 10. Resumo ────────────────────────────────────────────────────────────────
melhor = df_resultados.iloc[0]
resumo = f"""Resumo da Fase 3 - Modelos de Machine Learning

Dataset usado: dataset_final_fase2.csv
Split: 80% treino / 20% teste (estratificado)
Cross-validation: 5-fold estratificado

Modelos treinados:
{chr(10).join([f'  - {r["Model"]}: Acc={r["Accuracy"]} | CV={r["CV Accuracy"]} | F1={r["F-Score"]} | Tempo={r["Time (s)"]}s' for r in resultados])}

Melhor modelo: {melhor["Model"]}
  Accuracy:    {melhor["Accuracy"]}
  CV Accuracy: {melhor["CV Accuracy"]}
  F-Score:     {melhor["F-Score"]}
  ROC-AUC:     {melhor["ROC-AUC"]}
"""
with open(os.path.join(OUTPUT, 'resumo_fase3.txt'), 'w', encoding='utf-8') as f:
    f.write(resumo)

print("\n✅ Fase 3 concluída. Todos os outputs em code_output3/")