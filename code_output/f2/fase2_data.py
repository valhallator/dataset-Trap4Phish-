# Fase 2 - Tratamento de Dados, Normalização e Balanceamento
# Projeto: Detecção de Phishing em PDFs (CIC-Trap4Phish 2025)
# Data: 2026

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# =========================
# Configuração de paths
# =========================
RANDOM_STATE = 42
CAMINHO_DADOS = '../../data/PDF_Top10_features.csv'
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code_output2")
os.makedirs(output_dir, exist_ok=True)

# ── 1. Leitura ────────────────────────────────────────────────────────────────
df = pd.read_csv(CAMINHO_DADOS)
features = [c for c in df.columns if c != 'label']
X = df[features].copy()
y = df['label'].copy()
print(f"Dataset original: {df.shape}")

# ── 2. Remoção de features redundantes ───────────────────────────────────────
X = X.drop(columns=['endstream_count'])
features = list(X.columns)
print(f"Features após remoção de redundâncias: {features}")

# ── 3. Clipagem de outliers extremos (IQR × 3) ───────────────────────────────
features_continuas = [f for f in features if f != 'valid_pdf_header']

limites = {}
for col in features_continuas:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 3 * IQR
    upper = Q3 + 3 * IQR
    limites[col] = (lower, upper)
    X[col] = X[col].clip(lower=lower, upper=upper)

print("Clipagem de outliers aplicada (IQR × 3).")
pd.DataFrame(limites, index=['lower', 'upper']).to_csv(os.path.join(output_dir, 'limites_outliers.csv'))

# ── 4. Transformação logarítmica ──────────────────────────────────────────────
features_log = ['text_length', 'total_filters', 'title_chars',
                'file_size', 'object_count', 'stream_count',
                'metadata_size']

for col in features_log:
    X[col] = np.log1p(X[col])

print("Transformação log1p aplicada.")

df_original = pd.read_csv(CAMINHO_DADOS).drop(columns=['endstream_count', 'label'])
skew_antes  = df_original[features_log].skew()
skew_depois = X[features_log].skew()

skew_report = pd.DataFrame({'skew_antes': skew_antes, 'skew_depois': skew_depois})
skew_report.to_csv(os.path.join(output_dir, 'skewness_comparacao.csv'))
print("\nSkewness antes vs depois da transformação log:")
print(skew_report.round(2))

# Visualização distribuições pós-transformação
n_cols = 3
n_rows = -(-len(features) // n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
axes = axes.flatten()
for i, col in enumerate(features):
    axes[i].hist(X[col], bins=40, color='steelblue', alpha=0.8)
    axes[i].set_title(f'{col} (pós-transformação)')
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle('Distribuições após Clipagem e Log1p', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'distribuicoes_apos_transformacao.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Distribuições pós-transformação guardadas.")

# ── 5. Normalização com RobustScaler ─────────────────────────────────────────
features_escalar = [f for f in features if f != 'valid_pdf_header']

scaler = RobustScaler()
X_scaled = X.copy()
X_scaled[features_escalar] = scaler.fit_transform(X[features_escalar])

joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
print("RobustScaler aplicado e guardado.")

# ── 6. Balanceamento com SMOTE ────────────────────────────────────────────────
print(f"\nDistribuição antes do SMOTE: {dict(y.value_counts())}")

smote = SMOTE(random_state=RANDOM_STATE)
X_bal, y_bal = smote.fit_resample(X_scaled, y)

print(f"Distribuição após SMOTE:     {dict(pd.Series(y_bal).value_counts())}")

df_balanced = pd.DataFrame(X_bal, columns=features)
df_balanced['label'] = y_bal
df_balanced.to_csv(os.path.join(output_dir, 'dataset_balanceado.csv'), index=False)
print("Dataset balanceado guardado.")

fig, ax = plt.subplots(figsize=(5, 4))
pd.Series(y_bal).value_counts().plot(kind='bar', ax=ax,
    color=['steelblue', 'tomato'], edgecolor='black')
ax.set_title('Distribuição das Classes após SMOTE')
ax.set_xlabel('Classe (0=Legítimo, 1=Phishing)')
ax.set_ylabel('Número de Amostras')
ax.set_xticklabels(['Legítimo (0)', 'Phishing (1)'], rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'distribuicao_apos_smote.png'), dpi=150)
plt.close()

# ── 7. Seleção de Features ────────────────────────────────────────────────────
selector_kbest = SelectKBest(f_classif, k='all')
selector_kbest.fit(X_bal, y_bal)
scores_kbest = pd.Series(selector_kbest.scores_, index=features).sort_values(ascending=False)

rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_bal, y_bal)
scores_rf = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

feature_scores = pd.DataFrame({
    'ANOVA_F_score': scores_kbest,
    'RF_importance': scores_rf
}).sort_values('RF_importance', ascending=False)
feature_scores.to_csv(os.path.join(output_dir, 'feature_selection_scores.csv'))
print("\nScores de seleção de features:")
print(feature_scores.round(4))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
scores_kbest.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
axes[0].set_title('SelectKBest — ANOVA F-score')
axes[0].set_ylabel('F-score')
axes[0].tick_params(axis='x', rotation=45)
scores_rf.plot(kind='bar', ax=axes[1], color='tomato', edgecolor='black')
axes[1].set_title('Random Forest — Feature Importance')
axes[1].set_ylabel('Importância')
axes[1].tick_params(axis='x', rotation=45)
plt.suptitle('Seleção de Features — Dois Métodos', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_selection.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Gráfico de seleção de features guardado.")

top_features = scores_rf[scores_rf > 0.05].index.tolist()
print(f"\nTop features selecionadas (RF importance > 0.05): {top_features}")

df_final = pd.DataFrame(X_bal, columns=features)[top_features].copy()
df_final['label'] = y_bal
df_final.to_csv(os.path.join(output_dir, 'dataset_final_fase2.csv'), index=False)
print(f"Dataset final guardado: {df_final.shape}")

# ── 8. Resumo ─────────────────────────────────────────────────────────────────
resumo = f"""Resumo da Fase 2 - Tratamento de Dados

Feature removida (redundância): endstream_count (correlação ~1.0 com stream_count)

Outliers: Clipagem IQR × 3 aplicada a {len(features_continuas)} features contínuas

Transformação log1p aplicada a: {features_log}

Normalização: RobustScaler (usa mediana e IQR, robusto a outliers)

Balanceamento SMOTE:
  - Antes: {{0: 9297, 1: 9999}}
  - Depois: balanceado (50/50)

Seleção de features:
  - Método 1: ANOVA F-score (SelectKBest)
  - Método 2: Random Forest Feature Importance
  - Top features: {top_features}

Dataset final: dataset_final_fase2.csv
"""
with open(os.path.join(output_dir, 'resumo_fase2.txt'), 'w', encoding='utf-8') as f:
    f.write(resumo)

print("\n✅ Fase 2 concluída. Todos os outputs guardados.")