# Fase 4 - Simulação de Uso dos Modelos (Model Usage)
# Projeto: Detecção de Phishing em PDFs (CIC-Trap4Phish 2025)
# Data: 2026

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression

# Necessário para deserializar o modelo guardado na Fase 3
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

# =========================
# Configuração de paths
# =========================
BASE   = os.path.dirname(os.path.abspath(__file__))
DIR_F2 = os.path.join(BASE, '..', 'f2', 'code_output2')
DIR_F3 = os.path.join(BASE, '..', 'f3', 'code_output3')
OUTPUT = os.path.join(BASE, 'code_output4')
os.makedirs(OUTPUT, exist_ok=True)

# =========================
# 1. Carregar scaler e modelos
# =========================
scaler = joblib.load(os.path.join(DIR_F2, 'scaler.pkl'))

nomes_modelos = [
    'K_Neighbors_Classifier',
    'Logistic_Regression',
    'Decision_Tree',
    'Random_Forest',
    'SVM',
    'Naive_Bayes',
    'Linear_Regression',
]

modelos = {}
for nome in nomes_modelos:
    path = os.path.join(DIR_F3, f'modelo_{nome}.pkl')
    if os.path.exists(path):
        modelos[nome] = joblib.load(path)
        print(f"Modelo carregado: {nome}")
    else:
        print(f"[AVISO] Modelo não encontrado: {path}")

TOP_FEATURES = ['text_length', 'object_count', 'total_filters',
                'metadata_size', 'file_size', 'title_chars']

# =========================
# 2. Casos de simulação
# =========================
casos = {
    'Caso_A_Legitimo': {
        'text_length':   50000,
        'object_count':  150,
        'total_filters': 10,
        'metadata_size': 250,
        'file_size':     800000,
        'title_chars':   35,
    },
    'Caso_B_Phishing': {
        'text_length':   0,
        'object_count':  5,
        'total_filters': 0,
        'metadata_size': 0,
        'file_size':     15000,
        'title_chars':   0,
    },
    'Caso_C_Ambiguo': {
        'text_length':   1200,
        'object_count':  80,
        'total_filters': 5,
        'metadata_size': 80,
        'file_size':     120000,
        'title_chars':   10,
    },
}

# =========================
# 3. Pré-processamento
# =========================
def preprocessar(valores_originais: dict) -> np.ndarray:
    """
    Aplica o mesmo pipeline da Fase 2:
    1. log1p
    2. RobustScaler (scaler guardado na Fase 2)
    """
    features_scaler = ['text_length', 'total_filters', 'title_chars',
                       'file_size', 'object_count', 'stream_count',
                       'metadata_size', 'entropy_of_streams']

    df_full = pd.DataFrame(0.0, index=[0], columns=features_scaler)

    for col in TOP_FEATURES:
        if col in df_full.columns:
            df_full[col] = np.log1p(valores_originais[col])

    df_scaled = pd.DataFrame(
        scaler.transform(df_full),
        columns=features_scaler
    )

    return df_scaled[TOP_FEATURES].values

# =========================
# 4. Classificação
# =========================
resultados_simulacao = []

for nome_caso, valores in casos.items():
    print(f"\n{'='*55}")
    print(f"Simulação: {nome_caso}")
    print(f"Valores originais: {valores}")

    X_caso = preprocessar(valores)
    votos = []

    for nome_modelo, modelo in modelos.items():
        pred = modelo.predict(X_caso)[0]
        e_phishing = int(pred >= 0.5)
        label = 'PHISHING' if e_phishing else 'LEGÍTIMO'
        votos.append(e_phishing)

        try:
            proba = modelo.predict_proba(X_caso)[0]
            confianca = f"{max(proba)*100:.1f}%"
        except Exception:
            confianca = "N/A"

        print(f"  {nome_modelo:<30} → {label:<10} (confiança: {confianca})")

        resultados_simulacao.append({
            'Caso':     nome_caso,
            'Modelo':   nome_modelo,
            'Predicao': e_phishing,
            'Label':    label,
        })

    voto_final = 'PHISHING' if sum(votos) > len(votos) / 2 else 'LEGÍTIMO'
    print(f"  {'VOTO MAIORITÁRIO':<30} → {voto_final}")

# =========================
# 5. Guardar CSV
# =========================
df_sim = pd.DataFrame(resultados_simulacao)
df_sim.to_csv(os.path.join(OUTPUT, 'resultados_simulacao.csv'), index=False)

# =========================
# 6. Heatmap de predições
# =========================
pivot = df_sim.pivot(index='Modelo', columns='Caso', values='Predicao')

fig, ax = plt.subplots(figsize=(10, 6))
cmap = plt.cm.get_cmap('RdYlGn_r', 2)
ax.imshow(pivot.values, cmap=cmap, vmin=0, vmax=1, aspect='auto')

ax.set_xticks(range(len(pivot.columns)))
ax.set_yticks(range(len(pivot.index)))
ax.set_xticklabels([c.replace('_', ' ') for c in pivot.columns], fontsize=11)
ax.set_yticklabels([m.replace('_', ' ') for m in pivot.index], fontsize=10)

for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        texto = 'PHISHING' if val == 1 else 'LEGÍTIMO'
        cor = 'white' if val == 1 else 'black'
        ax.text(j, i, texto, ha='center', va='center',
                fontsize=10, fontweight='bold', color=cor)

legenda = [
    mpatches.Patch(color='#d73027', label='Phishing (1)'),
    mpatches.Patch(color='#1a9850', label='Legítimo (0)'),
]
ax.legend(handles=legenda, loc='upper right', bbox_to_anchor=(1.3, 1))
ax.set_title('Simulação de Classificação — Todos os Modelos', fontsize=13, pad=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, 'simulacao_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\nHeatmap guardado.")

# =========================
# 7. Resumo escrito
# =========================
linhas = ["Resumo da Fase 4 - Simulação de Uso dos Modelos\n"]
linhas.append("Pipeline aplicado:")
linhas.append("  1. Valores originais do PDF")
linhas.append("  2. Transformação log1p")
linhas.append("  3. RobustScaler (da Fase 2)")
linhas.append("  4. Predição com modelos da Fase 3\n")

for nome_caso, valores in casos.items():
    linhas.append(f"Caso: {nome_caso}")
    linhas.append(f"  Valores: {valores}")
    preds = df_sim[df_sim['Caso'] == nome_caso]
    for _, row in preds.iterrows():
        linhas.append(f"  {row['Modelo']}: {row['Label']}")
    votos_caso = preds['Predicao'].sum()
    voto = 'PHISHING' if votos_caso > len(preds) / 2 else 'LEGÍTIMO'
    linhas.append(f"  VOTO MAIORITÁRIO: {voto}\n")

with open(os.path.join(OUTPUT, 'resumo_fase4.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(linhas))

print("\n✅ Fase 4 concluída. Todos os outputs em code_output4/")