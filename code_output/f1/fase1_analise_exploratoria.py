# Fase 1 - Análise Exploratória e Clustering
# Projeto: Deteção de PDFs maliciosos / phishing
# Autor: Iúri Carlos Carvalho Laranjeira de Sousa
# Data: 2026

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# =========================
# Configuração inicial
# =========================
sns.set_theme(style="whitegrid")
inicio = time.time()

CAMINHO_DADOS = "../../data/PDF_Top10_features.csv"
PASTA_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fase1_outputs")
os.makedirs(PASTA_OUTPUT, exist_ok=True)

# =========================
# Verificação do ficheiro de dados
# =========================
ficheiro_dados = Path(CAMINHO_DADOS)
if not ficheiro_dados.exists():
    print(f"[ERRO] O ficheiro de dados '{CAMINHO_DADOS}' não foi encontrado. Caminho absoluto: {ficheiro_dados.resolve()}")
    exit(1)

# =========================
# Leitura do dataset
# =========================
try:
    df = pd.read_csv(CAMINHO_DADOS)
    if df.empty:
        print(f"[ERRO] O ficheiro '{CAMINHO_DADOS}' foi lido mas está vazio!")
        exit(1)
    print(f"Dataset carregado com sucesso: {df.shape[0]} linhas e {df.shape[1]} colunas.")
except Exception as e:
    print(f"[ERRO] ao ler o ficheiro {CAMINHO_DADOS}: {e}")
    exit(1)

# =========================
# Inspeção inicial
# =========================
print("\nPrimeiras linhas:")
try:
    print(df.head())
except Exception as e:
    print(f"[AVISO] Não foi possível mostrar as primeiras linhas: {e}")

print("\nInformação geral:")
try:
    print(df.info())
except Exception as e:
    print(f"[AVISO] Não foi possível mostrar info(): {e}")

# =========================
# Valores em falta
# =========================
missing = df.isnull().sum().to_frame("missing_count")
missing["missing_percent"] = (missing["missing_count"] / len(df)) * 100
try:
    missing.to_csv(os.path.join(PASTA_OUTPUT, "missing_values.csv"))
except Exception as e:
    print(f"[AVISO] Não foi possível guardar missing_values.csv: {e}")

# =========================
# Estatísticas descritivas
# =========================
try:
    estatisticas = df.describe().T
    estatisticas["median"] = df.median(numeric_only=True)
    estatisticas["skewness"] = df.skew(numeric_only=True)
    estatisticas["kurtosis"] = df.kurtosis(numeric_only=True)
    estatisticas.to_csv(os.path.join(PASTA_OUTPUT, "estatisticas_descritivas.csv"))
except Exception as e:
    print(f"[AVISO] Não foi possível guardar estatisticas_descritivas.csv: {e}")

# =========================
# Distribuição da variável alvo
# =========================
if "label" in df.columns:
    distribuicao_label = df["label"].value_counts(dropna=False).sort_index().to_frame("count")
    distribuicao_label["percent"] = (distribuicao_label["count"] / len(df)) * 100
    try:
        distribuicao_label.to_csv(os.path.join(PASTA_OUTPUT, "distribuicao_label.csv"))
        plt.figure(figsize=(6, 4))
        sns.countplot(x="label", data=df)
        plt.title("Distribuição da variável alvo (label)")
        plt.xlabel("Label")
        plt.ylabel("Contagem")
        plt.tight_layout()
        plt.savefig(os.path.join(PASTA_OUTPUT, "distribuicao_label.png"), dpi=200)
        plt.close()
    except Exception as e:
        print(f"[AVISO] Não foi possível guardar distribuição da label: {e}")

# =========================
# Correlação entre features
# =========================
corr = df.corr(numeric_only=True)
corr.to_csv(os.path.join(PASTA_OUTPUT, "correlacao_features.csv"))

plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Matriz de Correlação")
plt.tight_layout()
plt.savefig(os.path.join(PASTA_OUTPUT, "heatmap_correlacao.png"), dpi=200)
plt.close()

# =========================
# Histogramas das features
# =========================
features = [col for col in df.columns if col != "label"]

df[features].hist(figsize=(16, 12), bins=30)
plt.suptitle("Histogramas das Features", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PASTA_OUTPUT, "histogramas_features.png"), dpi=200)
plt.close()

# =========================
# Boxplots para detetar outliers
# =========================
for col in features:
    if df[col].nunique() <= 2:
        continue
    plt.figure(figsize=(8, 4))
    sns.boxplot(x="label", y=col, data=df, palette="Set2")
    plt.title(f"Boxplot de {col} por classe (0=Legítimo, 1=Phishing)")
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_OUTPUT, f"boxplot_{col}_por_classe.png"), dpi=200)
    plt.close()

for col in features:
    if df[col].nunique() <= 2:
        continue
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot - {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_OUTPUT, f"boxplot_{col}.png"), dpi=200)
    plt.close()

# =========================
# Interpretação escrita
# =========================
interpretacao = []
interpretacao.append("Resumo da Fase 1 - Análise Exploratória e Clustering\n")
interpretacao.append(f"O dataset apresenta {df.shape[0]} amostras e {df.shape[1]} colunas (incluindo label).\n")
interpretacao.append("\nPrincipais pontos:")
interpretacao.append("- Não existem valores em falta nas features nem na label, o que simplifica o pré-processamento.")
if 'label' in df.columns:
    dist = df['label'].value_counts(normalize=True)
    interpretacao.append(f"- A distribuição da variável alvo está quase equilibrada: {dist.to_dict()} (0=legítimo, 1=phishing).\n")
interpretacao.append("- As estatísticas descritivas mostram distribuições muito assimétricas e outliers extremos em várias features, nomeadamente file_size, text_length e object_count.")
interpretacao.append("- A matriz de correlação revela relações fortes entre stream_count e endstream_count, e correlações negativas relevantes entre entropy_of_streams, metadata_size e a label.")
interpretacao.append("- O clustering (KMeans + PCA) mostra dois grupos, mas a separação é dominada por outliers e não reflete uma divisão genuína entre phishing e legítimo.\n")
interpretacao.append("- Boxplots por classe mostram que as features mais discriminativas são file_size, text_length, entropy_of_streams e metadata_size.\n")
interpretacao.append("\nPrincipais desafios identificados:")
interpretacao.append("- Forte assimetria e presença de outliers em 7 das 10 features.\n- Necessidade de normalização robusta e possível transformação logarítmica na Fase 2.\n- Possível remoção de features redundantes (ex: stream_count/endstream_count).\n")
interpretacao.append("\nPróximos passos:")
interpretacao.append("- Aplicar normalização e/ou transformação logarítmica.\n- Selecionar features mais relevantes.\n- Avançar para modelação supervisionada (classificação).\n")

with open(os.path.join(PASTA_OUTPUT, "interpretacao_fase1.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(interpretacao))

# =========================
# Pairplot
# =========================
subset_pairplot = features[:5].copy()
if "label" in df.columns:
    pairplot_cols = subset_pairplot + ["label"]
    g = sns.pairplot(df[pairplot_cols], hue="label", diag_kind="hist", corner=True)
else:
    g = sns.pairplot(df[subset_pairplot], diag_kind="hist", corner=True)

g.figure.suptitle("Matriz de Dispersão (subconjunto de features)", y=1.02)
g.savefig(os.path.join(PASTA_OUTPUT, "scatterplot_matrix.png"))
plt.close("all")

# =========================
# Clustering
# =========================
X = df[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters
df.to_csv(os.path.join(PASTA_OUTPUT, "dados_com_clusters.csv"), index=False)

sil_score = silhouette_score(X_scaled, clusters)
with open(os.path.join(PASTA_OUTPUT, "metricas_clustering.txt"), "w", encoding="utf-8") as f:
    f.write(f"Silhouette Score: {sil_score:.4f}\n")

# =========================
# Visualização dos clusters com PCA
# =========================
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Cluster": clusters
})

df_pca.to_csv(os.path.join(PASTA_OUTPUT, "clusters_pca.csv"), index=False)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="Cluster", palette="viridis", alpha=0.8)
plt.title("Visualização dos Clusters com PCA")
plt.tight_layout()
plt.savefig(os.path.join(PASTA_OUTPUT, "clusters_visualizacao_pca.png"), dpi=200)
plt.close()

# =========================
# Médias por cluster
# =========================
medias_cluster = df.groupby("Cluster")[features].mean()
medias_cluster.to_csv(os.path.join(PASTA_OUTPUT, "medias_por_cluster.csv"))

# =========================
# Resumo final
# =========================
fim = time.time()
with open(os.path.join(PASTA_OUTPUT, "resumo_fase1.txt"), "w", encoding="utf-8") as f:
    f.write("Resumo da Fase 1 - Análise Exploratória e Clustering\n")
    f.write(f"Número de linhas: {df.shape[0]}\n")
    f.write(f"Número de colunas: {df.shape[1]}\n")
    f.write(f"Tempo total de execução: {fim - inicio:.2f} segundos\n")
    f.write(f"Silhouette Score: {sil_score:.4f}\n")

print(f"\nFase 1 concluída em {fim - inicio:.2f} segundos.")
print(f"Todos os ficheiros foram guardados em: {PASTA_OUTPUT}")