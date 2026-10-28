from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "dataset_enaho_escale_2025.parquet"
OUTPUT_DIR = BASE_DIR / "informes" / "resultados_clustering"
RANDOM_STATE = 42
MAX_CLUSTER_ROWS = 5_000
K_VALUES = range(2, 7)

NUMERIC_FEATURES = [
    "edad",
    "ingreso_per_capita",
    "gasto_per_capita",
    "escale_ratio_alumnos_docente",
    "escale_proporcion_publica",
]
CATEGORICAL_FEATURES = [
    "sexo",
    "lengua_materna",
    "electricidad_hogar",
    "internet_hogar",
    "dominio",
    "estrato",
]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No existe el dataset integrado: {DATA_PATH}. "
            "Ejecuta primero src/05_integrar_enaho_escale.py."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_parquet(DATA_PATH)
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing_columns = sorted(set(features) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Faltan variables para clustering: {missing_columns}")

    if len(data) > MAX_CLUSTER_ROWS:
        sample = data.sample(MAX_CLUSTER_ROWS, random_state=RANDOM_STATE).copy()
    else:
        sample = data.copy()

    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    matrix = preprocessor.fit_transform(sample[features])
    if not np.isfinite(matrix).all():
        raise ValueError("El preprocesamiento produjo valores no finitos.")

    metrics = []
    selected_k = None
    best_silhouette = -np.inf
    for k in K_VALUES:
        kmeans_labels = KMeans(
            n_clusters=k, n_init=10, random_state=RANDOM_STATE
        ).fit_predict(matrix)
        agglomerative_labels = AgglomerativeClustering(
            n_clusters=k, linkage="ward"
        ).fit_predict(matrix)
        for method, labels in [
            ("KMeans", kmeans_labels),
            ("Agglomerative_Ward", agglomerative_labels),
        ]:
            row = {
                "metodo": method,
                "k": k,
                "silhouette": silhouette_score(
                    matrix,
                    labels,
                    sample_size=min(3_000, len(matrix)),
                    random_state=RANDOM_STATE,
                ),
                "davies_bouldin": davies_bouldin_score(matrix, labels),
                "calinski_harabasz": calinski_harabasz_score(matrix, labels),
            }
            metrics.append(row)
            if method == "KMeans" and row["silhouette"] > best_silhouette:
                best_silhouette = row["silhouette"]
                selected_k = k

    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(
        OUTPUT_DIR / "metricas_clustering.csv",
        index=False,
        encoding="utf-8-sig",
    )

    final_model = KMeans(
        n_clusters=selected_k, n_init=10, random_state=RANDOM_STATE
    ).fit(matrix)
    sample["cluster"] = final_model.labels_
    sample.to_parquet(
        OUTPUT_DIR / "muestra_con_clusters.parquet",
        index=False,
        compression="snappy",
    )

    profiles = sample.groupby("cluster")[NUMERIC_FEATURES].median().reset_index()
    for column in CATEGORICAL_FEATURES:
        dominant = (
            sample.groupby("cluster")[column]
            .agg(lambda values: values.mode().iloc[0] if not values.mode().empty else pd.NA)
            .rename(f"{column}_moda")
        )
        profiles = profiles.merge(dominant, on="cluster", validate="one_to_one")
    profiles["cantidad_muestra"] = sample["cluster"].value_counts().sort_index().values
    income_bins = min(3, len(profiles))
    ratio_bins = min(3, len(profiles))
    profile_labels = ["bajo", "medio", "alto"][:income_bins]
    profiles["ingreso_relativo"] = pd.qcut(
        profiles["ingreso_per_capita"].rank(method="first"),
        q=income_bins,
        labels=profile_labels,
    ).astype(str)
    profiles["ratio_alumnos_docente_relativo"] = pd.qcut(
        profiles["escale_ratio_alumnos_docente"].rank(method="first"),
        q=ratio_bins,
        labels=["bajo", "medio", "alto"][:ratio_bins],
    ).astype(str)
    profiles["perfil_relativo"] = (
        "Ingreso "
        + profiles["ingreso_relativo"]
        + " / ratio alumno-docente "
        + profiles["ratio_alumnos_docente_relativo"]
    )
    profiles.to_csv(
        OUTPUT_DIR / "perfiles_clusters.csv",
        index=False,
        encoding="utf-8-sig",
    )

    silhouette_values = silhouette_samples(matrix, final_model.labels_)
    figure, axis = plt.subplots(figsize=(8, 6))
    lower = 10
    for cluster_id in range(selected_k):
        cluster_values = np.sort(silhouette_values[final_model.labels_ == cluster_id])
        upper = lower + len(cluster_values)
        axis.fill_betweenx(
            np.arange(lower, upper),
            0,
            cluster_values,
            alpha=0.75,
            label=f"Cluster {cluster_id}",
        )
        axis.text(-0.05, lower + len(cluster_values) / 2, str(cluster_id))
        lower = upper + 10
    axis.axvline(
        float(silhouette_values.mean()),
        color="red",
        linestyle="--",
        label="Promedio",
    )
    axis.set(
        xlabel="Coeficiente de silueta",
        ylabel="Observaciones ordenadas por cluster",
        title=f"Gráfico de silueta K-Means (k={selected_k})",
    )
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "silueta_kmeans.png", dpi=160)
    plt.close(figure)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    projected = pca.fit_transform(matrix)
    figure, axis = plt.subplots(figsize=(9, 6))
    scatter = axis.scatter(
        projected[:, 0],
        projected[:, 1],
        c=sample["cluster"],
        cmap="tab10",
        s=12,
        alpha=0.65,
    )
    axis.set_title(f"K-Means, k={selected_k}; PCA de variables preprocesadas")
    axis.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axis.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    figure.colorbar(scatter, ax=axis, label="Cluster")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "clusters_pca.png", dpi=160)
    plt.close(figure)

    print(f"Filas analizadas: {len(sample):,} (muestra reproducible)")
    print(f"k seleccionado para K-Means por Silhouette: {selected_k}")
    print(metrics_frame.to_string(index=False))
    print(f"Resultados: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
