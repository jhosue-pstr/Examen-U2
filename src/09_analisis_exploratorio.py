from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "dataset_enaho_escale_2025.parquet"
OUTPUT_DIR = BASE_DIR / "informes" / "resultados_exploracion"


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"No se encontró {DATA_PATH}. Ejecuta primero la etapa de integración."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(DATA_PATH)
    required = [
        "edad",
        "exclusion_digital",
        "internet_hogar",
        "lengua_materna",
        "dominio",
        "estrato",
    ]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Faltan columnas para el análisis exploratorio: {missing}")

    data[required].describe(include="all").transpose().to_csv(
        OUTPUT_DIR / "resumen_variables.csv",
        encoding="utf-8-sig",
    )
    target_counts = (
        data["exclusion_digital"]
        .value_counts(dropna=False)
        .rename_axis("clase")
        .reset_index(name="casos")
    )
    target_counts["proporcion"] = target_counts["casos"] / len(data)
    target_counts.to_csv(
        OUTPUT_DIR / "distribucion_etiqueta.csv",
        index=False,
        encoding="utf-8-sig",
    )

    missingness = (
        data.isna()
        .sum()
        .rename("faltantes")
        .to_frame()
        .assign(proporcion=lambda frame: frame["faltantes"] / len(data))
        .sort_values("proporcion", ascending=False)
    )
    missingness.to_csv(
        OUTPUT_DIR / "valores_faltantes.csv",
        encoding="utf-8-sig",
    )

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    data["edad"].dropna().hist(ax=axes[0], bins=12)
    axes[0].set(title="Edad de la cohorte", xlabel="Edad", ylabel="Estudiantes")
    data["exclusion_digital"].dropna().astype(int).value_counts().sort_index().plot.bar(
        ax=axes[1]
    )
    axes[1].set(
        title="Etiqueta operacional",
        xlabel="No usa Internet pese a asistir (0/1)",
        ylabel="Estudiantes",
    )
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "distribuciones_iniciales.png", dpi=160)
    plt.close(figure)

    print(f"Filas: {len(data):,} | columnas: {len(data.columns)}")
    print("Distribución de la etiqueta:")
    print(target_counts.to_string(index=False))
    print(f"Exploración guardada en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
