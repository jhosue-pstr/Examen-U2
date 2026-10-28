from pathlib import Path
import os
import subprocess
import sys

import pandas as pd


# Rutas principales del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_ENAHO = PROJECT_ROOT / "data" / "raw" / "enaho"
RAW_ESCALE = PROJECT_ROOT / "data" / "raw" / "minedu"

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_enaho_escale_2025.parquet"
INTEGRATION_REPORT = PROJECT_ROOT / "data" / "processed" / "reporte_integracion_2025.csv"

CLUSTER_DIR = PROJECT_ROOT / "informes" / "resultados_clustering"
CLASSIFICATION_DIR = PROJECT_ROOT / "informes" / "resultados_clasificacion"


# Archivos originales necesarios para iniciar el proceso
REQUIRED_INPUTS = [
    RAW_ENAHO / "enaho01-2021-2025-100-panel.csv",
    RAW_ENAHO / "enaho01-2021-2025-200-panel.csv",
    RAW_ENAHO / "enaho01a-2021-2025-300-panel.csv",
    RAW_ENAHO / "sumaria-2021-2025-panel.csv",
    RAW_ESCALE / "Padron.dbf",
]


# Etapas que se ejecutarán en orden
STAGES = [
    (
        "1/5 - Preparar e integrar datos",
        "Limpiar y unir ENAHO con ESCALE, filtrar estudiantes "
        "de 6 a 17 años y crear la etiqueta de estudio.",
        "src/05_integrar_enaho_escale.py",
    ),
    (
        "2/5 - Validar base integrada",
        "Comprobar que los datos estén completos, que no existan "
        "estudiantes duplicados y que la etiqueta esté disponible.",
        None,
    ),
    (
        "3/5 - Explorar datos",
        "Analizar las variables, los datos faltantes y la distribución "
        "de la etiqueta antes de entrenar los modelos.",
        "src/09_analisis_exploratorio.py",
    ),
    (
        "4/5 - Entrenar y evaluar clustering",
        "Probar K-Means y Ward con diferentes cantidades de grupos. "
        "Comparar métricas y generar gráficos de los grupos encontrados.",
        "src/06_experimentos_clustering.py",
    ),
    (
        "5/5 - Entrenar clasificación y explicar",
        "Comparar modelos de clasificación usando GroupKFold. "
        "Evaluar el mejor modelo y explicar sus predicciones con SHAP.",
        "src/07_modelar_clasificacion.py",
    ),
]


def _validate_inputs():
    """Comprueba que estén disponibles los archivos originales."""

    missing = [
        path for path in REQUIRED_INPUTS
        if not path.is_file()
    ]

    if missing:
        expected = "\n".join(f"  - {path}" for path in missing)

        raise FileNotFoundError(
            "No se puede iniciar el pipeline. Faltan archivos:\n"
            + expected
        )

    print(
        f"Entradas verificadas: {len(REQUIRED_INPUTS)} archivos.",
        flush=True,
    )


def _validate_dataset():
    """Verifica que la base integrada cumpla los requisitos mínimos."""

    # Comprueba que la integración haya generado sus archivos
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"No se generó el dataset: {DATASET_PATH}"
        )

    if not INTEGRATION_REPORT.is_file():
        raise FileNotFoundError(
            f"No se generó el reporte: {INTEGRATION_REPORT}"
        )

    data = pd.read_parquet(DATASET_PATH)

    # Identificadores que permiten reconocer a cada persona
    person_keys = [
        "CONGLOME_25",
        "VIVIENDA_25",
        "HOGAR_25",
        "CODPERSO_25",
    ]

    required_columns = person_keys + [
        "edad",
        "UBIGEO_25",
        "exclusion_digital",
        "escale_numero_colegios",
    ]

    # Comprueba que existan las columnas necesarias
    missing_columns = sorted(
        set(required_columns) - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Faltan columnas requeridas: "
            f"{missing_columns}"
        )

    if data.empty:
        raise ValueError("El dataset está vacío.")

    # Verifica que cada estudiante tenga sus identificadores
    if data[person_keys].isna().any().any():
        raise ValueError(
            "Hay identificadores personales ausentes."
        )

    # Evita que una misma persona aparezca más de una vez
    duplicates = int(data.duplicated(person_keys).sum())

    if duplicates:
        raise ValueError(
            f"Se encontraron {duplicates} estudiantes duplicados."
        )

    # La investigación se centra en estudiantes de 6 a 17 años
    if not data["edad"].between(6, 17).all():
        raise ValueError(
            "Hay estudiantes fuera del rango de edad establecido."
        )

    # Comprueba la cobertura de los datos de ESCALE
    unmatched_escale = int(
        data["escale_numero_colegios"].isna().sum()
    )

    labeled = int(
        data["exclusion_digital"].notna().sum()
    )

    if unmatched_escale:
        raise ValueError(
            f"{unmatched_escale} estudiantes no tienen "
            "indicadores ESCALE asociados."
        )

    if labeled == 0:
        raise ValueError(
            "No hay observaciones con etiqueta disponible."
        )

    report = pd.read_csv(INTEGRATION_REPORT)

    # Muestra un resumen de la base validada
    print(
        f"Base validada: {len(data):,} estudiantes; "
        f"{data.shape[1]} columnas; "
        f"{labeled:,} etiquetas; "
        f"{data['CONGLOME_25'].nunique():,} conglomerados.",
        flush=True,
    )

    print(
        "Cobertura ESCALE: completa. "
        f"Etapas de integración registradas: {len(report)}.",
        flush=True,
    )


def _validate_outputs():
    """Comprueba que las etapas hayan generado sus resultados."""

    # Archivos que deben existir al terminar el pipeline
    required_outputs = [
        PROJECT_ROOT / "data" / "processed" / "diccionario_variables_2025.csv",
        PROJECT_ROOT / "informes" / "resultados_exploracion" / "resumen_variables.csv",
        CLUSTER_DIR / "metricas_clustering.csv",
        CLUSTER_DIR / "perfiles_clusters.csv",
        CLASSIFICATION_DIR / "comparacion_modelos_groupkfold.csv",
        CLASSIFICATION_DIR / "comparacion_modelos_prueba.csv",
    ]

    missing = [
        path for path in required_outputs
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Faltan archivos de resultados:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )

    # Identifica el modelo seleccionado durante la validación cruzada
    test_metrics = pd.read_csv(
        CLASSIFICATION_DIR / "comparacion_modelos_prueba.csv"
    )

    selected = test_metrics.loc[
        test_metrics["seleccionado_por_cv"]
    ]

    if len(selected) != 1:
        raise ValueError(
            "Debe existir exactamente un modelo seleccionado."
        )

    model_name = str(selected.iloc[0]["modelo"])
    model_dir = CLASSIFICATION_DIR / model_name

    # Gráficos esperados para evaluar e interpretar el modelo ganador
    expected_figures = [
        "curva_roc.png",
        "curva_precision_recall.png",
        "curva_calibracion.png",
        "matriz_confusion.png",
        "shap_beeswarm.png",
        "shap_dependencia_lengua_internet.png",
        "shap_dependencia_internet_lengua.png",
        "shap_waterfall_alto_riesgo.png",
        "shap_waterfall_bajo_riesgo.png",
    ]

    missing_figures = [
        model_dir / name
        for name in expected_figures
        if not (model_dir / name).is_file()
    ]

    if missing_figures:
        raise FileNotFoundError(
            "Faltan gráficos del modelo ganador:\n"
            + "\n".join(f"  - {path}" for path in missing_figures)
        )

    return model_name


def main():
    # Permite mostrar correctamente los caracteres en la terminal
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 84)
    print("PIPELINE COMPLETO - BRECHA DIGITAL ENAHO-ESCALE 2025")
    print("=" * 84)

    print(
        "Orden: verificar fuentes -> integrar -> validar -> explorar "
        "-> clusterizar -> clasificar y explicar."
    )

    print("Semilla de reproducibilidad: 42.\n")

    # Primera comprobación: verificar los archivos de entrada
    _validate_inputs()

    total_stages = len(STAGES)

    # Ejecuta cada etapa respetando el orden definido
    for index, (title, description, script_relative_path) in enumerate(
        STAGES,
        start=1,
    ):
        print(f"\n{'=' * 84}", flush=True)
        print(f"PASO {index}/{total_stages}: {title}", flush=True)
        print(f"Qué hace: {description}", flush=True)

        # La segunda etapa valida la base creada por la primera
        if script_relative_path is None:
            _validate_dataset()

        else:
            script_path = PROJECT_ROOT / script_relative_path

            if not script_path.is_file():
                raise FileNotFoundError(
                    f"No se encontró el script: {script_path}"
                )

            print(
                f"Ejecutando: {script_relative_path}",
                flush=True,
            )

            # Ejecuta el script y detiene el pipeline si falla
            subprocess.run(
                [sys.executable, str(script_path)],
                cwd=PROJECT_ROOT,
                check=True,
                env={
                    **os.environ,
                    "PYTHONIOENCODING": "utf-8",
                },
            )

        print(
            f"PASO {index}/{total_stages} COMPLETADO.",
            flush=True,
        )

    # Verifica los resultados y recupera el nombre del mejor modelo
    selected_model = _validate_outputs()

    print(f"\n{'=' * 84}")
    print("PIPELINE FINALIZADO CORRECTAMENTE")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Clustering: {CLUSTER_DIR}")
    print(f"Comparación de modelos: {CLASSIFICATION_DIR}")
    print(
        f"SHAP del modelo ganador ({selected_model}): "
        f"{CLASSIFICATION_DIR / selected_model}"
    )
    print(
        "Revisa también las carpetas informes/resultados_exploracion "
        "y data/processed."
    )


if __name__ == "__main__":
    main()