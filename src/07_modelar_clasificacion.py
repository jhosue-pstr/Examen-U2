from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "dataset_enaho_escale_2025.parquet"
OUTPUT_DIR = BASE_DIR / "informes" / "resultados_clasificacion"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5

NUMERIC_FEATURES = [
    "edad",
    "nivel_educativo",
    "grado_educativo",
    "electricidad_hogar",
    "internet_hogar",
    "gasto_per_capita",
    "ingreso_per_capita",
    "miembros_hogar",
    "escale_ratio_alumnos_docente",
    "escale_proporcion_publica",
    "escale_latitud_media",
    "escale_longitud_media",
    "escale_numero_colegios",
    "escale_ratio_rural",
    "escale_ratio_urbana",
    "escale_proporcion_publica_rural",
    "escale_proporcion_publica_urbana",
]
CATEGORICAL_FEATURES = [
    "sexo",
    "lengua_materna",
    "matriculado",
    "dominio",
    "estrato",
    "pobreza",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _weighted_training_weights(y, expansion_weights):
    class_counts = y.value_counts()
    if len(class_counts) != 2 or (class_counts == 0).any():
        raise ValueError("Se necesitan las dos clases para aplicar pesos balanceados.")
    class_weights = {
        label: len(y) / (len(class_counts) * count)
        for label, count in class_counts.items()
    }
    balanced = expansion_weights.to_numpy(dtype=float, copy=True)
    if not np.isfinite(balanced).all() or (balanced <= 0).any():
        raise ValueError("FACTOR07 debe ser positivo y finito en las filas etiquetadas.")
    balanced *= y.map(class_weights).to_numpy(dtype=float)
    return balanced / balanced.mean()


def _make_pipeline(estimator):
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
    return Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])


def _models():
    return {
        "Regresion_Logistica_baseline": LogisticRegression(
            max_iter=1_000, random_state=RANDOM_STATE
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Extra_Trees": ExtraTreesClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=4,
            random_state=RANDOM_STATE,
        ),
    }


def _metrics(y_true, probability, weights):
    prediction = (probability >= 0.5).astype(int)
    return {
        "roc_auc_ponderado": roc_auc_score(
            y_true, probability, sample_weight=weights
        ),
        "pr_auc_ponderado": average_precision_score(
            y_true, probability, sample_weight=weights
        ),
        "f1_macro_ponderado": f1_score(
            y_true, prediction, average="macro", sample_weight=weights, zero_division=0
        ),
        "roc_auc_sin_ponderar": roc_auc_score(y_true, probability),
        "pr_auc_sin_ponderar": average_precision_score(y_true, probability),
        "f1_macro_sin_ponderar": f1_score(
            y_true, prediction, average="macro", zero_division=0
        ),
    }


def _save_test_plots(y_test, probability, weights, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction = (probability >= 0.5).astype(int)
    weighted_matrix = confusion_matrix(
        y_test, prediction, labels=[0, 1], sample_weight=weights
    )
    pd.DataFrame(
        weighted_matrix,
        index=["real_0", "real_1"],
        columns=["predicho_0", "predicho_1"],
    ).to_csv(output_dir / "matriz_confusion_ponderada.csv", encoding="utf-8-sig")

    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(weighted_matrix, display_labels=["No excluido", "Excluido"]).plot(
        ax=axis, values_format=".0f", colorbar=False
    )
    axis.set_title("Matriz de confusión ponderada por FACTOR07")
    figure.tight_layout()
    figure.savefig(output_dir / "matriz_confusion.png", dpi=160)
    plt.close(figure)

    fpr, tpr, _ = roc_curve(y_test, probability, sample_weight=weights)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test, probability, sample_weight=weights):.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos")
    axis.set_title("Curva ROC ponderada")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "curva_roc.png", dpi=160)
    plt.close(figure)

    precision, recall, _ = precision_recall_curve(
        y_test, probability, sample_weight=weights
    )
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(recall, precision)
    axis.set(xlabel="Recall", ylabel="Precision")
    axis.set_title(
        f"Curva Precision-Recall ponderada "
        f"(AP={average_precision_score(y_test, probability, sample_weight=weights):.3f})"
    )
    figure.tight_layout()
    figure.savefig(output_dir / "curva_precision_recall.png", dpi=160)
    plt.close(figure)

    calibration = pd.DataFrame(
        {"probabilidad": probability, "observado": y_test, "peso": weights}
    )
    calibration["bin"] = pd.qcut(
        calibration["probabilidad"], q=10, duplicates="drop"
    )
    calibration = (
        calibration.groupby("bin", observed=True)
        .apply(
            lambda group: pd.Series(
                {
                    "probabilidad_media": np.average(
                        group["probabilidad"], weights=group["peso"]
                    ),
                    "fraccion_positiva": np.average(
                        group["observado"], weights=group["peso"]
                    ),
                    "n": len(group),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    calibration.to_csv(
        output_dir / "calibracion_ponderada.csv",
        index=False,
        encoding="utf-8-sig",
    )
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(
        calibration["probabilidad_media"],
        calibration["fraccion_positiva"],
        marker="o",
        label="Modelo",
    )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    axis.set(xlabel="Probabilidad media predicha", ylabel="Proporción positiva observada")
    axis.set_title("Calibración ponderada por FACTOR07")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "curva_calibracion.png", dpi=160)
    plt.close(figure)


def _save_shap(best_pipeline, x_train, x_test, y_test, probabilities, output_dir):
    preprocessor = best_pipeline.named_steps["preprocessor"]
    classifier = best_pipeline.named_steps["classifier"]
    transformed_train = preprocessor.transform(x_train)
    transformed_test = preprocessor.transform(x_test)
    feature_names = preprocessor.get_feature_names_out()
    rng = np.random.default_rng(RANDOM_STATE)
    background_indices = rng.choice(
        len(transformed_train), min(80, len(transformed_train)), replace=False
    )
    explain_count = min(300, len(transformed_test))
    explain_indices = rng.choice(len(transformed_test), explain_count, replace=False)

    explainer = shap.Explainer(
        classifier,
        transformed_train[background_indices],
        feature_names=feature_names,
    )
    raw_explanation = explainer(transformed_test[explain_indices])
    values = np.asarray(raw_explanation.values)
    base_values = np.asarray(raw_explanation.base_values)
    if values.ndim == 3:
        values = values[:, :, 1]
        if base_values.ndim > 1:
            base_values = base_values[:, 1]
    explanation = shap.Explanation(
        values=values,
        base_values=base_values,
        data=transformed_test[explain_indices],
        feature_names=feature_names,
    )

    plt.figure()
    shap.plots.beeswarm(explanation, max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_beeswarm.png", dpi=180, bbox_inches="tight")
    plt.close()

    importance = np.abs(values).mean(axis=0)
    language_indices = [
        index
        for index, name in enumerate(feature_names)
        if str(name).startswith("lengua_materna_")
    ]
    internet_indices = [
        index
        for index, name in enumerate(feature_names)
        if str(name) == "internet_hogar"
    ]
    if not language_indices or not internet_indices:
        raise ValueError(
            "No se encontraron las variables transformadas de lengua materna "
            "e Internet del hogar para las gráficas SHAP."
        )
    language_index = max(language_indices, key=lambda index: importance[index])
    internet_index = internet_indices[0]
    for primary, interacting, output_name in [
        (
            language_index,
            internet_index,
            "shap_dependencia_lengua_internet.png",
        ),
        (
            internet_index,
            language_index,
            "shap_dependencia_internet_lengua.png",
        ),
    ]:
        plt.figure()
        shap.plots.scatter(
            explanation[:, primary],
            color=explanation[:, interacting],
            show=False,
        )
        plt.title(
            f"SHAP: {feature_names[primary]}; color={feature_names[interacting]}"
        )
        plt.tight_layout()
        plt.savefig(
            output_dir / output_name,
            dpi=180,
            bbox_inches="tight",
        )
        plt.close()

    sample_y = y_test.iloc[explain_indices].to_numpy()
    sample_probability = probabilities[explain_indices]
    sample_prediction = sample_probability >= 0.5
    for label, candidates, correct, order in [
        (
            "alto_riesgo",
            np.flatnonzero(sample_y == 1),
            np.flatnonzero((sample_y == 1) & sample_prediction),
            np.argmax,
        ),
        (
            "bajo_riesgo",
            np.flatnonzero(sample_y == 0),
            np.flatnonzero((sample_y == 0) & ~sample_prediction),
            np.argmin,
        ),
    ]:
        if len(candidates) == 0:
            raise ValueError(
                f"La muestra SHAP no contiene casos reales de la clase {label}."
            )
        eligible_cases = correct if len(correct) else candidates
        selected = eligible_cases[order(sample_probability[eligible_cases])]
        figure = plt.figure()
        shap.plots.waterfall(explanation[int(selected)], max_display=10, show=False)
        plt.title(
            f"{label}: prob={sample_probability[selected]:.3f}, "
            f"real={sample_y[selected]}, predicho={int(sample_probability[selected] >= 0.5)}"
        )
        plt.tight_layout()
        figure.savefig(
            output_dir / f"shap_waterfall_{label}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)

    pd.DataFrame(
        {"variable": feature_names, "shap_abs_medio": importance}
    ).sort_values("shap_abs_medio", ascending=False).to_csv(
        output_dir / "importancia_shap.csv",
        index=False,
        encoding="utf-8-sig",
    )


def evaluate_groupkfold(
    x_train,
    y_train,
    groups_train,
    weights_train,
    model_specs=None,
    output_dir=None,
):
    """Compara modelos con preprocesamiento ajustado dentro de cada fold agrupado."""
    model_specs = model_specs or _models()
    group_kfold = GroupKFold(n_splits=N_SPLITS)
    model_frames = {}

    for model_name, estimator in model_specs.items():
        fold_metrics = []
        for fold, (fit_indices, validation_indices) in enumerate(
            group_kfold.split(x_train, y_train, groups_train), start=1
        ):
            pipeline = _make_pipeline(estimator)
            fold_y = y_train.iloc[fit_indices]
            fold_weights = _weighted_training_weights(
                fold_y, weights_train.iloc[fit_indices]
            )
            pipeline.fit(
                x_train.iloc[fit_indices],
                fold_y,
                classifier__sample_weight=fold_weights,
            )
            validation_probability = pipeline.predict_proba(
                x_train.iloc[validation_indices]
            )[:, 1]
            metrics = _metrics(
                y_train.iloc[validation_indices].to_numpy(),
                validation_probability,
                weights_train.iloc[validation_indices].to_numpy(),
            )
            metrics["modelo"] = model_name
            metrics["fold"] = fold
            fold_metrics.append(metrics)

        model_frame = pd.DataFrame(fold_metrics)
        model_frames[model_name] = model_frame
        if output_dir is not None:
            model_output_dir = output_dir / model_name
            model_output_dir.mkdir(parents=True, exist_ok=True)
            model_frame.to_csv(
                model_output_dir / "resultados_groupkfold_por_fold.csv",
                index=False,
                encoding="utf-8-sig",
            )
        print(f"Validación agrupada completada: {model_name}", flush=True)

    cv_frame = pd.concat(model_frames.values(), ignore_index=True)
    cv_summary = cv_frame.groupby("modelo").agg(["mean", "std"]).reset_index()
    cv_summary.columns = [
        "_".join(str(part) for part in column if part).strip("_")
        if isinstance(column, tuple)
        else column
        for column in cv_summary.columns
    ]
    cv_summary = cv_summary.sort_values(
        "roc_auc_ponderado_mean", ascending=False
    ).reset_index(drop=True)
    return cv_frame, cv_summary


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No existe el dataset integrado: {DATA_PATH}. "
            "Ejecuta primero src/05_integrar_enaho_escale.py."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_parquet(DATA_PATH)
    required = set(FEATURES + ["exclusion_digital", "CONGLOME_25", "factor_expansion"])
    missing_columns = sorted(required - set(data.columns))
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas para el modelo: {missing_columns}")

    eligible = (
        data["exclusion_digital"].notna()
        & data["CONGLOME_25"].notna()
        & data["factor_expansion"].notna()
        & data["factor_expansion"].gt(0)
    )
    model_data = data.loc[eligible].copy()
    x = model_data[FEATURES]
    y = model_data["exclusion_digital"].astype(int)
    groups = model_data["CONGLOME_25"].astype(str)
    expansion_weights = model_data["factor_expansion"].astype(float)
    if y.nunique() != 2:
        raise ValueError("La etiqueta exclusion_digital no contiene las dos clases.")
    if groups.nunique() < N_SPLITS + 1:
        raise ValueError(
            f"Solo hay {groups.nunique()} conglomerados; se necesitan al menos "
            f"{N_SPLITS + 1} para la partición y la validación agrupadas."
        )

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_indices, test_indices = next(splitter.split(x, y, groups))
    x_train, x_test = x.iloc[train_indices], x.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    groups_train = groups.iloc[train_indices]
    weights_train = expansion_weights.iloc[train_indices]
    weights_test = expansion_weights.iloc[test_indices].to_numpy()

    overlap = set(groups.iloc[train_indices]) & set(groups.iloc[test_indices])
    if overlap:
        raise RuntimeError(f"Fuga de conglomerados detectada: {len(overlap)} grupos.")
    model_specs = _models()
    cv_frame, cv_summary = evaluate_groupkfold(
        x_train,
        y_train,
        groups_train,
        weights_train,
        model_specs=model_specs,
        output_dir=OUTPUT_DIR,
    )
    cv_frame.to_csv(
        OUTPUT_DIR / "resultados_groupkfold_por_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cv_summary.to_csv(
        OUTPUT_DIR / "comparacion_modelos_groupkfold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    best_name = str(cv_summary.iloc[0]["modelo"])
    test_rows = []
    fitted_models = {}
    for model_name, estimator in model_specs.items():
        pipeline = _make_pipeline(estimator)
        final_training_weights = _weighted_training_weights(y_train, weights_train)
        pipeline.fit(
            x_train,
            y_train,
            classifier__sample_weight=final_training_weights,
        )
        probability = pipeline.predict_proba(x_test)[:, 1]
        test_metrics = _metrics(y_test.to_numpy(), probability, weights_test)
        test_metrics.update(
            {
                "modelo": model_name,
                "seleccionado_por_cv": model_name == best_name,
                "n_entrenamiento": len(x_train),
                "n_prueba": len(x_test),
                "conglomerados_entrenamiento": groups_train.nunique(),
                "conglomerados_prueba": groups.iloc[test_indices].nunique(),
                "interseccion_conglomerados": len(overlap),
                "prevalencia_ponderada_prueba": np.average(
                    y_test, weights=weights_test
                ),
            }
        )
        test_rows.append(test_metrics)
        model_output_dir = OUTPUT_DIR / model_name
        pd.DataFrame([test_metrics]).to_csv(
            model_output_dir / "metricas_prueba.csv",
            index=False,
            encoding="utf-8-sig",
        )
        _save_test_plots(
            y_test.to_numpy(), probability, weights_test, model_output_dir
        )
        joblib.dump(pipeline, model_output_dir / "modelo.joblib")
        fitted_models[model_name] = (pipeline, probability)

    test_results = pd.DataFrame(test_rows)
    test_results.to_csv(
        OUTPUT_DIR / "comparacion_modelos_prueba.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best_pipeline, best_probability = fitted_models[best_name]
    best_output_dir = OUTPUT_DIR / best_name
    _save_shap(
        best_pipeline,
        x_train,
        x_test,
        y_test,
        best_probability,
        best_output_dir,
    )
    joblib.dump(best_pipeline, OUTPUT_DIR / "mejor_modelo.joblib")

    print(f"\nMejor modelo por ROC-AUC ponderado CV: {best_name}")
    print(test_results.to_string(index=False))
    print(f"Resultados: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
