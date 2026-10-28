from pathlib import Path

import numpy as np
import pandas as pd
from dbfread import DBF


BASE_DIR = Path(__file__).resolve().parent.parent
ENAHO_DIR = BASE_DIR / "data" / "raw" / "enaho"
MINEDU_DIR = BASE_DIR / "data" / "raw" / "minedu"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "dataset_enaho_escale_2025.parquet"
REPORT_PATH = OUTPUT_DIR / "reporte_integracion_2025.csv"
DICTIONARY_PATH = OUTPUT_DIR / "diccionario_variables_2025.csv"

HOUSEHOLD_KEYS = ["CONGLOME_25", "VIVIENDA_25", "HOGAR_25"]
PERSON_KEYS = HOUSEHOLD_KEYS + ["CODPERSO_25"]
MISSING_VALUES = ["", " ", "  ", "   ", "    ", "     "]
RANDOM_STATE = 42


def _read_enaho(path, requested_columns):
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo requerido: {path}")

    raw_columns = pd.read_csv(path, encoding="latin1", nrows=0).columns
    normalized_to_raw = {
        str(column).replace("\ufeff", "").strip().upper(): column
        for column in raw_columns
    }
    selected = [
        normalized_to_raw[name]
        for name in requested_columns
        if name in normalized_to_raw
    ]
    missing = sorted(set(requested_columns) - set(normalized_to_raw))
    if missing:
        raise ValueError(f"{path.name}: faltan columnas requeridas: {missing}")

    frame = pd.read_csv(
        path,
        encoding="latin1",
        dtype=str,
        usecols=selected,
        na_values=MISSING_VALUES,
        low_memory=False,
    )
    frame.columns = [
        str(column).replace("\ufeff", "").strip().upper()
        for column in frame.columns
    ]
    for column in frame.columns:
        frame[column] = frame[column].str.strip().replace("", pd.NA)
    return frame


def _number(series):
    return pd.to_numeric(series, errors="coerce")


def _keep_valid_keys(frame, keys, source, report):
    before = len(frame)
    frame = frame.dropna(subset=keys).copy()
    duplicates = int(frame.duplicated(keys).sum())
    if duplicates:
        raise ValueError(
            f"{source}: {duplicates} llaves duplicadas en registros 2025; "
            "se detiene la integración para evitar multiplicar filas."
        )
    report.append(
        {
            "paso": source,
            "filas_entrada": before,
            "filas_salida": len(frame),
            "llaves_sin_dato_eliminadas": before - len(frame),
            "filas_sin_coincidencia": 0,
        }
    )
    return frame


def _merge_one_to_one(left, right, keys, name, report, validate="one_to_one"):
    before = len(left)
    indicator = f"_merge_{name}"
    merged = left.merge(
        right,
        on=keys,
        how="left",
        validate=validate,
        indicator=indicator,
    )
    unmatched = int(merged[indicator].eq("left_only").sum())
    report.append(
        {
            "paso": f"union_{name}",
            "filas_entrada": before,
            "filas_salida": len(merged),
            "llaves_sin_dato_eliminadas": 0,
            "filas_sin_coincidencia": unmatched,
        }
    )
    return merged.drop(columns=indicator)


def _load_escale(report):
    padron_path = MINEDU_DIR / "Padron.dbf"
    if not padron_path.exists():
        raise FileNotFoundError(f"No se encontró el padrón ESCALE: {padron_path}")

    padron = pd.DataFrame(
        iter(DBF(str(padron_path), encoding="latin1", load=True))
    )
    required = {
        "CODGEO",
        "COD_MOD",
        "DAREACENSO",
        "GESTION",
        "TALUM",
        "TDOC",
        "NLAT_IE",
        "NLONG_IE",
    }
    missing = sorted(required - set(padron.columns))
    if missing:
        raise ValueError(f"Padron.dbf: faltan campos requeridos: {missing}")

    padron["CODGEO"] = padron["CODGEO"].astype("string").str.strip()
    padron = padron[padron["CODGEO"].str.fullmatch(r"\d{6}", na=False)].copy()
    for column in ["TALUM", "TDOC", "NLAT_IE", "NLONG_IE"]:
        padron[column] = pd.to_numeric(padron[column], errors="coerce")
    padron["AREA"] = padron["DAREACENSO"].astype("string").str.strip().str.upper()
    padron["ES_PUBLICA"] = padron["GESTION"].astype("string").str.strip().isin(
        ["1", "2"]
    )

    def summarize(group):
        valid_ratio = group["TALUM"].ge(0) & group["TDOC"].gt(0)
        ratio = (
            group.loc[valid_ratio, "TALUM"].sum()
            / group.loc[valid_ratio, "TDOC"].sum()
            if valid_ratio.any()
            else np.nan
        )
        school_rows = group.drop_duplicates(["COD_MOD"])
        local_rows = group.drop_duplicates(["CODLOCAL"]) if "CODLOCAL" in group else group
        return pd.Series(
            {
                "ratio_alumnos_docente": ratio,
                "proporcion_colegios_publicos": school_rows["ES_PUBLICA"].mean(),
                "latitud_media": local_rows["NLAT_IE"].mean(),
                "longitud_media": local_rows["NLONG_IE"].mean(),
                "numero_colegios": school_rows["COD_MOD"].nunique(),
            }
        )

    by_district_area = (
        padron.groupby(["CODGEO", "AREA"], dropna=False)
        .apply(summarize, include_groups=False)
        .reset_index()
    )
    district = (
        padron.groupby("CODGEO", dropna=False)
        .apply(summarize, include_groups=False)
        .reset_index()
        .rename(
            columns={
                "ratio_alumnos_docente": "escale_ratio_alumnos_docente",
                "proporcion_colegios_publicos": "escale_proporcion_publica",
                "latitud_media": "escale_latitud_media",
                "longitud_media": "escale_longitud_media",
                "numero_colegios": "escale_numero_colegios",
            }
        )
    )

    for area, suffix in [("RURAL", "rural"), ("URBANA", "urbana")]:
        area_features = by_district_area[by_district_area["AREA"].eq(area)][
            [
                "CODGEO",
                "ratio_alumnos_docente",
                "proporcion_colegios_publicos",
            ]
        ].rename(
            columns={
                "ratio_alumnos_docente": f"escale_ratio_{suffix}",
                "proporcion_colegios_publicos": f"escale_proporcion_publica_{suffix}",
            }
        )
        district = district.merge(
            area_features, on="CODGEO", how="left", validate="one_to_one"
        )

    report.append(
        {
            "paso": "escale_agregado_distrito",
            "filas_entrada": len(padron),
            "filas_salida": len(district),
            "llaves_sin_dato_eliminadas": 0,
            "filas_sin_coincidencia": 0,
        }
    )
    return district.rename(columns={"CODGEO": "UBIGEO_25"})


def _dictionary():
    rows = [
        ("CONGLOME_25", "Conglomerado ENAHO 2025; variable de agrupación para separar entrenamiento y prueba."),
        ("VIVIENDA_25", "Código de vivienda ENAHO 2025."),
        ("HOGAR_25", "Código de hogar ENAHO 2025."),
        ("CODPERSO_25", "Código de persona ENAHO 2025."),
        ("UBIGEO_25", "Código geográfico distrital de seis dígitos."),
        ("edad", "Edad en años cumplidos en 2025; la cohorte se restringe a 6–17 años."),
        ("sexo", "Código de sexo reportado en ENAHO, P207."),
        ("lengua_materna", "Código de lengua materna reportado en ENAHO, P300A."),
        ("matriculado", "P306: 1=Sí, 2=No."),
        ("asistio", "P307: 1=Sí, 2=No."),
        ("internet_uso", "P314A: 1=Sí, 2=No; uso individual declarado."),
        ("exclusion_digital", "Etiqueta binaria: 1 si asistió (P307=1) y declaró no usar Internet (P314A=2). No mide competencias digitales."),
        ("electricidad_hogar", "P1121 del módulo 100; conservar códigos originales."),
        ("internet_hogar", "P1144 del módulo 100; conservar códigos originales."),
        ("dominio", "Dominio geográfico ENAHO, módulo 100."),
        ("estrato", "Estrato ENAHO, módulo 100; conservar código original."),
        ("pobreza", "Categoría POBREZA_25 de Sumaria; predictor socioeconómico, no es la etiqueta del modelo."),
        ("gasto_hogar", "GASHOG2D_25 de Sumaria."),
        ("ingreso_hogar", "INGHOG2D_25 de Sumaria."),
        ("factor_expansion", "FACTOR07_25 de Sumaria; ponderador muestral."),
        ("miembros_hogar", "Personas distintas con llave de hogar válida en el módulo 200 para 2025."),
        ("gasto_per_capita", "GASHOG2D_25 dividido entre miembros_hogar."),
        ("ingreso_per_capita", "INGHOG2D_25 dividido entre miembros_hogar."),
        ("escale_ratio_alumnos_docente", "Ratio distrital: total de alumnos / docentes válidos de ESCALE."),
        ("escale_proporcion_publica", "Proporción de instituciones del distrito clasificadas como gestión pública (GESTION 1 o 2)."),
        ("escale_latitud_media", "Latitud media distrital entre locales con coordenadas."),
        ("escale_longitud_media", "Longitud media distrital entre locales con coordenadas."),
        ("escale_numero_colegios", "Número de COD_MOD distintos en el padrón distrital."),
        ("escale_ratio_rural", "Ratio alumnos/docente agregado para registros ESCALE de área rural."),
        ("escale_ratio_urbana", "Ratio alumnos/docente agregado para registros ESCALE de área urbana."),
        ("escale_proporcion_publica_rural", "Proporción pública ESCALE en área rural."),
        ("escale_proporcion_publica_urbana", "Proporción pública ESCALE en área urbana."),
    ]
    return pd.DataFrame(rows, columns=["variable", "descripcion"])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = []

    print("Integracion 1/7: leer educacion y filtrar edades de 6 a 17.", flush=True)
    edu = _read_enaho(
        ENAHO_DIR / "enaho01a-2021-2025-300-panel.csv",
        PERSON_KEYS
        + [
            "UBIGEO_25",
            "P208A_25",
            "P300A_25",
            "P306_25",
            "P307_25",
            "P308A_25",
            "P308B_25",
            "P314A_25",
        ],
    )
    age = _number(edu["P208A_25"])
    edu = edu[age.between(6, 17)].copy()
    edu = _keep_valid_keys(edu, PERSON_KEYS, "educacion_edad_6_17", report)
    print(f"  Estudiantes con llaves validas: {len(edu):,}", flush=True)
    edu["UBIGEO_25"] = edu["UBIGEO_25"].astype("string").str.strip()
    edu = edu.rename(
        columns={
            "P208A_25": "edad",
            "P300A_25": "lengua_materna",
            "P306_25": "matriculado",
            "P307_25": "asistio",
            "P308A_25": "nivel_educativo",
            "P308B_25": "grado_educativo",
            "P314A_25": "internet_uso",
        }
    )

    print("Integracion 2/7: leer personas, sexo y calcular miembros por hogar.", flush=True)
    persons = _read_enaho(
        ENAHO_DIR / "enaho01-2021-2025-200-panel.csv",
        PERSON_KEYS + ["P207_25"],
    )
    persons = _keep_valid_keys(persons, PERSON_KEYS, "personas_2025", report)
    household_size = (
        persons.groupby(HOUSEHOLD_KEYS)["CODPERSO_25"]
        .nunique()
        .rename("miembros_hogar")
        .reset_index()
    )
    persons = persons.rename(columns={"P207_25": "sexo"})
    edu = _merge_one_to_one(edu, persons, PERSON_KEYS, "personas", report)
    edu = edu.merge(
        household_size, on=HOUSEHOLD_KEYS, how="left", validate="many_to_one"
    )
    print(f"  Union persona completada: {len(edu):,} filas.", flush=True)

    print("Integracion 3/7: leer modulo del hogar y unir servicios y geografia.", flush=True)
    households = _read_enaho(
        ENAHO_DIR / "enaho01-2021-2025-100-panel.csv",
        HOUSEHOLD_KEYS
        + [
            "UBIGEO_25",
            "DOMINIO_25",
            "ESTRATO_25",
            "P1121_25",
            "P1144_25",
        ],
    )
    households = households.drop(columns=["UBIGEO_25"])
    households = _keep_valid_keys(households, HOUSEHOLD_KEYS, "hogares_2025", report)
    households = households.rename(
        columns={
            "DOMINIO_25": "dominio",
            "ESTRATO_25": "estrato",
            "P1121_25": "electricidad_hogar",
            "P1144_25": "internet_hogar",
        }
    )
    edu = _merge_one_to_one(
        edu, households, HOUSEHOLD_KEYS, "hogares", report, validate="many_to_one"
    )
    print(f"  Union hogar completada: {len(edu):,} filas.", flush=True)

    print("Integracion 4/7: leer Sumaria y unir pobreza, ingreso, gasto y ponderador.", flush=True)
    summary = _read_enaho(
        ENAHO_DIR / "sumaria-2021-2025-panel.csv",
        HOUSEHOLD_KEYS
        + ["POBREZA_25", "GASHOG2D_25", "INGHOG2D_25", "FACTOR07_25"],
    )
    summary = _keep_valid_keys(summary, HOUSEHOLD_KEYS, "sumaria_2025", report)
    summary = summary.rename(
        columns={
            "POBREZA_25": "pobreza",
            "GASHOG2D_25": "gasto_hogar",
            "INGHOG2D_25": "ingreso_hogar",
            "FACTOR07_25": "factor_expansion",
        }
    )
    edu = _merge_one_to_one(
        edu, summary, HOUSEHOLD_KEYS, "sumaria", report, validate="many_to_one"
    )
    print(f"  Union Sumaria completada: {len(edu):,} filas.", flush=True)

    print("Integracion 5/7: convertir variables numericas y calcular valores per capita.", flush=True)
    for column in [
        "edad",
        "nivel_educativo",
        "grado_educativo",
        "miembros_hogar",
        "gasto_hogar",
        "ingreso_hogar",
        "factor_expansion",
    ]:
        edu[column] = _number(edu[column])
    edu["gasto_per_capita"] = edu["gasto_hogar"] / edu["miembros_hogar"]
    edu["ingreso_per_capita"] = edu["ingreso_hogar"] / edu["miembros_hogar"]

    print("Integracion 6/7: agregar ESCALE por distrito y unir mediante UBIGEO.", flush=True)
    escale = _load_escale(report)
    edu = edu.merge(
        escale, on="UBIGEO_25", how="left", validate="many_to_one", indicator=True
    )
    report.append(
        {
            "paso": "union_escale_distrito",
            "filas_entrada": len(edu),
            "filas_salida": len(edu),
            "llaves_sin_dato_eliminadas": 0,
            "filas_sin_coincidencia": int(edu["_merge"].eq("left_only").sum()),
        }
    )
    edu = edu.drop(columns="_merge")
    print(
        f"  Union ESCALE completada: {int(edu['escale_numero_colegios'].notna().sum()):,} "
        "estudiantes con indicador distrital.",
        flush=True,
    )

    print("Integracion 7/7: crear etiqueta y guardar base, reporte y diccionario.", flush=True)
    edu["exclusion_digital"] = pd.Series(pd.NA, index=edu.index, dtype="Int64")
    eligible = edu["asistio"].eq("1") & edu["internet_uso"].isin(["1", "2"])
    edu.loc[eligible, "exclusion_digital"] = (
        edu.loc[eligible, "internet_uso"].eq("2").astype("int64")
    )

    edu.to_parquet(OUTPUT_PATH, index=False, compression="snappy")
    pd.DataFrame(report).to_csv(REPORT_PATH, index=False, encoding="utf-8-sig")
    _dictionary().to_csv(DICTIONARY_PATH, index=False, encoding="utf-8-sig")

    print(f"Dataset: {OUTPUT_PATH}")
    print(f"Filas: {len(edu):,} | columnas: {len(edu.columns)}")
    print(f"Etiqueta disponible en: {int(edu['exclusion_digital'].notna().sum()):,} filas")
    print(f"Filas sin cruce ESCALE: {int(edu['escale_numero_colegios'].isna().sum()):,}")
    print(f"Reporte de uniones: {REPORT_PATH}")
    print(f"Diccionario: {DICTIONARY_PATH}")


if __name__ == "__main__":
    main()
