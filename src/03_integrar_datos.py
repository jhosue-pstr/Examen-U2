from pathlib import Path
import gc
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# =====================================================
# CONFIGURACIÓN
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent
ENAHO_DIR = BASE_DIR / "data" / "raw" / "enaho"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "enaho"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Bloques pequeños para reducir el uso de RAM.
# Si todavía falta memoria, reducir a 100.
TAMANO_BLOQUE = 500

COLUMNAS_CODIGO = {
    "AÑO": str,
    "ANIO": str,
    "MES": str,
    "CONGLOME": str,
    "VIVIENDA": str,
    "HOGAR": str,
    "CODPERSO": str,
    "UBIGEO": str,
    "NUMPANH": str,
    "NUMPER": str,
}


# =====================================================
# PROCESAMIENTO POR BLOQUES
# =====================================================

def procesar_csv(ruta):
    print(f"\nProcesando: {ruta.name}", flush=True)

    # Cada archivo tendrá su propio directorio de salida.
    nombre_salida = ruta.stem
    directorio_salida = OUTPUT_DIR / nombre_salida
    directorio_salida.mkdir(parents=True, exist_ok=True)

    total_filas = 0
    total_bloques = 0
    columnas = None

    # Leer el CSV progresivamente, sin cargarlo entero.
    lector = pd.read_csv(
        ruta,
        encoding="utf-8-sig",
        sep=",",
        dtype=COLUMNAS_CODIGO,
        na_values=[" ", "     ", ""],
        chunksize=TAMANO_BLOQUE,
        low_memory=True,
    )

    for numero_bloque, bloque in enumerate(lector, start=1):
        # Normalizar nombres de columnas.
        bloque.columns = (
            bloque.columns.astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.upper()
            .str.strip()
        )

        if columnas is None:
            columnas = bloque.columns.tolist()

            print(
                f"Columnas detectadas: {len(columnas)}",
                flush=True
            )

        # Guardar cada bloque como un archivo Parquet independiente.
        salida_bloque = (
            directorio_salida
            / f"parte_{numero_bloque:06d}.parquet"
        )

        tabla = pa.Table.from_pandas(
            bloque,
            preserve_index=False
        )

        pq.write_table(
            tabla,
            salida_bloque,
            compression="snappy"
        )

        total_filas += len(bloque)
        total_bloques += 1

        print(
            f"  Bloque {numero_bloque}: "
            f"{len(bloque):,} filas | "
            f"Acumulado: {total_filas:,}",
            flush=True
        )

        del tabla, bloque
        gc.collect()

    # Guardar un resumen del archivo procesado.
    return {
        "archivo": ruta.name,
        "filas": total_filas,
        "columnas": len(columnas) if columnas else 0,
        "bloques": total_bloques,
        "columnas_disponibles": (
            ", ".join(columnas) if columnas else ""
        ),
        "salida": str(directorio_salida.relative_to(BASE_DIR)),
    }


# =====================================================
# PROCESO PRINCIPAL
# =====================================================

def main():
    if not ENAHO_DIR.exists():
        raise FileNotFoundError(
            f"No existe el directorio: {ENAHO_DIR}"
        )

    archivos = sorted(ENAHO_DIR.glob("*.csv"))

    if not archivos:
        raise FileNotFoundError(
            f"No hay archivos CSV en {ENAHO_DIR}"
        )

    print("=" * 65)
    print("PROCESAMIENTO ENAHO 2021-2025")
    print("=" * 65)
    print(f"Archivos encontrados: {len(archivos)}")
    print(f"Tamaño del bloque: {TAMANO_BLOQUE:,} filas")

    resultados = []

    for i, ruta in enumerate(archivos, start=1):
        print(f"\nArchivo {i}/{len(archivos)}", flush=True)

        try:
            resultado = procesar_csv(ruta)
            resultados.append(resultado)

        except Exception as error:
            print(
                f"ERROR en {ruta.name}: {error}",
                flush=True
            )

            resultados.append({
                "archivo": ruta.name,
                "filas": None,
                "columnas": None,
                "bloques": None,
                "columnas_disponibles": None,
                "salida": None,
            })

        gc.collect()

    # Guardar inventario final.
    reporte = pd.DataFrame(resultados)

    salida_reporte = (
        BASE_DIR / "data" / "processed" / "inventario_enaho.csv"
    )

    reporte.to_csv(
        salida_reporte,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 65)
    print("RESUMEN FINAL")
    print("=" * 65)

    print(
        reporte[["archivo", "filas", "columnas", "bloques"]]
        .to_string(index=False)
    )

    print(f"\nInventario: {salida_reporte}")
    print(f"Datos procesados: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
