from pathlib import Path
import pandas as pd

# Directorio principal del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

ENAHO_DIR = BASE_DIR / "data" / "raw" / "enaho"
MINEDU_DIR = BASE_DIR / "data" / "raw" / "minedu"

# Columnas que pueden contener códigos con ceros iniciales
COLUMNAS_CODIGO = {
    "AÑO": str,
    "ANIO": str,
    "MES": str,
    "CONGLOME": str,
    "VIVIENDA": str,
    "HOGAR": str,
    "CODPERSO": str,
    "UBIGEO": str,
    "CODGEO": str,
    "COD_MOD": str,
    "CODLOCAL": str,
}

def inspeccionar_csv(ruta):
    print("\n" + "=" * 80)
    print(f"ARCHIVO: {ruta.name}")
    print("=" * 80)

    try:
        # Leer solamente una muestra para inspeccionar el archivo.
        muestra = pd.read_csv(
            ruta,
            encoding="latin1",
            dtype=COLUMNAS_CODIGO,
            na_values=[" ", "     ", ""],
            low_memory=False,
            nrows=5,
        )

        # Normalizar los nombres de columnas.
        muestra.columns = muestra.columns.str.upper().str.strip()

        print("\nColumnas:")
        print(muestra.columns.tolist())

        print("\nTipos de datos:")
        print(muestra.dtypes.to_string())

        print("\nPrimeras filas:")
        print(muestra.head().to_string(index=False))

    except Exception as e:
        print(f"ERROR AL LEER {ruta.name}: {e}")


def main():
    print("INSPECCIÓN DE ARCHIVOS DEL PROYECTO")
    print(f"Directorio: {BASE_DIR}")

    print("\n\n========== ARCHIVOS ENAHO ==========")

    if ENAHO_DIR.exists():
        archivos_csv = sorted(ENAHO_DIR.glob("*.csv"))

        if not archivos_csv:
            print("No se encontraron archivos CSV en ENAHO.")

        for archivo in archivos_csv:
            inspeccionar_csv(archivo)
    else:
        print(f"No existe el directorio: {ENAHO_DIR}")

    print("\n\n========== ARCHIVOS ESCALE ==========")

    if MINEDU_DIR.exists():
        for archivo in sorted(MINEDU_DIR.glob("*.dbf")):
            print(f"\nArchivo encontrado: {archivo.name}")
            print(f"Tamaño: {archivo.stat().st_size / (1024 ** 2):.2f} MB")
    else:
        print(f"No existe el directorio: {MINEDU_DIR}")


if __name__ == "__main__":
    main()
