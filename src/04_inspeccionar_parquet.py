from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed" / "enaho"

# Palabras que nos interesa localizar en los nombres de columnas.
PALABRAS_CLAVE = [
    "ANIO", "AÑO", "YEAR", "MES",
    "CONGLOME", "VIVIENDA", "HOGAR",
    "CODPERSO", "UBIGEO", "DOMINIO",
    "ESTRATO", "PESO", "FACTOR",
    "EDAD", "P208", "P207", "P203",
    "P300", "P301", "P302", "P303",
]

def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"No existe el directorio: {DATA_DIR}"
        )

    carpetas = sorted(
        p for p in DATA_DIR.iterdir() if p.is_dir()
    )

    if not carpetas:
        print("No se encontraron carpetas de Parquet.")
        return

    for carpeta in carpetas:
        archivos = sorted(carpeta.glob("*.parquet"))

        if not archivos:
            print(f"\n{carpeta.name}: sin archivos Parquet")
            continue

        # Leer únicamente el primer bloque.
        archivo = archivos[0]

        print("\n" + "=" * 75)
        print(f"ARCHIVO: {carpeta.name}")
        print(f"BLOQUES DISPONIBLES: {len(archivos)}")
        print(f"MUESTRA: {archivo.name}")
        print("=" * 75)

        muestra = pd.read_parquet(archivo)

        columnas = list(muestra.columns)

        print(f"Columnas: {len(columnas)}")
        print("\nColumnas relevantes:")

        relevantes = [
            c for c in columnas
            if any(p in c.upper() for p in PALABRAS_CLAVE)
        ]

        for columna in relevantes:
            print(f"  - {columna}")

        print("\nPrimeras 2 filas de la muestra:")
        print(muestra.head(2).to_string(index=False))

        del muestra

    print("\nInspección finalizada.")

if __name__ == "__main__":
    main()
