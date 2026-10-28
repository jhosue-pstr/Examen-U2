from pathlib import Path
from dbfread import DBF

BASE_DIR = Path(__file__).resolve().parent.parent
MINEDU_DIR = BASE_DIR / "data" / "raw" / "minedu"


def inspeccionar_dbf(ruta):
    print("\n" + "=" * 80)
    print(f"ARCHIVO: {ruta.name}")
    print("=" * 80)

    try:
        tabla = DBF(
            str(ruta),
            encoding="latin1",
            load=False,
        )

        print("\nColumnas:")
        print(tabla.field_names)

        print("\nPrimeros cinco registros:")

        for i, registro in enumerate(tabla):
            print(dict(registro))

            if i >= 4:
                break

    except Exception as e:
        print(f"Error al leer {ruta.name}: {e}")


def main():
    for nombre in ["Padron.dbf", "Padlocal.dbf"]:
        ruta = MINEDU_DIR / nombre

        if ruta.exists():
            inspeccionar_dbf(ruta)
        else:
            print(f"No se encontró: {ruta}")


if __name__ == "__main__":
    main()
