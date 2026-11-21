# Evalua la precision del reconocimiento de patentes sobre imagenes de prueba conocidas.
#
# Uso:  .\.venv\Scripts\python.exe evaluar.py
#
# Compara la patente detectada por el pipeline completo (YOLO + OCR + correccion)
# contra la patente real de cada imagen y calcula precision, aciertos y errores.

from pathlib import Path
from ultralytics import YOLO

import main

# ---------------------------------------------------------------------------
# Verdad de campo (ground truth) para cada imagen de prueba.
# Formato: nombre_archivo -> patente_limpia_esperada (sin espacios ni guion)
# ---------------------------------------------------------------------------
GROUND_TRUTH = {
    "auto3.jpg":    "FDPZ97",   # FD PZ-97  Moderna
    "auto7.jpg":    "SDFT30",   # SD FT-30  Moderna
    "auto8.jpg":    "BBBB10",   # BB BB-10  Moderna
    "auto9.jpg":    "HSDW28",   # HS DW-28  Moderna
    "autoazul.png": "JJVF14",   # JJ VF-14  Moderna
}

CARPETA_PRUEBAS = Path(__file__).resolve().parent / "Registro de vehiculos"
RUTA_MODELO    = Path(__file__).resolve().parent / "modelos" / "best.pt"


def evaluar():
    if not RUTA_MODELO.is_file():
        print(f"Error: no se encontro el modelo en {RUTA_MODELO}")
        return

    modelo = YOLO(str(RUTA_MODELO))

    print("=" * 65)
    print("  EVALUACION DE PRECISION - RECONOCIMIENTO DE PATENTES")
    print("=" * 65)
    print(f"{'Archivo':<18} {'Esperada':<10} {'Detectada':<10} {'Resultado'}")
    print("-" * 65)

    aciertos = 0
    total    = len(GROUND_TRUTH)

    for nombre, esperada in GROUND_TRUTH.items():
        ruta = CARPETA_PRUEBAS / nombre

        if not ruta.is_file():
            print(f"  {nombre:<16} {esperada:<10} {'---':<10} ARCHIVO NO ENCONTRADO")
            total -= 1
            continue

        _, vehiculos = main.procesar_archivo_foto(ruta, modelo)

        if vehiculos:
            detectada = vehiculos[0]["patente_limpia"]
        else:
            detectada = "NINGUNA"

        ok = detectada == esperada
        if ok:
            aciertos += 1
            marca = "✓"
        else:
            marca = "✗"

        print(f"  {nombre:<16} {esperada:<10} {detectada:<10} {marca}")

    print("-" * 65)
    if total > 0:
        precision = aciertos / total * 100
        print(f"\n  Aciertos : {aciertos}/{total}")
        print(f"  Precision: {precision:.1f}%")
    else:
        print("  No habia imagenes de prueba disponibles.")
    print("=" * 65)


if __name__ == "__main__":
    evaluar()
