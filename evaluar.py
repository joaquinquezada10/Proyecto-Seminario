# Evalua el modelo entrenado con las imagenes de prueba y guarda sus resultados

from pathlib import Path
from ultralytics import YOLO

def main():
    carpeta = Path(__file__).resolve().parent

    ruta_modelo = (
        carpeta / "modelos" / "best.pt"
    )

    modelo = YOLO(str(ruta_modelo))

    modelo.val(
        data=str(carpeta / "dataset" / "data.yaml"),
        split="test",
        imgsz=640,
        batch=4,
        device="cpu",
        workers=0,
        plots=True,
        project=str(carpeta / "runs"),
        name="evaluacion_test"
    )

if __name__ == "__main__":
    main()