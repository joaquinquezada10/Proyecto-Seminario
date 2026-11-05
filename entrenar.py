from pathlib import Path

import yaml
from ultralytics import YOLO


def main():
    # Localizar las carpetas del proyecto
    proyecto = Path(__file__).resolve().parent
    dataset = proyecto / "dataset"

    # Leer la configuracion original
    archivo_original = dataset / "data.yaml"

    if not archivo_original.exists():
        print("No se encontro dataset/data.yaml")
        return

    with archivo_original.open(encoding="utf-8") as archivo:
        configuracion = yaml.safe_load(archivo)

    # Ajustar las rutas para este computador
    configuracion["path"] = dataset.as_posix()
    configuracion["train"] = "train/images"
    configuracion["val"] = "valid/images"
    configuracion["test"] = "test/images"

    resultados = proyecto / "runs"
    resultados.mkdir(exist_ok=True)

    archivo_config = resultados / "data_entrenamiento.yaml"

    with archivo_config.open("w", encoding="utf-8") as archivo:
        yaml.safe_dump(configuracion, archivo, allow_unicode=True)

    # Cargar YOLOv8 nano y comenzar la prueba
    modelo = YOLO("yolov8n.pt")

    modelo.train(
        data=str(archivo_config),
        epochs=3,
        imgsz=640,
        batch=4,
        device="cpu",
        workers=0,
        fliplr=0.0,
        project=str(resultados),
        name="prueba_patentes"
    )

    print("Entrenamiento terminado")
    print("Modelo guardado en:", modelo.trainer.best)


if __name__ == "__main__":
    main()
