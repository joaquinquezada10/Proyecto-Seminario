from pathlib import Path
from ultralytics import YOLO


def main():
    carpeta = Path(__file__).resolve().parent

    modelo = YOLO("yolov8n.pt")

    modelo.train(
        data=str(carpeta / "dataset" / "data.yaml"),
        epochs=3,
        imgsz=640,
        batch=4,
        device="cpu",
        workers=0,
        project=str(carpeta / "runs"),
        name="prueba_patentes"
    )


if __name__ == "__main__":
    main()