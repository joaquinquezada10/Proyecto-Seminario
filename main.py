# Detecta patentes en fotos o videos y abre el resultado de las fotos.

from pathlib import Path
import math
import cv2
import tkinter as tk
from tkinter import filedialog

from ultralytics import YOLO
from ultralytics.engine.results import Results


def seleccionar_archivo():
    ventana = tk.Tk()
    ventana.withdraw()

    try:
        return filedialog.askopenfilename(
            parent=ventana,
            title="Selecciona una foto o video",
            filetypes=[
                ("Fotos", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Videos", "*.mp4 *.avi *.mov *.mkv *.m4v"),
                ("Todos los archivos", "*.*")
            ]
        )
    finally:
        ventana.destroy()


def main():
    carpeta = Path(__file__).resolve().parent
    ruta_modelo = carpeta / "modelos" / "best.pt"

    if not ruta_modelo.is_file():
        print("No se encontro modelos/best.pt")
        return

    archivo = seleccionar_archivo()

    if not archivo:
        print("Seleccion cancelada.")
        return

    entrada = Path(archivo)
    fotos = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    videos = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
    extension = entrada.suffix.lower()

    if not entrada.is_file() or extension not in fotos | videos:
        print("Selecciona una foto o video compatible.")
        return

    es_foto = extension in fotos
    modelo = YOLO(str(ruta_modelo))

    salida = carpeta / "runs" / "deteccion"
    numero = 2

    while salida.exists():
        salida = carpeta / "runs" / f"deteccion{numero}"
        numero += 1

    salida.mkdir(parents=True)

    fps = 30.0
    if not es_foto:
        captura = cv2.VideoCapture(str(entrada))
        try:
            if not captura.isOpened():
                raise RuntimeError("No se pudo abrir el video.")
            fps = captura.get(cv2.CAP_PROP_FPS)
        finally:
            captura.release()

        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0

    print("Procesando archivo...")

    resultados = modelo.predict(
        source=str(entrada),
        imgsz=640,
        conf=0.25,
        device="cpu",
        save=es_foto,
        stream=True,
        verbose=False,
        project=str(carpeta / "runs"),
        name=salida.name,
        exist_ok=True
    )

    escritor = None
    cantidad = 0
    ruta_video = salida / (entrada.stem + "_deteccion.mp4")

    try:
        for resultado in resultados:
            if not isinstance(resultado, Results):
                raise TypeError("YOLO no devolvio un resultado de deteccion.")

            cantidad += 1

            if es_foto:
                try:
                    resultado.show()
                except Exception as error:
                    print(f"No se pudo abrir el visor: {error}")
            else:
                imagen = resultado.plot()

                if escritor is None:
                    alto, ancho = imagen.shape[:2]
                    escritor = cv2.VideoWriter(
                        str(ruta_video),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (ancho, alto)
                    )
                    if not escritor.isOpened():
                        raise RuntimeError("No se pudo crear el video MP4.")

                escritor.write(imagen)

                if cantidad % 100 == 0:
                    print(f"Fotogramas procesados: {cantidad}")
    finally:
        if escritor is not None:
            escritor.release()

    if cantidad == 0:
        print("No se pudieron procesar imagenes del archivo.")
    elif es_foto:
        print(f"Resultados guardados en: {salida}")
    else:
        print(f"Video MP4 guardado en: {ruta_video}")


if __name__ == "__main__":
    main()
