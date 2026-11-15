# Detecta patentes en fotos o videos, lee sus caracteres y verifica autorizacion.

import csv
from datetime import datetime
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
from ultralytics import YOLO

from lector import leer_patente
from registro import verificar_patente

CARPETA_PROYECTO = Path(__file__).resolve().parent
RUTA_MODELO = CARPETA_PROYECTO / "modelos" / "best.pt"
CARPETA_VEHICULOS = CARPETA_PROYECTO / "Registro de vehiculos"
CARPETA_SALIDAS = CARPETA_PROYECTO / "runs" / "control_accesos"
RUTA_LOG_ACCESOS = CARPETA_SALIDAS / "registro_accesos.csv"

EXTENSIONES_FOTOS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXTENSIONES_VIDEOS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def validar_y_formatear_chilena(texto: str) -> Tuple[str, str, bool]:
    """
    Evalua el texto segun el formato de patente chilena moderna (4 letras + 2 numeros):
    Ejemplo: 'PTKX93' -> 'PT KX-93'
    Aplica correcciones de OCR posicionales:
    - Posiciones 1 a 4: Si se leyo un numero confuso, lo transforma a su letra equivalente.
    - Posiciones 5 y 6: Si se leyo una letra confusa, la transforma a su numero equivalente.
    Retorna: (patente_limpia, patente_formateada, es_formato_moderno)
    """
    limpio = re.sub(r"[^A-Z0-9]", "", texto.upper().strip())

    if len(limpio) == 6:
        mapa_num_a_letra = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B"}
        mapa_letra_a_num = {
            "O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "T": "1",
            "Z": "2", "S": "5", "B": "8"
        }

        chars = list(limpio)
        # Primeros 4 caracteres deben ser letras
        for i in range(4):
            if chars[i].isdigit() and chars[i] in mapa_num_a_letra:
                chars[i] = mapa_num_a_letra[chars[i]]

        # Ultimos 2 caracteres deben ser digitos
        for i in range(4, 6):
            if chars[i].isalpha() and chars[i] in mapa_letra_a_num:
                chars[i] = mapa_letra_a_num[chars[i]]

        corregida = "".join(chars)
        es_moderna = bool(re.match(r"^[A-Z]{4}[0-9]{2}$", corregida))

        if es_moderna:
            formato = f"{corregida[:2]} {corregida[2:4]}-{corregida[4:]}"
            return corregida, formato, True

        formato_fallback = f"{limpio[:2]} {limpio[2:4]}-{limpio[4:]}"
        return limpio, formato_fallback, False

    return limpio, limpio, False


def registrar_evento_acceso(
    archivo_origen: str,
    patente_detectada: str,
    patente_registrada: str,
    conductor: str,
    estado: str,
    coincidencia: str
) -> None:
    """
    Guarda un registro de cada vehiculo detectado en un archivo CSV de auditoria.
    """
    CARPETA_SALIDAS.mkdir(parents=True, exist_ok=True)
    existe = RUTA_LOG_ACCESOS.is_file()

    ahora = datetime.now()
    fila = {
        "Fecha": ahora.strftime("%Y-%m-%d"),
        "Hora": ahora.strftime("%H:%M:%S"),
        "Archivo": archivo_origen,
        "Patente_Detectada": patente_detectada,
        "Patente_Registrada": patente_registrada,
        "Conductor": conductor,
        "Estado": estado,
        "Coincidencia": coincidencia
    }

    with open(RUTA_LOG_ACCESOS, mode="a", encoding="utf-8-sig", newline="") as f:
        escritor = csv.DictWriter(
            f,
            fieldnames=[
                "Fecha", "Hora", "Archivo", "Patente_Detectada",
                "Patente_Registrada", "Conductor", "Estado", "Coincidencia"
            ]
        )
        if not existe:
            escritor.writeheader()
        escritor.writerow(fila)


def dibujar_recuadro_con_etiqueta(
    imagen: Any,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    texto_linea1: str,
    texto_linea2: str,
    color_bgr: Tuple[int, int, int]
) -> None:
    """
    Dibuja un rectangulo alrededor de la patente y una pancarta con texto de alta legibilidad.
    """
    # 1. Bounding box de la patente
    cv2.rectangle(imagen, (x1, y1), (x2, y2), color_bgr, 3)

    # 2. Pancarta superior para el texto
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    escala = 0.60
    grosor = 2

    (w1, h1), _ = cv2.getTextSize(texto_linea1, fuente, escala, grosor)
    w2, h2 = 0, 0
    if texto_linea2:
        (w2, h2), _ = cv2.getTextSize(texto_linea2, fuente, escala - 0.1, grosor - 1)

    ancho_fondo = max(w1, w2) + 16
    alto_fondo = (h1 + 8) + (h2 + 8 if texto_linea2 else 4)

    bx1 = x1
    by1 = max(0, y1 - alto_fondo - 4)
    bx2 = min(imagen.shape[1], x1 + ancho_fondo)
    by2 = max(by1 + alto_fondo, y1)

    # Fondo solido de estado
    cv2.rectangle(imagen, (bx1, by1), (bx2, by2), color_bgr, -1)

    # Texto linea 1 (Estado y patente)
    cv2.putText(
        imagen,
        texto_linea1,
        (bx1 + 8, by1 + h1 + 4),
        fuente,
        escala,
        (255, 255, 255),
        grosor,
        cv2.LINE_AA
    )

    # Texto linea 2 (Conductor / Detalle)
    if texto_linea2:
        cv2.putText(
            imagen,
            texto_linea2,
            (bx1 + 8, by1 + h1 + h2 + 10),
            fuente,
            escala - 0.1,
            (255, 255, 255),
            grosor - 1,
            cv2.LINE_AA
        )


def procesar_cuadro(
    imagen: Any,
    modelo: YOLO,
    nombre_archivo: str = "",
    guardar_en_log: bool = True
) -> Tuple[Any, List[Dict[str, Any]]]:
    """
    Procesa una imagen/fotograma:
    1. Detecta patentes con YOLO.
    2. Recorta cada patente.
    3. Lee los caracteres con EasyOCR.
    4. Valida el formato chileno moderno.
    5. Consulta la base de datos de patentes autorizadas.
    6. Dibuja las etiquetas verdes (autorizado) o rojas (no autorizado).
    """
    predicciones = modelo.predict(source=imagen, conf=0.25, verbose=False)
    resultado = predicciones[0] if isinstance(predicciones, list) else next(iter(predicciones))

    cajas_xyxy: List[List[float]] = []
    boxes = getattr(resultado, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        cajas_xyxy = boxes.xyxy.tolist()

    anotada = imagen.copy()
    alto_img, ancho_img = imagen.shape[:2]
    vehiculos_detectados: List[Dict[str, Any]] = []

    for box in cajas_xyxy:
        x1 = max(0, int(box[0]))
        y1 = max(0, int(box[1]))
        x2 = min(ancho_img, int(box[2]))
        y2 = min(alto_img, int(box[3]))

        recorte = imagen[y1:y2, x1:x2]
        texto_ocr, conf_ocr = leer_patente(recorte)

        patente_limpia, patente_formato, es_moderna = validar_y_formatear_chilena(texto_ocr)

        if not patente_limpia:
            # Deteccion visual sin texto reconocible
            dibujar_recuadro_con_etiqueta(
                anotada, x1, y1, x2, y2,
                "PATENTE NO LEGIBLE", "",
                (128, 128, 128)
            )
            continue

        # Verificar autorizacion en registro
        autorizado, detalle = verificar_patente(patente_limpia)

        conductor = detalle.get("conductor", "Desconocido")
        patente_registrada = detalle.get("patente_registrada", "")
        coincidencia = detalle.get("coincidencia", "NO_ENCONTRADA")

        # Colores: Verde (0, 180, 0) para autorizado, Rojo (0, 0, 220) para no autorizado
        if autorizado:
            color = (0, 180, 0)
            linea1 = f"AUTORIZADO: {patente_formato}"
            linea2 = f"Conductor: {conductor}"
        else:
            color = (0, 0, 220)
            linea1 = f"NO AUTORIZADO: {patente_formato}"
            linea2 = "Acceso denegado a obra"

        dibujar_recuadro_con_etiqueta(anotada, x1, y1, x2, y2, linea1, linea2, color)

        datos_vehiculo = {
            "patente_detectada": patente_formato,
            "patente_limpia": patente_limpia,
            "patente_registrada": patente_registrada,
            "conductor": conductor,
            "autorizado": autorizado,
            "confianza_ocr": conf_ocr,
            "es_moderna": es_moderna
        }
        vehiculos_detectados.append(datos_vehiculo)

        if guardar_en_log:
            registrar_evento_acceso(
                archivo_origen=nombre_archivo,
                patente_detectada=patente_formato,
                patente_registrada=patente_registrada,
                conductor=conductor,
                estado="AUTORIZADO" if autorizado else "NO AUTORIZADO",
                coincidencia=coincidencia
            )

    return anotada, vehiculos_detectados


def procesar_archivo_foto(ruta_archivo: Path, modelo: YOLO) -> None:
    """
    Procesa una foto individual, muestra el resultado visualmente y lo guarda.
    """
    print(f"\nProcesando foto: {ruta_archivo.name}")
    imagen = cv2.imread(str(ruta_archivo))
    if imagen is None:
        print(f"Error: No se pudo abrir la imagen en {ruta_archivo}")
        return

    anotada, vehiculos = procesar_cuadro(
        imagen,
        modelo,
        nombre_archivo=ruta_archivo.name,
        guardar_en_log=True
    )

    if not vehiculos:
        print("  -> No se detectaron patentes legibles en la foto.")
    else:
        for v in vehiculos:
            estado_txt = "AUTORIZADO" if v["autorizado"] else "NO AUTORIZADO"
            print(f"  -> [{estado_txt}] Patente: {v['patente_detectada']} | Conductor: {v['conductor']}")

    CARPETA_SALIDAS.mkdir(parents=True, exist_ok=True)
    salida_img = CARPETA_SALIDAS / f"resultado_{ruta_archivo.name}"
    cv2.imwrite(str(salida_img), anotada)
    print(f"  Foto guardada en: {salida_img}")

    # Mostrar ventana interactiva
    titulo = f"Control de Acceso - {ruta_archivo.name}"
    try:
        cv2.namedWindow(titulo, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(titulo, 1024, 680)
        cv2.imshow(titulo, anotada)
        print("  -> Mostrando ventana. Haz clic en la imagen y presiona cualquier tecla para continuar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Aviso ventana visual: {e}")


def procesar_archivo_video(ruta_archivo: Path, modelo: YOLO) -> None:
    """
    Procesa un archivo de video fotograma a fotograma y guarda el video resultante.
    """
    print(f"\nProcesando video: {ruta_archivo.name}")
    captura = cv2.VideoCapture(str(ruta_archivo))
    if not captura.isOpened():
        print(f"Error: No se pudo abrir el video en {ruta_archivo}")
        return

    fps = captura.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0

    ancho = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))

    CARPETA_SALIDAS.mkdir(parents=True, exist_ok=True)
    salida_video = CARPETA_SALIDAS / f"resultado_{ruta_archivo.stem}.mp4"

    escritor = cv2.VideoWriter(
        str(salida_video),
        cv2.VideoWriter.fourcc(*"mp4v"),
        fps,
        (ancho, alto)
    )

    fotograma_num = 0
    vehiculos_registrados_video = set()

    try:
        while True:
            ret, frame = captura.read()
            if not ret:
                break

            fotograma_num += 1

            # Procesar OCR cada 15 fotogramas o cuando cambia para optimizar velocidad
            guardar_log = (fotograma_num % 15 == 0)

            anotado, vehiculos = procesar_cuadro(
                frame,
                modelo,
                nombre_archivo=ruta_archivo.name,
                guardar_en_log=guardar_log
            )

            for v in vehiculos:
                clave = (v["patente_detectada"], v["autorizado"])
                if clave not in vehiculos_registrados_video:
                    vehiculos_registrados_video.add(clave)
                    est = "AUTORIZADO" if v["autorizado"] else "NO AUTORIZADO"
                    print(f"  Frame {fotograma_num}: [{est}] {v['patente_detectada']} ({v['conductor']})")

            escritor.write(anotado)

            if total_frames > 0 and fotograma_num % 30 == 0:
                progreso = (fotograma_num / total_frames) * 100
                print(f"  Progreso: {fotograma_num}/{total_frames} ({progreso:.1f}%)")

    finally:
        captura.release()
        escritor.release()

    print(f"Video procesado y guardado en: {salida_video}")


def main() -> None:
    print("=" * 60)
    print(" SISTEMA DE CONTROL DE ACCESO VEHICULAR - OBRA ")
    print("=" * 60)

    if not RUTA_MODELO.is_file():
        print(f"Error: No se encontro el modelo entrenado en {RUTA_MODELO}")
        return

    # Cargar todos los archivos de la carpeta por defecto
    CARPETA_VEHICULOS.mkdir(parents=True, exist_ok=True)
    archivos = [
        f for f in sorted(CARPETA_VEHICULOS.iterdir())
        if f.is_file() and f.suffix.lower() in (EXTENSIONES_FOTOS | EXTENSIONES_VIDEOS)
    ]

    if not archivos:
        print(f"\nNo se encontraron fotos ni videos en la carpeta:")
        print(f"  {CARPETA_VEHICULOS}")
        print("Agrega archivos a esa carpeta y vuelve a ejecutar el programa.")
        return

    print(f"\nCargando modelo YOLO ({RUTA_MODELO.name})...")
    modelo = YOLO(str(RUTA_MODELO))

    print(f"Archivos encontrados en 'Registro de vehiculos': {len(archivos)}")

    for i, archivo in enumerate(archivos, start=1):
        print(f"\n--- [{i}/{len(archivos)}] Procesando: {archivo.name} ---")
        ext = archivo.suffix.lower()

        if ext in EXTENSIONES_FOTOS:
            procesar_archivo_foto(archivo, modelo)
        elif ext in EXTENSIONES_VIDEOS:
            procesar_archivo_video(archivo, modelo)
        else:
            print(f"Formato no compatible: {ext}")

    print("\n" + "=" * 60)
    print(" PROCESAMIENTO COMPLETADO ")
    print(f"• Resultados guardados en: {CARPETA_SALIDAS}")
    print(f"• Registro de auditoria CSV: {RUTA_LOG_ACCESOS}")
    print("=" * 60)


if __name__ == "__main__":
    main()
