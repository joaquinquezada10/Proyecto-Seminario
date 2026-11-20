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


def extraer_y_validar_patente_chilena(texto: str) -> Tuple[str, str, str, bool]:
    """
    Evalua el texto segun los dos formatos oficiales de patentes chilenas:
    1. Moderna (2007 a la fecha): 4 letras + 2 numeros (ej: PTKX93 -> 'PT KX-93')
    2. Antigua (1985 a 2007): 2 letras + 4 numeros (ej: BB1000 -> 'BB 10-00')

    Corrige anomalías comunes de OCR:
    - Remache circular central ('0' u 'O' leido entre bloques).
    - Confusiones posicionales (letras en posiciones numericas y viceversa).
    - Palabras de ruido residuales (CHILE, WILE).

    Retorna: (patente_limpia, patente_formato, tipo_formato, es_valida)
             tipo_formato: 'MODERNA', 'ANTIGUA', o ''
    """
    raw = re.sub(r"[^A-Z0-9]", "", texto.upper().strip())
    raw = raw.replace("CHILE", "").replace("WILE", "").replace("CHLE", "")

    mapa_num_a_letra = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B"}
    mapa_letra_a_num = {
        "O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "T": "1",
        "Z": "2", "S": "5", "B": "8"
    }

    candidatos = []
    if len(raw) == 6:
        candidatos.append(raw)
    elif len(raw) == 7:
        # Remache central entre letras y numeros en patente moderna
        if raw[4] in ("0", "O", "Q", "D"):
            candidatos.append(raw[:4] + raw[5:])
        # Remache central en patente antigua
        if raw[2] in ("0", "O", "Q", "D"):
            candidatos.append(raw[:2] + raw[3:])
        candidatos.append(raw[:6])
        candidatos.append(raw[1:])
    elif len(raw) > 7:
        # Extraer posibles subsecuencias de 6 caracteres
        for i in range(len(raw) - 5):
            candidatos.append(raw[i:i + 6])
    else:
        return raw, raw, "", False

    mejores = []
    for cand in candidatos:
        # 1. Probar hipotesis Formato Moderno (4 letras + 2 numeros)
        chars_mod = list(cand)
        score_mod = 0
        for i in range(4):
            if chars_mod[i].isalpha():
                score_mod += 1
            elif chars_mod[i] in mapa_num_a_letra:
                chars_mod[i] = mapa_num_a_letra[chars_mod[i]]
        for i in range(4, 6):
            if chars_mod[i].isdigit():
                score_mod += 1
            elif chars_mod[i] in mapa_letra_a_num:
                chars_mod[i] = mapa_letra_a_num[chars_mod[i]]
        cand_mod = "".join(chars_mod)
        if re.match(r"^[A-Z]{4}[0-9]{2}$", cand_mod):
            formato_mod = f"{cand_mod[:2]} {cand_mod[2:4]}-{cand_mod[4:]}"
            mejores.append((score_mod, cand_mod, formato_mod, "MODERNA"))

        # 2. Probar hipotesis Formato Antiguo (2 letras + 4 numeros)
        chars_ant = list(cand)
        score_ant = 0
        for i in range(2):
            if chars_ant[i].isalpha():
                score_ant += 1
            elif chars_ant[i] in mapa_num_a_letra:
                chars_ant[i] = mapa_num_a_letra[chars_ant[i]]
        for i in range(2, 6):
            if chars_ant[i].isdigit():
                score_ant += 1
            elif chars_ant[i] in mapa_letra_a_num:
                chars_ant[i] = mapa_letra_a_num[chars_ant[i]]
        cand_ant = "".join(chars_ant)
        if re.match(r"^[A-Z]{2}[0-9]{4}$", cand_ant):
            formato_ant = f"{cand_ant[:2]} {cand_ant[2:4]}-{cand_ant[4:]}"
            mejores.append((score_ant, cand_ant, formato_ant, "ANTIGUA"))

    if mejores:
        # Seleccionar el candidato con mayor coincidencia nativa
        mejores.sort(key=lambda x: x[0], reverse=True)
        _, limpia, formato, tipo = mejores[0]
        return limpia, formato, tipo, True

    formato_fallback = f"{raw[:2]} {raw[2:4]}-{raw[4:]}" if len(raw) == 6 else raw
    return raw, formato_fallback, "", False


def registrar_evento_acceso(
    archivo_origen: str,
    patente_detectada: str,
    patente_registrada: str,
    conductor: str,
    tipo_patente: str,
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
        "Tipo_Patente": tipo_patente,
        "Estado": estado,
        "Coincidencia": coincidencia
    }

    with open(RUTA_LOG_ACCESOS, mode="a", encoding="utf-8-sig", newline="") as f:
        escritor = csv.DictWriter(
            f,
            fieldnames=[
                "Fecha", "Hora", "Archivo", "Patente_Detectada",
                "Patente_Registrada", "Conductor", "Tipo_Patente",
                "Estado", "Coincidencia"
            ]
        )
        if not existe:
            escritor.writeheader()
        escritor.writerow(fila)


def filtrar_y_unificar_cajas(
    cajas_xyxy: List[List[float]],
    confianzas: List[float],
    ancho_img: int,
    alto_img: int
) -> List[Tuple[List[int], float]]:
    """
    Descarta falsos positivos por proporcion (aspect ratio) o baja confianza,
    y unifica cajas solapadas que pertenecen a una misma patente.
    """
    cajas_filtradas: List[Tuple[List[int], float]] = []

    for box, conf in zip(cajas_xyxy, confianzas):
        conf_val = float(conf)
        if conf_val < 0.35:
            continue

        x1 = max(0, int(box[0]))
        y1 = max(0, int(box[1]))
        x2 = min(ancho_img, int(box[2]))
        y2 = min(alto_img, int(box[3]))

        w = x2 - x1
        h = y2 - y1

        if w < 35 or h < 15:
            continue

        aspect = w / float(max(h, 1))
        # Las patentes chilenas rectangulares tienen proporcion entre 1.3 y 5.5
        if aspect < 1.3 or aspect > 5.5:
            continue

        cajas_filtradas.append(([x1, y1, x2, y2], conf_val))

    if not cajas_filtradas:
        return []

    # Unificar cajas fuertemente solapadas (mismo vehiculo / patente detectada en fragmentos)
    cajas_unificadas: List[Tuple[List[int], float]] = []
    usadas = [False] * len(cajas_filtradas)

    for i in range(len(cajas_filtradas)):
        if usadas[i]:
            continue

        caja_a, conf_a = cajas_filtradas[i]
        xa1, ya1, xa2, ya2 = caja_a
        conf_max = conf_a
        usadas[i] = True

        for j in range(i + 1, len(cajas_filtradas)):
            if usadas[j]:
                continue

            caja_b, conf_b = cajas_filtradas[j]
            xb1, yb1, xb2, yb2 = caja_b

            inter_x1 = max(xa1, xb1)
            inter_y1 = max(ya1, yb1)
            inter_x2 = min(xa2, xb2)
            inter_y2 = min(ya2, yb2)

            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            area_inter = inter_w * inter_h

            area_a = (xa2 - xa1) * (ya2 - ya1)
            area_b = (xb2 - xb1) * (yb2 - yb1)
            area_union = area_a + area_b - area_inter

            iou = area_inter / float(max(area_union, 1))
            solape_min = area_inter / float(max(min(area_a, area_b), 1))

            if iou > 0.30 or solape_min > 0.45:
                xa1 = min(xa1, xb1)
                ya1 = min(ya1, yb1)
                xa2 = max(xa2, xb2)
                ya2 = max(ya2, yb2)
                conf_max = max(conf_max, conf_b)
                usadas[j] = True

        cajas_unificadas.append(([xa1, ya1, xa2, ya2], conf_max))

    return cajas_unificadas


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
    cv2.rectangle(imagen, (x1, y1), (x2, y2), color_bgr, 3)

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

    cv2.rectangle(imagen, (bx1, by1), (bx2, by2), color_bgr, -1)
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
    2. Filtra y unifica cajas solapadas o invalidas.
    3. Agrega margen de seguridad para no cortar letras en los bordes.
    4. Lee caracteres con EasyOCR.
    5. Valida formatos chilenos (Moderna y Antigua).
    6. Consulta registro de patentes autorizadas.
    7. Dibuja anotaciones legibles.
    """
    predicciones = modelo.predict(source=imagen, conf=0.25, verbose=False)
    resultado = predicciones[0] if isinstance(predicciones, list) else next(iter(predicciones))

    cajas_xyxy: List[List[float]] = []
    confianzas_yolo: List[float] = []

    boxes = getattr(resultado, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        cajas_xyxy = boxes.xyxy.tolist()
        confianzas_yolo = boxes.conf.tolist()

    anotada = imagen.copy()
    alto_img, ancho_img = imagen.shape[:2]
    vehiculos_detectados: List[Dict[str, Any]] = []

    cajas_validas = filtrar_y_unificar_cajas(cajas_xyxy, confianzas_yolo, ancho_img, alto_img)

    for box, conf_yolo in cajas_validas:
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1

        # Agregar margen para capturar caracteres cercanos a los bordes
        pad_x = int(bw * 0.12)
        pad_y = int(bh * 0.08)
        px1 = max(0, x1 - pad_x)
        px2 = min(ancho_img, x2 + pad_x)
        py1 = max(0, y1 - pad_y)
        py2 = min(alto_img, y2 + pad_y)

        recorte = imagen[py1:py2, px1:px2]
        texto_ocr, conf_ocr = leer_patente(recorte)

        patente_limpia, patente_formato, tipo_patente, es_valida = extraer_y_validar_patente_chilena(texto_ocr)

        if not patente_limpia:
            dibujar_recuadro_con_etiqueta(
                anotada, px1, py1, px2, py2,
                "PATENTE NO LEGIBLE", "",
                (128, 128, 128)
            )
            continue

        if not es_valida:
            dibujar_recuadro_con_etiqueta(
                anotada, px1, py1, px2, py2,
                "FORMATO NO VALIDO", f"Leido: {patente_formato}",
                (128, 128, 128)
            )
            continue

        # Verificar autorizacion en registro CSV
        autorizado, detalle = verificar_patente(patente_limpia)

        conductor = detalle.get("conductor", "Desconocido")
        patente_registrada = detalle.get("patente_registrada", "")
        coincidencia = detalle.get("coincidencia", "NO_ENCONTRADA")

        # Color: Verde para autorizado, Rojo para no autorizado
        if autorizado:
            color = (0, 180, 0)
            linea1 = f"AUTORIZADO: {patente_formato}"
            linea2 = f"Conductor: {conductor} ({tipo_patente})"
        else:
            color = (0, 0, 220)
            linea1 = f"NO AUTORIZADO: {patente_formato}"
            linea2 = f"Acceso denegado | Tipo: {tipo_patente}"

        dibujar_recuadro_con_etiqueta(anotada, px1, py1, px2, py2, linea1, linea2, color)

        datos_vehiculo = {
            "patente_detectada": patente_formato,
            "patente_limpia": patente_limpia,
            "patente_registrada": patente_registrada,
            "conductor": conductor,
            "tipo_patente": tipo_patente,
            "autorizado": autorizado,
            "confianza_ocr": conf_ocr,
            "confianza_yolo": conf_yolo
        }
        vehiculos_detectados.append(datos_vehiculo)

        if guardar_en_log:
            registrar_evento_acceso(
                archivo_origen=nombre_archivo,
                patente_detectada=patente_formato,
                patente_registrada=patente_registrada,
                conductor=conductor,
                tipo_patente=tipo_patente,
                estado="AUTORIZADO" if autorizado else "NO AUTORIZADO",
                coincidencia=coincidencia
            )

    return anotada, vehiculos_detectados


def procesar_archivo_foto(
    ruta_archivo: Path,
    modelo: YOLO
) -> Tuple[Optional[Any], List[Dict[str, Any]]]:
    """
    Procesa una foto individual sin bloquear, guarda el resultado y lo retorna para la galeria final.
    """
    imagen = cv2.imread(str(ruta_archivo))
    if imagen is None:
        print(f"Error: No se pudo abrir la imagen en {ruta_archivo}")
        return None, []

    anotada, vehiculos = procesar_cuadro(
        imagen,
        modelo,
        nombre_archivo=ruta_archivo.name,
        guardar_en_log=True
    )

    if not vehiculos:
        print("  -> Sin patentes validas reconocidas.")
    else:
        for v in vehiculos:
            estado_txt = "AUTORIZADO" if v["autorizado"] else "NO AUTORIZADO"
            print(f"  -> [{estado_txt}] Patente: {v['patente_detectada']} ({v['tipo_patente']}) | Conductor: {v['conductor']}")

    CARPETA_SALIDAS.mkdir(parents=True, exist_ok=True)
    salida_img = CARPETA_SALIDAS / f"resultado_{ruta_archivo.name}"
    cv2.imwrite(str(salida_img), anotada)

    return anotada, vehiculos


def procesar_archivo_video(ruta_archivo: Path, modelo: YOLO) -> None:
    """
    Procesa un video fotograma a fotograma y guarda el video resultante en runs/control_accesos.
    """
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
                print(f"  Progreso video: {fotograma_num}/{total_frames} ({progreso:.1f}%)")

    finally:
        captura.release()
        escritor.release()

    print(f"  Video guardado en: {salida_video}")


def mostrar_galeria(lista_galeria: List[Dict[str, Any]]) -> None:
    """
    Abre una galeria visual interactiva al final del procesamiento por lotes:
    - Tecla [A] o [Flecha Izquierda]: Imagen anterior
    - Tecla [D] o [Flecha Derecha]: Imagen siguiente
    - Tecla [Q] o [ESC]: Cerrar galeria
    """
    if not lista_galeria:
        print("\nNo hay imagenes procesadas para mostrar en la galeria.")
        return

    total = len(lista_galeria)
    idx = 0
    nombre_ventana = "Galeria de control de acceso"

    cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nombre_ventana, 1200, 750)

    print("\n" + "=" * 60)
    print(" MOSTRANDO GALERIA INTERACTIVA ")
    print(" Controles:")
    print("   [A] o [<-]  : Imagen anterior")
    print("   [D] o [->]  : Imagen siguiente")
    print("   [Q] o [ESC] : Cerrar galeria")
    print("=" * 60)

    while True:
        item = lista_galeria[idx]
        imagen_base = item["imagen"].copy()
        nombre_archivo = item["nombre_archivo"]

        # Dibujar banner inferior exclusivamente con [N°/Total] nombre_archivo
        h_img, w_img = imagen_base.shape[:2]
        alto_banner = max(45, int(h_img * 0.08))
        color_banner = (40, 40, 40)

        cv2.rectangle(
            imagen_base,
            (0, h_img - alto_banner),
            (w_img, h_img),
            color_banner,
            -1
        )

        texto_banner = f"[{idx + 1}/{total}] {nombre_archivo}"

        fuente = cv2.FONT_HERSHEY_SIMPLEX
        escala_fuente = min(0.65, max(0.45, w_img / 1200.0))
        grosor = 2 if escala_fuente >= 0.55 else 1

        cv2.putText(
            imagen_base,
            texto_banner,
            (15, h_img - int(alto_banner * 0.35)),
            fuente,
            escala_fuente,
            (255, 255, 255),
            grosor,
            cv2.LINE_AA
        )

        cv2.imshow(nombre_ventana, imagen_base)

        tecla = cv2.waitKeyEx(50)

        # Si el usuario cerro la ventana con la 'X'
        try:
            if cv2.getWindowProperty(nombre_ventana, cv2.WND_PROP_VISIBLE) < 1:
                break
        except Exception:
            break

        if tecla in (ord("q"), ord("Q"), 27):
            break
        # Flecha izquierda (Windows: 2424832, Linux: 81) o tecla 'A'
        elif tecla in (ord("a"), ord("A"), 2424832, 81):
            idx = (idx - 1) % total
        # Flecha derecha (Windows: 2555904, Linux: 83) o tecla 'D'
        elif tecla in (ord("d"), ord("D"), 2555904, 83):
            idx = (idx + 1) % total

    cv2.destroyAllWindows()
    print("Galeria cerrada correctamente.")


def main() -> None:
    print("=" * 60)
    print(" SISTEMA DE CONTROL DE ACCESO VEHICULAR - OBRA ")
    print("=" * 60)

    if not RUTA_MODELO.is_file():
        print(f"Error: No se encontro el modelo entrenado en {RUTA_MODELO}")
        return

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
    print("Iniciando procesamiento por lotes...\n")

    lista_galeria: List[Dict[str, Any]] = []

    for i, archivo in enumerate(archivos, start=1):
        ext = archivo.suffix.lower()
        print(f"[{i}/{len(archivos)}] Procesando: {archivo.name}")

        if ext in EXTENSIONES_FOTOS:
            anotada, vehiculos = procesar_archivo_foto(archivo, modelo)
            if anotada is not None:
                lista_galeria.append({
                    "nombre_archivo": archivo.name,
                    "imagen": anotada,
                    "vehiculos": vehiculos
                })
        elif ext in EXTENSIONES_VIDEOS:
            procesar_archivo_video(archivo, modelo)
        else:
            print(f"  Formato no compatible: {ext}")

    print("\n" + "=" * 60)
    print(" PROCESAMIENTO POR LOTES COMPLETADO ")
    print(f"• Resultados guardados en: {CARPETA_SALIDAS}")
    print(f"• Registro de auditoria CSV: {RUTA_LOG_ACCESOS}")
    print("=" * 60)

    # Abrir la galeria interactiva al final con todas las imagenes procesadas
    if lista_galeria:
        mostrar_galeria(lista_galeria)


if __name__ == "__main__":
    main()
