# Lee los caracteres de la patente mediante EasyOCR

from pathlib import Path
import re
import tkinter as tk
from tkinter import filedialog
import cv2
import easyocr
from ultralytics import YOLO

# Directorio local para almacenar y cargar los modelos de EasyOCR sin problemas de permisos
CARPETA = Path(__file__).resolve().parent
RUTA_MODELOS_OCR = CARPETA / "modelos" / "easyocr"
RUTA_MODELOS_OCR.mkdir(parents=True, exist_ok=True)

lector = easyocr.Reader(
    ["en"],
    gpu=False,
    model_storage_directory=str(RUTA_MODELOS_OCR),
    user_network_directory=str(RUTA_MODELOS_OCR),
    download_enabled=True
)


def formatear_patente(texto: str) -> str:
    """
    Formatea patentes chilenas segun su tipo:
    - Moderna (4 letras + 2 numeros): 'PTKX93' -> 'PT KX-93'
    - Antigua (2 letras + 4 numeros): 'BB1000' -> 'BB 10-00'
    """
    limpio = re.sub(r"[^A-Z0-9]", "", texto.upper().strip())
    if len(limpio) == 6:
        # Moderna: 4 letras + 2 numeros
        if limpio[:4].isalpha() and limpio[4:].isdigit():
            return f"{limpio[:2]} {limpio[2:4]}-{limpio[4:]}"
        # Antigua: 2 letras + 4 numeros
        if limpio[:2].isalpha() and limpio[2:].isdigit():
            return f"{limpio[:2]} {limpio[2:4]}-{limpio[4:]}"
    return limpio


def preprocesar_imagen(imagen):
    """
    Amplia el recorte y mejora el contraste para facilitar la lectura.
    """
    if imagen is None or imagen.size == 0:
        return None

    alto, ancho = imagen.shape[:2]
    escala = max(2.0, 180 / float(alto))

    imagen = cv2.resize(
        imagen,
        (int(ancho * escala), int(alto * escala)),
        interpolation=cv2.INTER_CUBIC
    )

    if len(imagen.shape) == 3:
        if imagen.shape[2] == 4:
            gris = cv2.cvtColor(imagen, cv2.COLOR_BGRA2GRAY)
        else:
            gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        gris = imagen.copy()

    suavizado = cv2.bilateralFilter(gris, 5, 50, 50)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(suavizado)


def generar_variantes_ocr(imagen):
    """
    Genera versiones con diferentes niveles de contraste y umbralizacion.
    """
    mejorada = preprocesar_imagen(imagen)
    if mejorada is None:
        return []

    _, otsu = cv2.threshold(
        mejorada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    _, otsu_invertida = cv2.threshold(
        mejorada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    adaptativa = cv2.adaptiveThreshold(
        mejorada,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return [mejorada, otsu, otsu_invertida, adaptativa]


PALABRAS_RUIDO = {"CHILE", "WILE", "CHLE", "CHI", "ILE", "CHIL", "HILE"}


def _obtener_lecturas_linea(salida_ocr, alto_img=None):
    """
    Une fragmentos que pertenecen a una misma linea, sin mezclar
    detecciones que EasyOCR ubico en alturas diferentes.
    Descarta palabras de ruido (CHILE, marcas de portapatente) y
    detecciones situadas en el tercio inferior del recorte.
    """
    detecciones = []

    for item in salida_ocr:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue

        bbox, texto_item, confianza = item
        texto_item = re.sub(r"[^A-Z0-9]", "", str(texto_item).upper())

        if not texto_item or texto_item in PALABRAS_RUIDO:
            continue

        try:
            centro_x = sum(float(p[0]) for p in bbox) / len(bbox)
            centro_y = sum(float(p[1]) for p in bbox) / len(bbox)
            alto = max(float(p[1]) for p in bbox) - min(float(p[1]) for p in bbox)
        except (IndexError, TypeError, ValueError):
            continue

        # Si tenemos la altura del recorte, descartar textos que esten en la parte inferior
        # y de baja altura (texto 'CHILE' o marcas del portapatente)
        if alto_img is not None and alto_img > 0:
            rel_y = centro_y / float(alto_img)
            rel_h = alto / float(alto_img)
            if rel_y > 0.68 and rel_h < 0.35:
                continue

        detecciones.append({
            "x": centro_x,
            "y": centro_y,
            "alto": max(alto, 1.0),
            "texto": texto_item,
            "confianza": float(confianza)
        })

    if not detecciones:
        return []

    detecciones.sort(key=lambda d: d["y"])
    lineas = []

    for deteccion in detecciones:
        linea_encontrada = None

        for linea in lineas:
            centro_linea = sum(d["y"] for d in linea) / len(linea)
            alto_medio = sum(d["alto"] for d in linea) / len(linea)

            if abs(deteccion["y"] - centro_linea) <= alto_medio * 0.6:
                linea_encontrada = linea
                break

        if linea_encontrada is None:
            lineas.append([deteccion])
        else:
            linea_encontrada.append(deteccion)

    lecturas = []

    for linea in lineas:
        linea.sort(key=lambda d: d["x"])
        texto = "".join(d["texto"] for d in linea)
        confianza = sum(d["confianza"] for d in linea) / len(linea)

        if texto:
            lecturas.append((texto, confianza))

    return lecturas


def leer_patente(imagen):
    """
    Prueba varias versiones del recorte y selecciona la lectura mas confiable.
    Descarta ruido inferior (palabra CHILE) y prioriza formatos validos de patente chilena.
    """
    if imagen is None or imagen.size == 0:
        return "", 0.0

    variantes = generar_variantes_ocr(imagen)
    candidatos = []

    for variante in variantes:
        salida_ocr = lector.readtext(
            variante,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            detail=1,
            paragraph=False
        )

        for texto, confianza in _obtener_lecturas_linea(salida_ocr, alto_img=variante.shape[0]):
            if confianza < 0.20:
                continue
            candidatos.append((texto, confianza))

    if not candidatos:
        return "", 0.0

    # Agrupar lecturas repetidas en distintas variantes
    conteo = {}
    for texto, confianza in candidatos:
        conteo.setdefault(texto, []).append(confianza)

    resultados = []
    for texto, confs in conteo.items():
        conf_max = max(confs)
        conf_avg = sum(confs) / len(confs)
        conf_base = conf_max * 0.7 + conf_avg * 0.3
        apoyo = min(len(confs) - 1, 3) * 0.05

        puntaje_final = conf_base + apoyo
        if len(texto) == 6:
            puntaje_final += 0.25
        elif len(texto) == 7 and (texto[4] in ("0", "O") or texto[2] in ("0", "O")):
            puntaje_final += 0.18
        elif len(texto) < 6:
            # Penalizar lecturas truncadas/incompletas
            puntaje_final -= 0.30

        resultados.append((puntaje_final, texto, conf_max))

    mejor = max(resultados, key=lambda item: item[0])
    return mejor[1], mejor[2]


def probar_con_foto():
    """
    Permite seleccionar una foto desde el PC, detecta la patente con best.pt,
    la recorta y la lee con EasyOCR.
    """
    ruta_modelo = CARPETA / "modelos" / "best.pt"
    if not ruta_modelo.is_file():
        print(f"Error: No se encontro el modelo en {ruta_modelo}")
        return

    ventana = tk.Tk()
    ventana.withdraw()
    try:
        archivo = filedialog.askopenfilename(
            parent=ventana,
            title="Selecciona una foto de un auto para probar el lector",
            filetypes=[
                ("Imagenes", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Todos los archivos", "*.*")
            ]
        )
    finally:
        ventana.destroy()

    if not archivo:
        print("Seleccion cancelada.")
        return

    print(f"\nArchivo seleccionado: {archivo}")
    imagen = cv2.imread(archivo)
    if imagen is None:
        print("Error: No se pudo cargar la imagen seleccionada.")
        return

    print("Cargando modelo YOLO (best.pt)...")
    modelo = YOLO(str(ruta_modelo))

    print("Detectando patente en la imagen...")
    predicciones = modelo.predict(source=imagen, conf=0.25, verbose=False)
    resultado = predicciones[0] if isinstance(predicciones, list) else next(iter(predicciones))

    cajas_xyxy = []
    confianzas_yolo = []

    boxes = getattr(resultado, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        cajas_xyxy = boxes.xyxy.tolist()
        confianzas_yolo = boxes.conf.tolist()

    imagen_anotada = imagen.copy()
    alto_img, ancho_img = imagen.shape[:2]

    if len(cajas_xyxy) == 0:
        print("\nYOLO no detecto patentes con umbral conf=0.25.")
        print("Intentando leer la imagen completa por si era un recorte directo...")
        texto, conf = leer_patente(imagen)
        if texto:
            texto_formato = formatear_patente(texto)
            print(f"-> Patente leida: {texto_formato} (Confianza OCR: {conf * 100:.1f}%)")
        else:
            print("-> No se pudo extraer texto alfanumerico.")
    else:
        print(f"\n¡Se encontraron {len(cajas_xyxy)} patente(s) detectada(s)!")
        for i, (coords, conf_yolo_raw) in enumerate(zip(cajas_xyxy, confianzas_yolo), start=1):
            x1, y1 = int(coords[0]), int(coords[1])
            x2, y2 = int(coords[2]), int(coords[3])
            conf_yolo = float(conf_yolo_raw)

            # Asegurar coordenadas dentro de la imagen
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(ancho_img, x2), min(alto_img, y2)

            recorte = imagen[y1:y2, x1:x2]
            texto, conf_ocr = leer_patente(recorte)

            print(f"\n--- Patente #{i} ---")
            print(f"Coordenadas: ({x1}, {y1}) a ({x2}, {y2})")
            print(f"Confianza YOLO: {conf_yolo * 100:.1f}%")
            if texto:
                texto_formato = formatear_patente(texto)
                print(f"Texto reconocido (OCR): {texto_formato}")
                print(f"Confianza promedio OCR: {conf_ocr * 100:.1f}%")
                etiqueta = f"{texto_formato} ({conf_ocr * 100:.0f}%)"
            else:
                print("Texto reconocido (OCR): No se reconocio texto legible.")
                etiqueta = f"Patente ({conf_yolo * 100:.0f}%)"

            # Dibujar recuadro y texto en la imagen anotada
            cv2.rectangle(imagen_anotada, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                imagen_anotada,
                etiqueta,
                (x1, max(y1 - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

    # Guardar copia con las anotaciones en runs/pruebas
    carpeta_runs = CARPETA / "runs" / "pruebas"
    carpeta_runs.mkdir(parents=True, exist_ok=True)
    ruta_guardado = carpeta_runs / f"resultado_{Path(archivo).name}"
    cv2.imwrite(str(ruta_guardado), imagen_anotada)
    print(f"\nResultado guardado en: {ruta_guardado}")

    # Mostrar ventana interactiva
    try:
        titulo_ventana = "Prueba de Lector - Presiona cualquier tecla para cerrar"
        cv2.namedWindow(titulo_ventana, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(titulo_ventana, 1024, 680)
        cv2.imshow(titulo_ventana, imagen_anotada)
        print("Mostrando ventana de resultado. Haz clic en la ventana y presiona cualquier tecla para cerrarla...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Aviso al mostrar ventana interactiva: {e}")


if __name__ == "__main__":
    print("Iniciando prueba interactiva con modelos/best.pt...")
    probar_con_foto()
