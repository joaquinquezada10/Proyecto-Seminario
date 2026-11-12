# 

import cv2
import easyocr
import re

lector = easyocr.Reader(["en"], gpu=False)

def preprocesar_imagen(imagen):
    """
    Prepara el recorte de la patente para facilitar su lectura.
    """

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    alto, ancho = gris.shape
    gris = cv2.resize(
        gris,
        (ancho * 3, alto * 3),
        interpolation=cv2.INTER_CUBIC
    )
    gris = cv2.equalizeHist(gris)

    return gris

def leer_patente(imagen):
    """
    Recibe el recorte de una patente y devuelve:
    - Texto reconocido
    - Nivel de confianza
    """

    if imagen is None or imagen.size == 0:
        return "", 0.0

    procesada = preprocesar_imagen(imagen)

    resultados = lector.readtext(
        procesada,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        detail=1,
        paragraph=False
    )

    if not resultados:
        return "", 0.0

    resultados.sort(key=lambda resultado: resultado[0][0][0])

    texto = ""
    confianzas = []

    for resultado in resultados:
        caracter = resultado[1].upper()
        confianza = resultado[2]

        caracter = re.sub(r"[^A-Z0-9]", "", caracter)

        texto += caracter
        confianzas.append(confianza)

    if not texto:
        return "", 0.0

    confianza_final = sum(confianzas) / len(confianzas)

    return texto, confianza_final