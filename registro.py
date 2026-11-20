# Consulta si la patente esta en el CSV de la empresa

import csv
from pathlib import Path
import re
from typing import Any, Dict, Optional, Tuple

CARPETA = Path(__file__).resolve().parent
RUTA_CSV_DEFECTO = CARPETA / "patentes_autorizadas.csv"

# Cache en memoria para evitar leer el disco en cada consulta
_cache_registro: Optional[Dict[str, Dict[str, Any]]] = None

# Confusiones comunes entre letras y numeros que comete EasyOCR
CONFUSIONES_OCR = {
    "0": {"O", "D", "Q", "U"},
    "O": {"0", "D", "Q", "U"},
    "1": {"I", "L", "T"},
    "I": {"1", "L", "T"},
    "8": {"B", "3"},
    "B": {"8", "3"},
    "3": {"8", "B"},
    "5": {"S"},
    "S": {"5"},
    "2": {"Z"},
    "Z": {"2"},
}


def limpiar_patente(texto: str) -> str:
    """
    Normaliza el texto de una patente:
    - Convierte a mayusculas.
    - Elimina espacios, guiones, puntos y cualquier simbolo no alfanumerico.
    Ejemplo: 'pt kx-93' -> 'PTKX93'
    """
    if not texto:
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(texto).upper().strip())


def cargar_registro(ruta_csv: Optional[Path] = None, forzar_recarga: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Carga el archivo CSV en un diccionario indexado por la patente limpia.
    Si el usuario agrega nuevas columnas al CSV en el futuro, se capturan automaticamente.
    """
    global _cache_registro

    if _cache_registro is not None and not forzar_recarga and ruta_csv is None:
        return _cache_registro

    archivo = ruta_csv if ruta_csv is not None else RUTA_CSV_DEFECTO

    if not archivo.is_file():
        print(f"Aviso: No se encontro el archivo de registro en {archivo}")
        return {}

    registro: Dict[str, Dict[str, Any]] = {}

    # Soporte para codificacion UTF-8 y UTF-8 con BOM (Excel)
    encodings = ["utf-8-sig", "utf-8", "latin-1"]
    contenido_leido = False

    for codificacion in encodings:
        try:
            with open(archivo, mode="r", encoding=codificacion) as f:
                lector = csv.DictReader(f)
                for fila in lector:
                    if not fila:
                        continue

                    # Normalizar claves de encabezado (minusculas y sin espacios)
                    fila_limpia = {
                        str(k).strip().lower(): str(v).strip()
                        for k, v in fila.items()
                        if k is not None
                    }

                    col_patente = fila_limpia.get("patente", "")
                    patente_key = limpiar_patente(col_patente)

                    if patente_key:
                        registro[patente_key] = fila_limpia

            contenido_leido = True
            break
        except UnicodeDecodeError:
            continue

    if not contenido_leido:
        print(f"Error: No se pudo leer el archivo CSV {archivo} con las codificaciones habituales.")
        return {}

    if ruta_csv is None:
        _cache_registro = registro

    return registro


def _buscar_patente_similar(patente_limpia: str, registro: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """
    Busca una posible patente similar por confusiones comunes del OCR.
    Esta coincidencia solo sirve como referencia y nunca autoriza el acceso.
    """
    if len(patente_limpia) != 6:
        return None

    for patente_registrada in registro.keys():
        if len(patente_registrada) != 6:
            continue

        diferencias = 0
        for c_leido, c_reg in zip(patente_limpia, patente_registrada):
            if c_leido != c_reg:
                if c_reg in CONFUSIONES_OCR.get(c_leido, set()):
                    diferencias += 1
                else:
                    diferencias += 2

        if diferencias <= 1:
            return patente_registrada

    return None


def verificar_patente(patente: str, ruta_csv: Optional[Path] = None) -> Tuple[bool, Dict[str, Any]]:
    """
    Verifica si una patente esta autorizada en el registro de la empresa.

    Solo una coincidencia exacta puede autorizar el acceso.
    Una coincidencia aproximada se informa para revision, pero se rechaza
    para evitar permitir el ingreso por una lectura OCR incorrecta.
    """
    patente_limpia = limpiar_patente(patente)
    if not patente_limpia:
        return False, {
            "patente_consultada": patente,
            "patente_registrada": "",
            "conductor": "No especificado",
            "estado": "NO AUTORIZADO",
            "motivo": "Patente vacia o invalida",
            "coincidencia": "INVALIDA"
        }

    registro = cargar_registro(ruta_csv)

    # 1. Busqueda exacta
    if patente_limpia in registro:
        datos = registro[patente_limpia].copy()
        estado = datos.get("estado", "AUTORIZADO").upper()
        es_autorizado = estado == "AUTORIZADO"

        return es_autorizado, {
            "patente_consultada": patente_limpia,
            "patente_registrada": patente_limpia,
            "conductor": datos.get("conductor", "Sin asignar"),
            "estado": estado,
            "datos_completos": datos,
            "coincidencia": "EXACTA"
        }

    # 2. Coincidencia aproximada: solo se entrega como sugerencia
    patente_similar = _buscar_patente_similar(patente_limpia, registro)
    if patente_similar is not None:
        datos = registro[patente_similar].copy()
        return False, {
            "patente_consultada": patente_limpia,
            "patente_registrada": patente_similar,
            "conductor": datos.get("conductor", "Sin asignar"),
            "estado": "REVISAR",
            "datos_completos": datos,
            "coincidencia": "APROXIMADA_REVISAR",
            "motivo": "La lectura OCR no coincide exactamente. Se requiere revision manual."
        }

    # 3. No encontrada
    return False, {
        "patente_consultada": patente_limpia,
        "patente_registrada": "",
        "conductor": "Desconocido",
        "estado": "NO AUTORIZADO",
        "datos_completos": {},
        "coincidencia": "NO_ENCONTRADA"
    }


if __name__ == "__main__":
    print("=== Prueba del modulo registro.py ===")
    datos_cargados = cargar_registro(forzar_recarga=True)
    print(f"Patentes cargadas en registro: {len(datos_cargados)}\n")

    pruebas = [
        "PTKX93",      # Coincidencia exacta
        "PT KX-93",   # Coincidencia exacta con formato
        "PTKX9B",      # Posible confusion OCR: requiere revision manual
        "BBCL12",     # Otra patente registrada
        "XXYY99"      # Patente no registrada
    ]

    for p in pruebas:
        autorizado, detalle = verificar_patente(p)
        print(f"Consulta: '{p}' -> Autorizado: {autorizado}")
        print(f"   Conductor: {detalle['conductor']} | Estado: {detalle['estado']} | Tipo: {detalle['coincidencia']}")
        print("-" * 50)
