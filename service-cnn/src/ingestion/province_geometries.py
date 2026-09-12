"""Mapeo de provincias delimitadas a sus geometrías reales (FAO/GAUL/2015/level2).

Referencia: docs/02-arquitectura-tecnica.md §2. Las 80 provincias de
configs/provincias.csv (nombres en MAYÚSCULAS sin tildes, convención de
MIDAGRI) se mapean a sus polígonos reales en el dataset público
FAO/GAUL/2015/level2 de Google Earth Engine (nombres en Título Case, con
tildes) — verificado que ambas fuentes listan exactamente las mismas 80
provincias para los 8 departamentos delimitados por la tesis.

`normalize_province_name` es lógica pura, testeable sin credenciales de GEE;
`build_province_feature_collection` sí requiere una sesión de Earth Engine
inicializada (recibe el módulo `ee` inyectado, mismo patrón que
`gee_client.py`).
"""
from __future__ import annotations

import unicodedata

_DATASET_GAUL_NIVEL_PROVINCIA = "FAO/GAUL/2015/level2"
_PAIS = "Peru"


def normalize_province_name(nombre: str) -> str:
    """Normaliza un nombre de provincia para comparación robusta entre fuentes.

    Quita tildes, colapsa espacios múltiples y convierte a mayúsculas — así
    "Sánchez Carrión" (GAUL) y "SANCHEZ CARRION" (MIDAGRI) comparan iguales.
    """
    sin_tildes = "".join(
        c for c in unicodedata.normalize("NFKD", nombre) if not unicodedata.combining(c)
    )
    return " ".join(sin_tildes.upper().split())


def build_province_feature_collection(
    ee_module,
    provincia_ids: list[str],
    departamentos_gaul: list[str],
):
    """Construye una `ee.FeatureCollection` con una geometría por provincia.

    Args:
        ee_module: el módulo `ee` real, ya inicializado (`ee.Initialize()`
            ejecutado previamente por el llamador).
        provincia_ids: lista de `provincia_id` (convención MIDAGRI,
            MAYÚSCULAS sin tildes) a incluir — normalmente las 80 filas de
            configs/provincias.csv.
        departamentos_gaul: nombres de los 8 departamentos delimitados en el
            formato usado por GAUL (ej. "Puno", "Apurímac", "La Libertad"),
            para acotar el filtro antes de comparar nombres de provincia.

    Returns:
        `ee.FeatureCollection` con un feature por cada `provincia_id`
        encontrado, con la propiedad `provincia_id` agregada (para que
        `GeeClient.extract_daily_series` pueda identificar cada polígono en
        el resultado de `reduceRegions`). Las provincias de
        `provincia_ids` que no se encuentren en GAUL se omiten silenciosamente
        — el llamador debe verificar la cuenta si el tamaño no coincide con
        lo esperado.
    """
    gaul = ee_module.FeatureCollection(_DATASET_GAUL_NIVEL_PROVINCIA)
    peru = gaul.filter(ee_module.Filter.eq("ADM0_NAME", _PAIS))
    subset = peru.filter(ee_module.Filter.inList("ADM1_NAME", departamentos_gaul))

    ids_normalizados = {normalize_province_name(pid): pid for pid in provincia_ids}

    info = subset.getInfo()
    features_con_id = []
    for feature in info["features"]:
        nombre_gaul = feature["properties"]["ADM2_NAME"]
        clave = normalize_province_name(nombre_gaul)
        if clave in ids_normalizados:
            propiedades = dict(feature["properties"])
            propiedades["provincia_id"] = ids_normalizados[clave]
            features_con_id.append(
                ee_module.Feature(feature["geometry"], propiedades)
            )

    return ee_module.FeatureCollection(features_con_id)
