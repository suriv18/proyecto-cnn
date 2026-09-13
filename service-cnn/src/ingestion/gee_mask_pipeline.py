"""Máscara de superficie agrícola server-side en Google Earth Engine.

Referencia: service-cnn/docs/superpowers/specs/2026-09-13-mascara-agricola-gee-design.md

Reimplementa los 4 criterios de `src.preprocessing.mask.apply_agricultural_mask`
como una `ee.Image` booleana (cobertura de suelo + altitud + pendiente +
amplitud NDVI), evaluada server-side antes de `reduceRegions` — necesario
porque a resolución de celda (NDVI 250 m, SRTM ~30 m) sobre 80 provincias el
volumen de píxeles excede ampliamente el límite de 5000 elementos de
`ee.FeatureCollection.getInfo()` (ya documentado en gee_extraction_pipeline.py),
lo que hace inviable traer celdas individuales a pandas para el flujo de
producción. `apply_agricultural_mask` (pandas) se conserva sin cambios para
calibración y auditoría con una muestra acotada de celdas (ver
`sample_cells_for_calibration`).

No importa `ee` a nivel de módulo: recibe el módulo `ee` (real o doble de
prueba) inyectado, mismo patrón que `gee_client.py`.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from src.ingestion.gee_config import get_collection_spec
from src.preprocessing.mask import MaskThresholds

_WORLDCOVER_ID = "ESA/WorldCover/v200/2021"
_SRTM_ID = "USGS/SRTMGL1_003"
_CLASE_WORLDCOVER_CROPLAND = 40


def build_agricultural_mask_image(
    ee_module: Any,
    umbrales: MaskThresholds,
    fecha_inicio: date,
    fecha_fin: date,
) -> Any:
    """Construye la máscara agrícola booleana combinando los 4 criterios.

    Args:
        ee_module: el módulo `ee` real, ya inicializado, o un doble de
            prueba con la misma interfaz mínima usada aquí.
        umbrales: `MaskThresholds` ya calibrados (mismo dataclass que usa
            `apply_agricultural_mask` en pandas).
        fecha_inicio, fecha_fin: ventana de extracción de la campaña
            (ver `gee_extraction_pipeline.campana_a_ventana_extraccion`) —
            la amplitud NDVI se calcula sobre esta ventana específica, no
            sobre un año fijo, porque el ciclo agrícola cambia campaña a
            campaña.

    Returns:
        Una `ee.Image` de una banda booleana: `True` solo en las celdas que
        cumplen los 4 criterios (cobertura agrícola, altitud en rango,
        pendiente aceptable, amplitud NDVI suficiente).
    """
    cobertura = ee_module.Image(_WORLDCOVER_ID).select("Map")
    es_cultivo = cobertura.eq(_CLASE_WORLDCOVER_CROPLAND)

    elevacion = ee_module.Image(_SRTM_ID).select("elevation")
    altitud_en_rango = elevacion.gte(umbrales.altitud_min_m).And(
        elevacion.lte(umbrales.altitud_max_m)
    )

    pendiente = ee_module.Terrain.slope(elevacion)
    pendiente_aceptable = pendiente.lte(umbrales.pendiente_max_grados)

    spec_ndvi = get_collection_spec("ndvi")
    coleccion_ndvi = (
        ee_module.ImageCollection(spec_ndvi.collection_id)
        .filterDate(fecha_inicio, fecha_fin)
        .select(spec_ndvi.band)
    )
    amplitud_cruda = coleccion_ndvi.max().subtract(coleccion_ndvi.min())
    amplitud = amplitud_cruda.multiply(spec_ndvi.factor_escala)
    amplitud_suficiente = amplitud.gte(umbrales.amplitud_ndvi_min)

    return (
        es_cultivo.And(altitud_en_rango)
        .And(pendiente_aceptable)
        .And(amplitud_suficiente)
    )


def extract_masked_ndvi_series(
    cliente: Any,
    geometrias_por_provincia: Any,
    umbrales: MaskThresholds,
    fecha_inicio: date,
    fecha_fin: date,
) -> "pd.DataFrame":
    """Extrae NDVI agregado por provincia, calculado solo sobre celdas agrícolas.

    Args:
        cliente: `GeeClient` ya inicializado (real o doble de prueba) — se
            usa `cliente.ee_module` para acceder a la sesión de Earth Engine
            ya inicializada, mismo patrón que `province_geometries.
            build_province_feature_collection`.
        geometrias_por_provincia: `ee.FeatureCollection` con un feature por
            provincia (ver `province_geometries.py`).
        umbrales: `MaskThresholds` ya calibrados.
        fecha_inicio, fecha_fin: ventana de extracción de la campaña.

    Returns:
        DataFrame con columnas `provincia_id`, `fecha`, `valor` — mismo
        contrato que `GeeClient.extract_daily_series`, con `valor` ya
        convertido a NDVI real ([-1, 1]) vía el `factor_escala` del spec.
    """
    ee = cliente.ee_module
    mascara = build_agricultural_mask_image(ee, umbrales, fecha_inicio, fecha_fin)

    spec = get_collection_spec("ndvi")
    coleccion = (
        ee.ImageCollection(spec.collection_id)
        .filterDate(fecha_inicio.isoformat(), fecha_fin.isoformat())
        .select(spec.band)
    )

    def _reducir_imagen_enmascarada(imagen):
        fecha = imagen.date().format("YYYY-MM-dd")
        imagen_enmascarada = imagen.updateMask(mascara)
        reducido = imagen_enmascarada.reduceRegions(
            collection=geometrias_por_provincia,
            reducer=ee.Reducer.mean(),
            scale=spec.spatial_resolution_m,
        )
        return reducido.map(lambda f: f.set("fecha", fecha))

    resultado_aplanado = coleccion.map(_reducir_imagen_enmascarada).flatten()

    filas = [f["properties"] for f in resultado_aplanado.getInfo()["features"]]
    tabla = pd.DataFrame(filas)
    if tabla.empty:
        return pd.DataFrame(columns=["provincia_id", "fecha", "valor"])
    tabla = tabla.rename(columns={"mean": "valor"})[["provincia_id", "fecha", "valor"]]
    tabla["valor"] = tabla["valor"] * spec.factor_escala + spec.offset_aditivo
    return tabla
