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
    amplitud = coleccion_ndvi.max().subtract(coleccion_ndvi.min())
    amplitud_suficiente = amplitud.gte(umbrales.amplitud_ndvi_min)

    return (
        es_cultivo.And(altitud_en_rango)
        .And(pendiente_aceptable)
        .And(amplitud_suficiente)
    )
