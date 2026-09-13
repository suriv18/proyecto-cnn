"""Configuración de colecciones de Google Earth Engine por variable (Tabla 7).

Referencia: docs/Proyecto_Tesis_Maestria_UNMSM_WMSR.docx, sección 4.9, Tabla 7
("Fuentes de datos, resolución y regla de agregación").

Este módulo NO importa `ee`/`geemap`: es configuración pura, testeable sin
credenciales de Google Earth Engine. La capa que sí ejecuta consultas reales
contra la API vive en `gee_client.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AggregationRule(str, Enum):
    """Regla de agregación temporal por fase fenológica (Tabla 7)."""

    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"


@dataclass(frozen=True)
class GeeCollectionSpec:
    """Especificación de una colección de Earth Engine para una variable.

    `factor_escala` y `offset_aditivo` convierten el valor crudo devuelto por
    Earth Engine a la unidad real de la variable: `valor_real = crudo *
    factor_escala + offset_aditivo`. Por defecto no aplican conversión (1.0 y
    0.0) — solo NDVI (factor de escala entero→decimal) y las variables de
    ERA5-Land (Kelvin→Celsius) los necesitan, verificado contra el catálogo
    oficial MODIS/ECMWF.

    `banda_calidad` y `reducers` cubren el caso NDVI (Tabla 7: "Filtrado de
    calidad; media y máximo por fase sobre máscara agrícola"): si
    `banda_calidad` no es `None`, `extract_daily_series` filtra por esa banda
    antes de reducir; si `reducers` no está vacío, se calculan TODOS esos
    reducers (en vez de uno solo, `aggregation`), devolviendo una columna por
    reducer. Ambos son `None`/vacío por defecto — las variables sin banda de
    calidad propia (CHIRPS, ERA5-Land) no cambian de comportamiento.
    """

    collection_id: str
    band: str
    spatial_resolution_m: int
    temporal_resolution: str
    aggregation: AggregationRule
    factor_escala: float = 1.0
    offset_aditivo: float = 0.0
    banda_calidad: str | None = None
    reducers: tuple[AggregationRule, ...] = ()


_SPECS: dict[str, GeeCollectionSpec] = {
    "precipitacion": GeeCollectionSpec(
        collection_id="UCSB-CHG/CHIRPS/DAILY",
        band="precipitation",
        spatial_resolution_m=5566,  # ~0.05° en el ecuador
        temporal_resolution="diaria",
        # Tabla 7: "Suma por fase y racha máxima de días con precipitación < 1 mm"
        aggregation=AggregationRule.SUM,
    ),
    # ECMWF/ERA5_LAND/HOURLY no expone bandas _max/_min/_sum ya agregadas
    # (solo valores horarios instantáneos, ej. temperature_2m); se usa la
    # variante DAILY_AGGR, que agrega nativamente a un valor por día — una
    # imagen por fecha, consistente con extract_daily_series(). Verificado
    # contra la API real (proyecto cnn-sentinel): ambas bandas existen ahí.
    "temperatura_maxima": GeeCollectionSpec(
        collection_id="ECMWF/ERA5_LAND/DAILY_AGGR",
        band="temperature_2m_max",
        spatial_resolution_m=11132,  # ~0.1°
        temporal_resolution="diaria",
        aggregation=AggregationRule.MAX,
        offset_aditivo=-273.15,  # Kelvin -> Celsius
    ),
    "temperatura_minima": GeeCollectionSpec(
        collection_id="ECMWF/ERA5_LAND/DAILY_AGGR",
        band="temperature_2m_min",
        spatial_resolution_m=11132,
        temporal_resolution="diaria",
        aggregation=AggregationRule.MIN,
        offset_aditivo=-273.15,
    ),
    "radiacion_solar": GeeCollectionSpec(
        collection_id="ECMWF/ERA5_LAND/DAILY_AGGR",
        band="surface_solar_radiation_downwards_sum",
        spatial_resolution_m=11132,
        temporal_resolution="diaria",
        # Tabla 7: "Acumulado o media diaria por fase"
        aggregation=AggregationRule.SUM,
    ),
    "ndvi": GeeCollectionSpec(
        collection_id="MODIS/061/MOD13Q1",
        band="NDVI",
        spatial_resolution_m=250,
        temporal_resolution="16 dias",
        # Tabla 7: "media y máximo por fase sobre máscara agrícola" — ambos
        # reducers se calculan (ver banda_calidad/reducers abajo); aggregation
        # se conserva como el reducer "principal" para compatibilidad con
        # llamadores que aún no distinguen entre columnas.
        aggregation=AggregationRule.MEAN,
        # MOD13Q1 codifica NDVI como entero [-2000, 10000] (catálogo oficial
        # MODIS/USGS): valor_real = crudo * 0.0001.
        factor_escala=0.0001,
        # DetailedQA (16 bits, catálogo oficial MODIS/061/MOD13Q1 + MODIS VI
        # User's Guide): bits 0-1 = VI Quality (00=buena, 01=revisar otras
        # QA, 10=nublado, 11=no producido); bit 8 = nube adyacente; bit 10 =
        # nubes mixtas posibles. Se filtra en extract_daily_series.
        banda_calidad="DetailedQA",
        reducers=(AggregationRule.MEAN, AggregationRule.MAX),
    ),
}


def get_collection_spec(variable: str) -> GeeCollectionSpec:
    """Devuelve la especificación de colección/banda/agregación de una variable.

    Raises:
        KeyError: si `variable` no está en el catálogo de la Tabla 7 — señal de
            que se pidió una variable no contemplada en el diseño de la tesis,
            preferible a devolver una configuración inventada.
    """
    try:
        return _SPECS[variable]
    except KeyError as exc:
        raise KeyError(
            f"Variable '{variable}' no está definida en gee_config. "
            f"Variables disponibles: {sorted(_SPECS)}"
        ) from exc
