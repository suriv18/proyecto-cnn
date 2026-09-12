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
    """Especificación de una colección de Earth Engine para una variable."""

    collection_id: str
    band: str
    spatial_resolution_m: int
    temporal_resolution: str
    aggregation: AggregationRule


_SPECS: dict[str, GeeCollectionSpec] = {
    "precipitacion": GeeCollectionSpec(
        collection_id="UCSB-CHG/CHIRPS/DAILY",
        band="precipitation",
        spatial_resolution_m=5566,  # ~0.05° en el ecuador
        temporal_resolution="diaria",
        # Tabla 7: "Suma por fase y racha máxima de días con precipitación < 1 mm"
        aggregation=AggregationRule.SUM,
    ),
    "temperatura_maxima": GeeCollectionSpec(
        collection_id="ECMWF/ERA5_LAND/HOURLY",
        band="temperature_2m_max",
        spatial_resolution_m=11132,  # ~0.1°
        temporal_resolution="horaria",
        aggregation=AggregationRule.MAX,
    ),
    "temperatura_minima": GeeCollectionSpec(
        collection_id="ECMWF/ERA5_LAND/HOURLY",
        band="temperature_2m_min",
        spatial_resolution_m=11132,
        temporal_resolution="horaria",
        aggregation=AggregationRule.MIN,
    ),
    "radiacion_solar": GeeCollectionSpec(
        collection_id="ECMWF/ERA5_LAND/HOURLY",
        band="surface_solar_radiation_downwards",
        spatial_resolution_m=11132,
        temporal_resolution="horaria",
        # Tabla 7: "Acumulado o media diaria por fase"
        aggregation=AggregationRule.SUM,
    ),
    "ndvi": GeeCollectionSpec(
        collection_id="MODIS/061/MOD13Q1",
        band="NDVI",
        spatial_resolution_m=250,
        temporal_resolution="16 dias",
        # Tabla 7: "media y máximo por fase sobre máscara agrícola" — se modela
        # la media aquí; el máximo se calcula aparte donde se necesite (B3).
        aggregation=AggregationRule.MEAN,
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
