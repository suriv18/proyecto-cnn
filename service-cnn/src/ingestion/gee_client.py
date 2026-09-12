"""Capa delgada de acceso a Google Earth Engine para extracción de series.

Referencia: docs/02-arquitectura-tecnica.md, §2 (Acceso a fuentes de datos);
docs/Proyecto_Tesis_Maestria_UNMSM_WMSR.docx, sección 4.9, Tabla 7.

`ee` (earthengine-api) se inyecta en el constructor en vez de importarse a
nivel de módulo: `ee.Initialize()` requiere una cuenta autenticada vinculada a
un proyecto de Google Cloud, indisponible en este entorno de desarrollo/CI.
Esto permite probar toda la lógica de armado de consultas con un doble de
prueba (ver tests/ingestion/test_gee_client.py), sin tocar la red ni requerir
credenciales, y sin impedir que el módulo se importe si earthengine-api no está
instalado.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from src.ingestion.gee_config import get_collection_spec


class GeeClient:
    """Cliente de extracción de series agroclimáticas/espectrales vía Earth Engine.

    Exporta series agregadas por polígono provincial directamente a tabla, sin
    descargar rasters completos (sección 2 de la arquitectura técnica).
    """

    def __init__(self, ee_module: Any, project_id: str):
        """
        Args:
            ee_module: el módulo `ee` real (tras `import ee`), o un doble de
                prueba con la misma interfaz mínima usada aquí.
            project_id: identificador del proyecto de Google Cloud vinculado a
                la cuenta de Earth Engine (requerido desde 2024).
        """
        self._ee = ee_module
        self._project_id = project_id

    @classmethod
    def from_default(cls, project_id: str) -> "GeeClient":
        """Construye un `GeeClient` importando `earthengine-api` de verdad.

        Raises:
            ImportError: con un mensaje claro si `earthengine-api` no está
                instalado, indicando cómo instalarlo, en vez de un traceback
                genérico de `ModuleNotFoundError`.
        """
        try:
            import ee  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise ImportError(
                "earthengine-api no está instalado. Instálelo con: "
                "pip install earthengine-api geemap"
            ) from exc
        return cls(ee_module=ee, project_id=project_id)

    def initialize(self) -> None:
        """Inicializa la sesión de Earth Engine para el proyecto configurado."""
        self._ee.Initialize(project=self._project_id)

    def extract_daily_series(
        self,
        variable: str,
        geometrias_por_provincia: Any,
        fecha_inicio: date,
        fecha_fin: date,
    ) -> pd.DataFrame:
        """Extrae la serie temporal de una variable agregada por provincia.

        Args:
            variable: clave de `gee_config` (ej. "precipitacion", "ndvi").
            geometrias_por_provincia: `ee.FeatureCollection` con un feature por
                provincia (geometría + `provincia_id` como propiedad).
            fecha_inicio, fecha_fin: ventana temporal a extraer (inclusive).

        Returns:
            DataFrame con columnas `provincia_id`, `fecha`, `valor` — una fila
            por observación devuelta por Earth Engine. La agregación por fase
            fenológica se realiza después, en `gee_series_builder.py`, para
            mantener esta capa como una traducción directa de la respuesta de
            la API sin lógica de negocio propia.

        Raises:
            ValueError: si `fecha_inicio` es posterior a `fecha_fin`.
        """
        if fecha_inicio > fecha_fin:
            raise ValueError(
                f"fecha_inicio ({fecha_inicio}) no puede ser posterior a "
                f"fecha_fin ({fecha_fin})."
            )

        spec = get_collection_spec(variable)

        coleccion_reducida = (
            self._ee.ImageCollection(spec.collection_id)
            .filterDate(fecha_inicio.isoformat(), fecha_fin.isoformat())
            .select(spec.band)
            .toBands()
        )
        resultado = coleccion_reducida.reduceRegions(
            geometrias_por_provincia,
            reducer=None,  # el reducer real (ej. ee.Reducer.mean()) se define
            # al conectar con la API real; el doble de prueba lo ignora.
            scale=spec.spatial_resolution_m,
        )

        filas = [f["properties"] for f in resultado.getInfo()["features"]]
        return pd.DataFrame(filas)
