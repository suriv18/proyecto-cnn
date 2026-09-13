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

    @property
    def ee_module(self) -> Any:
        """El módulo `ee` (o doble de prueba) inyectado en el constructor.

        Expuesto para que otros módulos (ej. `province_geometries.
        build_province_feature_collection`) reutilicen la misma sesión de
        Earth Engine ya inicializada, en vez de importar `ee` por su cuenta.
        """
        return self._ee

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
            por observación devuelta por Earth Engine, con `valor` ya
            convertido a su unidad real (`factor_escala`/`offset_aditivo` del
            spec — ej. NDVI entero->decimal, temperatura Kelvin->Celsius).
            La agregación por fase fenológica se realiza después, en
            `gee_series_builder.py`, para mantener esta capa como una
            traducción directa de la respuesta de la API sin lógica de
            negocio agronómica propia.

        Raises:
            ValueError: si `fecha_inicio` es posterior a `fecha_fin`.

        Nota de implementación (verificado contra la API real, proyecto
        cnn-sentinel): el enfoque `.toBands().reduceRegions(reducer=None)`
        falla contra la API real con "Parameter 'reducer' is required and may
        not be null" y además produce estructura ancha (una columna por
        fecha). El enfoque correcto reduce cada imagen por separado con
        `ee.Reducer.mean()` explícito, le agrega la fecha como propiedad, y
        aplana el resultado con `.flatten()` — así se obtiene directamente el
        formato largo documentado arriba.
        """
        if fecha_inicio > fecha_fin:
            raise ValueError(
                f"fecha_inicio ({fecha_inicio}) no puede ser posterior a "
                f"fecha_fin ({fecha_fin})."
            )

        spec = get_collection_spec(variable)
        ee = self._ee

        coleccion = (
            ee.ImageCollection(spec.collection_id)
            .filterDate(fecha_inicio.isoformat(), fecha_fin.isoformat())
            .select(spec.band)
        )

        def _reducir_imagen(imagen):
            fecha = imagen.date().format("YYYY-MM-dd")
            reducido = imagen.reduceRegions(
                collection=geometrias_por_provincia,
                reducer=ee.Reducer.mean(),
                scale=spec.spatial_resolution_m,
            )
            return reducido.map(lambda f: f.set("fecha", fecha))

        resultado_aplanado = coleccion.map(_reducir_imagen).flatten()

        filas = [f["properties"] for f in resultado_aplanado.getInfo()["features"]]
        tabla = pd.DataFrame(filas)
        if tabla.empty:
            return pd.DataFrame(columns=["provincia_id", "fecha", "valor"])
        tabla = tabla.rename(columns={"mean": "valor"})[["provincia_id", "fecha", "valor"]]
        tabla["valor"] = tabla["valor"] * spec.factor_escala + spec.offset_aditivo
        return tabla
