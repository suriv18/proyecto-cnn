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

from src.ingestion.gee_config import AggregationRule, get_collection_spec


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
            DataFrame con columnas `provincia_id`, `fecha`, y ya sea `valor`
            (si `spec.reducers` está vacío — comportamiento actual) o una
            columna `valor_<nombre_reducer>` por cada reducer en
            `spec.reducers` (ej. `valor_mean`, `valor_max` para NDVI, Tabla
            7: "Filtrado de calidad; media y máximo por fase sobre máscara
            agrícola"). `valor`/`valor_*` ya convertidos a su unidad real
            (`factor_escala`/`offset_aditivo` del spec — ej. NDVI
            entero->decimal, temperatura Kelvin->Celsius).

            Si `spec.banda_calidad` no es `None`, cada imagen se enmascara
            antes de reducir: solo se conservan píxeles con VI Quality
            (bits 0-1 de la banda) en {buena calidad, revisar otras QA}, sin
            nube adyacente (bit 8) ni nubes mixtas (bit 10) — verificado
            contra el catálogo oficial de `MODIS/061/MOD13Q1` y el MODIS VI
            User's Guide.

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

        bandas = [spec.band, spec.banda_calidad] if spec.banda_calidad else [spec.band]
        coleccion = (
            ee.ImageCollection(spec.collection_id)
            .filterDate(fecha_inicio.isoformat(), fecha_fin.isoformat())
            .select(bandas)
        )

        reducers = spec.reducers if spec.reducers else (spec.aggregation,)
        reducer_ee = _combinar_reducers(ee, reducers)

        def _reducir_imagen(imagen):
            fecha = imagen.date().format("YYYY-MM-dd")
            imagen_datos = imagen.select(spec.band)
            if spec.banda_calidad:
                qa = imagen.select(spec.banda_calidad)
                buena_calidad = _mascara_calidad_ndvi(ee, qa)
                imagen_datos = imagen_datos.updateMask(buena_calidad)
            reducido = imagen_datos.reduceRegions(
                collection=geometrias_por_provincia,
                reducer=reducer_ee,
                scale=spec.spatial_resolution_m,
            )
            return reducido.map(lambda f: f.set("fecha", fecha))

        resultado_aplanado = coleccion.map(_reducir_imagen).flatten()

        filas = [f["properties"] for f in resultado_aplanado.getInfo()["features"]]
        tabla = pd.DataFrame(filas)

        if len(reducers) == 1:
            if tabla.empty:
                return pd.DataFrame(columns=["provincia_id", "fecha", "valor"])
            nombre_reducer = reducers[0].value
            tabla = tabla.rename(columns={nombre_reducer: "valor"})[
                ["provincia_id", "fecha", "valor"]
            ]
            tabla["valor"] = tabla["valor"] * spec.factor_escala + spec.offset_aditivo
            return tabla

        columnas_salida = ["provincia_id", "fecha"]
        if tabla.empty:
            return pd.DataFrame(
                columns=columnas_salida + [f"valor_{r.value}" for r in reducers]
            )
        for regla in reducers:
            # ee.Reducer.combine(sharedInputs=True) antepone el nombre de la
            # banda de entrada al nombre del reducer en cada propiedad de
            # salida (ej. "NDVI_mean", "NDVI_max") — verificado contra la
            # guía oficial de Earth Engine (Reducer Overview: "the names of
            # the reducers have been appended to the names of the inputs").
            # Esto solo aplica al caso combinado (más de un reducer); con
            # un solo reducer (rama de arriba) reduceRegions expone el
            # nombre del reducer sin prefijo, ya verificado en
            # gee_mask_pipeline.py.
            columna_origen = f"{spec.band}_{regla.value}"
            columna_salida = f"valor_{regla.value}"
            tabla[columna_salida] = (
                tabla[columna_origen] * spec.factor_escala + spec.offset_aditivo
            )
            columnas_salida.append(columna_salida)
        return tabla[columnas_salida]


def _combinar_reducers(ee_module: Any, reducers: tuple) -> Any:
    """Combina uno o más `AggregationRule` en un único `ee.Reducer`.

    Con un solo reducer, lo construye directamente (`ee.Reducer.mean()`,
    etc.). Con varios (hoy solo NDVI: MEAN y MAX), los combina con
    `.combine()` para que `reduceRegions` los calcule en una sola pasada
    sobre la misma imagen ya enmascarada, en vez de leerla dos veces.

    Efecto en el nombre de las propiedades de salida (verificado contra la
    guía oficial de Earth Engine, "Reducer Overview"): con `sharedInputs=
    True`, cada propiedad de `reduceRegions` queda nombrada
    `<banda>_<reducer>` (ej. `NDVI_mean`, `NDVI_max`), no solo `<reducer>` a
    secas — el llamador (`extract_daily_series`) debe leer
    `f"{spec.band}_{regla.value}"`, no `regla.value`, al desempaquetar el
    resultado combinado.
    """
    _CONSTRUCTOR = {
        AggregationRule.SUM: ee_module.Reducer.sum,
        AggregationRule.MEAN: ee_module.Reducer.mean,
        AggregationRule.MAX: ee_module.Reducer.max,
        AggregationRule.MIN: ee_module.Reducer.min,
    }
    reducer_actual = _CONSTRUCTOR[reducers[0]]()
    for siguiente in reducers[1:]:
        reducer_actual = reducer_actual.combine(
            reducer2=_CONSTRUCTOR[siguiente](), sharedInputs=True
        )
    return reducer_actual


def _mascara_calidad_ndvi(ee_module: Any, banda_qa: Any) -> Any:
    """Máscara booleana de calidad para MOD13Q1 `DetailedQA`.

    Verificado contra el catálogo oficial de Earth Engine
    (MODIS/061/MOD13Q1) y el MODIS VI User's Guide:
    - Bits 0-1 (VI Quality/MODLAND_QA): 00=buena calidad, 01=revisar otras
      QA, 10=probablemente nublado, 11=no producido. Se aceptan 00 y 01
      (equivalente a `vi_quality <= 1`).
    - Bit 8 (nube adyacente detectada): se acepta solo 0.
    - Bit 10 (nubes mixtas posibles): se acepta solo 0.
    """
    vi_quality = banda_qa.rightShift(0).bitwiseAnd(0b11)
    nube_adyacente = banda_qa.rightShift(8).bitwiseAnd(0b1)
    nubes_mixtas = banda_qa.rightShift(10).bitwiseAnd(0b1)
    return vi_quality.lte(1).And(nube_adyacente.eq(0)).And(nubes_mixtas.eq(0))
