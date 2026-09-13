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
        .filterDate(fecha_inicio.isoformat(), fecha_fin.isoformat())
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
    ventana_mascara: tuple[date, date],
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
        fecha_inicio, fecha_fin: ventana de extracción de ESTA llamada (un
            mes, ver `gee_extraction_pipeline.dividir_en_meses`) — sobre
            esta ventana se reduce el NDVI que se agrega por provincia.
        ventana_mascara: `(fecha_inicio, fecha_fin)` de la CAMPAÑA completa
            (ver `gee_extraction_pipeline.campana_a_ventana_extraccion`),
            distinta de `fecha_inicio`/`fecha_fin` — la amplitud NDVI de la
            máscara debe calcularse sobre el ciclo agrícola completo, no
            sobre el trozo mensual en el que se pagina esta llamada
            (`extract_all_provinces_all_variables` invoca esta función una
            vez por mes dentro de la misma campaña; recalcular la máscara
            con la ventana mensual la haría variar mes a mes dentro de una
            misma campaña y desalinearía el umbral, calibrado sobre la
            ventana anual completa en `sample_cells_for_calibration`).

    Returns:
        DataFrame con columnas `provincia_id`, `fecha`, `valor` — mismo
        contrato que `GeeClient.extract_daily_series`, con `valor` ya
        convertido a NDVI real ([-1, 1]) vía el `factor_escala` del spec.
    """
    ee = cliente.ee_module
    fecha_inicio_mascara, fecha_fin_mascara = ventana_mascara
    mascara = build_agricultural_mask_image(
        ee, umbrales, fecha_inicio_mascara, fecha_fin_mascara
    )

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


def sample_cells_for_calibration(
    ee_module: Any,
    geometrias_por_provincia: Any,
    provincia_ids: list[str],
    fecha_inicio: date,
    fecha_fin: date,
    escala_muestreo_m: int,
    n_por_provincia: int,
) -> "pd.DataFrame":
    """Exporta una muestra acotada de celdas para calibrar/auditar la máscara.

    Uso: alimentar `src.preprocessing.mask.apply_agricultural_mask` (para
    auditar cuántas celdas caen en cada motivo de exclusión) y
    `calibrate_ndvi_amplitude_threshold` (para fijar `amplitud_ndvi_min`
    sobre campañas de entrenamiento) — nunca para el flujo de producción,
    que usa `extract_masked_ndvi_series` (server-side, sin traer celdas).

    Args:
        ee_module: el módulo `ee` real, ya inicializado, o un doble de
            prueba.
        geometrias_por_provincia: `ee.FeatureCollection` con un feature por
            provincia.
        provincia_ids: provincias incluidas en `geometrias_por_provincia` —
            usado solo para validar el tamaño total de la muestra contra el
            límite de `.getInfo()`.
        fecha_inicio, fecha_fin: ventana de extracción de la campaña (debe
            restringirse a campañas de ENTRENAMIENTO cuando el resultado se
            use para calibrar `amplitud_ndvi_min` — responsabilidad del
            llamador, misma disciplina anti-fuga que
            `calibrate_ndvi_amplitude_threshold`).
        escala_muestreo_m: resolución de muestreo en metros, pasada a
            `sampleRegions`.
        n_por_provincia: máximo de celdas a muestrear por provincia.

    Returns:
        DataFrame con columnas `celda_id`, `cobertura_suelo` (`"cultivo"` o
        `"no_cultivo"`), `altitud_m`, `pendiente_grados`, `amplitud_ndvi` —
        exactamente el esquema que espera `apply_agricultural_mask`.

    Raises:
        ValueError: si `n_por_provincia * len(provincia_ids)` alcanza o
            excede 5000 — límite real de `ee.FeatureCollection.getInfo()`
            (`Collection query aborted after accumulating over 5000
            elements`, ya documentado en `gee_extraction_pipeline.
            dividir_en_meses`).
    """
    tamano_muestra_total = n_por_provincia * len(provincia_ids)
    if tamano_muestra_total >= 5000:
        raise ValueError(
            f"n_por_provincia ({n_por_provincia}) x provincias "
            f"({len(provincia_ids)}) = {tamano_muestra_total} alcanza o "
            "excede el límite de 5000 elementos de "
            "ee.FeatureCollection.getInfo(). Reduzca n_por_provincia."
        )

    cobertura = ee_module.Image(_WORLDCOVER_ID).select("Map").rename(["cobertura_suelo"])
    elevacion = ee_module.Image(_SRTM_ID).select("elevation").rename(["altitud_m"])
    pendiente = ee_module.Terrain.slope(elevacion).rename(["pendiente_grados"])

    spec_ndvi = get_collection_spec("ndvi")
    coleccion_ndvi = (
        ee_module.ImageCollection(spec_ndvi.collection_id)
        .filterDate(fecha_inicio.isoformat(), fecha_fin.isoformat())
        .select(spec_ndvi.band)
    )
    amplitud = (
        coleccion_ndvi.max()
        .subtract(coleccion_ndvi.min())
        .multiply(spec_ndvi.factor_escala)
        .rename(["amplitud_ndvi"])
    )

    imagen_compuesta = (
        cobertura.addBands(elevacion).addBands(pendiente).addBands(amplitud)
    )

    muestra = imagen_compuesta.sampleRegions(
        collection=geometrias_por_provincia,
        scale=escala_muestreo_m,
        geometries=False,
    )

    filas = [f["properties"] for f in muestra.getInfo()["features"]]
    tabla = pd.DataFrame(filas)
    if tabla.empty:
        return pd.DataFrame(
            columns=[
                "celda_id",
                "cobertura_suelo",
                "altitud_m",
                "pendiente_grados",
                "amplitud_ndvi",
            ]
        )

    tabla["celda_id"] = [f"celda_{i}" for i in range(len(tabla))]
    tabla["cobertura_suelo"] = tabla["cobertura_suelo"].apply(
        lambda clase: "cultivo" if clase == _CLASE_WORLDCOVER_CROPLAND else "no_cultivo"
    )
    return tabla[
        ["celda_id", "cobertura_suelo", "altitud_m", "pendiente_grados", "amplitud_ndvi"]
    ]
