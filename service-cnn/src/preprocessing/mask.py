"""Máscara de superficie agrícola (sección 4.10.1).

Combina capa de cobertura de suelo + modelo digital de elevación (MDE) +
criterios fenométricos de la serie NDVI para excluir coberturas no cultivadas
del cálculo de NDVI. Los umbrales de altitud, pendiente y amplitud se fijan y
registran antes de la evaluación definitiva, calibrados solo con campañas de
entrenamiento o fuentes externas independientes del resultado de prueba.

Limitación declarada en la tesis: la máscara reduce contaminación por
coberturas no agrícolas, pero NO distingue quinua de otros cultivos — esa
limitación se controla aparte con la proporción de superficie sembrada con
quinua (covariable dinámica) y análisis de sensibilidad por representatividad.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

_COBERTURAS_AGRICOLAS = {"cultivo"}


@dataclass(frozen=True)
class MaskThresholds:
    """Umbrales de la máscara agrícola, fijados y registrados antes de H2."""

    altitud_min_m: float
    altitud_max_m: float
    pendiente_max_grados: float
    amplitud_ndvi_min: float


def apply_agricultural_mask(
    celdas: pd.DataFrame, umbrales: MaskThresholds
) -> pd.DataFrame:
    """Clasifica cada celda como agrícola o no, según los umbrales dados.

    Args:
        celdas: columnas `celda_id`, `cobertura_suelo`, `altitud_m`,
            `pendiente_grados`, `amplitud_ndvi`.
        umbrales: `MaskThresholds` ya calibrados (ver
            `calibrate_ndvi_amplitude_threshold` para el umbral de NDVI).

    Returns:
        `celdas` con dos columnas adicionales:
          - `es_agricola` (bool): True solo si supera los 4 criterios.
          - `motivo_exclusion` (str): lista separada por comas de TODOS los
            criterios que la celda no cumplió (vacío si es elegible) — se
            reportan todos, no solo el primero, para permitir el análisis de
            sensibilidad por umbral que exige la sección 4.10.1.
    """
    resultado = celdas.copy()

    motivos = pd.Series([[] for _ in range(len(resultado))], index=resultado.index)

    fuera_de_altitud = (resultado["altitud_m"] < umbrales.altitud_min_m) | (
        resultado["altitud_m"] > umbrales.altitud_max_m
    )
    for i in resultado.index[fuera_de_altitud]:
        motivos[i].append("altitud")

    pendiente_excesiva = resultado["pendiente_grados"] > umbrales.pendiente_max_grados
    for i in resultado.index[pendiente_excesiva]:
        motivos[i].append("pendiente")

    amplitud_insuficiente = resultado["amplitud_ndvi"] < umbrales.amplitud_ndvi_min
    for i in resultado.index[amplitud_insuficiente]:
        motivos[i].append("amplitud_ndvi")

    cobertura_no_agricola = ~resultado["cobertura_suelo"].isin(_COBERTURAS_AGRICOLAS)
    for i in resultado.index[cobertura_no_agricola]:
        motivos[i].append("cobertura_suelo")

    resultado["motivo_exclusion"] = motivos.apply(lambda lista: ", ".join(lista))
    resultado["es_agricola"] = motivos.apply(len) == 0

    return resultado


def calibrate_ndvi_amplitude_threshold(
    serie_ndvi_entrenamiento: pd.DataFrame, percentil: float
) -> float:
    """Calibra el umbral de amplitud NDVI usando únicamente datos de entrenamiento.

    Sección 4.10.1: los umbrales se calibran solo con campañas de entrenamiento
    o fuentes externas independientes del resultado de prueba. Esta función
    recibe explícitamente un DataFrame ya restringido a entrenamiento — es
    responsabilidad del llamador (el pliegue de validación, sección 4.12.2)
    garantizar esa restricción antes de invocarla.

    Args:
        serie_ndvi_entrenamiento: columnas `celda_id`, `campana_id`,
            `ndvi_max`, `ndvi_min` — ya filtradas a campañas de entrenamiento.
        percentil: percentil (0-100) de la distribución de amplitudes
            (ndvi_max - ndvi_min) usado como umbral mínimo.

    Returns:
        El umbral de amplitud NDVI calibrado (float positivo).
    """
    amplitudes = (
        serie_ndvi_entrenamiento["ndvi_max"] - serie_ndvi_entrenamiento["ndvi_min"]
    )
    return float(amplitudes.quantile(percentil / 100.0))


def summarize_amplitude_calibration(
    tabla: pd.DataFrame,
    percentiles: list[float],
    altitud_min_m: float,
    altitud_max_m: float,
    pendiente_max_grados: float,
    campanas_entrenamiento: list[int],
) -> dict:
    """Resume la calibración de `amplitud_ndvi_min` para varios percentiles.

    El umbral se calibra solo con celdas de cultivo que ya cumplen altitud y
    pendiente: la amplitud no distingue cultivo de otras coberturas (eso lo
    hace la clase de cobertura), solo descarta celdas "cultivo" sin
    estacionalidad. Para cada percentil se informa además qué porcentaje de
    las celdas elegibles de cada departamento quedaría excluido, para detectar
    un sesgo regional antes de fijar el umbral.

    Args:
        tabla: columnas `departamento`, `campana_id`, `cobertura_suelo`,
            `altitud_m`, `pendiente_grados`, `ndvi_max`, `ndvi_min`.
        percentiles: percentiles (0-100) a evaluar.
        altitud_min_m, altitud_max_m, pendiente_max_grados: umbrales ya
            fijados de la máscara.
        campanas_entrenamiento: campañas permitidas.

    Raises:
        ValueError: si `tabla` contiene campañas fuera del entrenamiento
            (regla anti-fuga, sección 4.12.2).
    """
    fuera = set(tabla["campana_id"].unique()) - set(campanas_entrenamiento)
    if fuera:
        raise ValueError(
            f"La tabla contiene campañas fuera del conjunto de entrenamiento: "
            f"{sorted(fuera)}. La calibración solo puede usar entrenamiento "
            "(sección 4.12.2)."
        )

    cultivo = tabla[tabla["cobertura_suelo"].isin(_COBERTURAS_AGRICOLAS)]
    elegibles = cultivo[
        cultivo["altitud_m"].between(altitud_min_m, altitud_max_m)
        & (cultivo["pendiente_grados"] <= pendiente_max_grados)
    ].copy()
    elegibles["amplitud"] = elegibles["ndvi_max"] - elegibles["ndvi_min"]

    umbrales: dict[str, dict] = {}
    for percentil in percentiles:
        umbral = calibrate_ndvi_amplitude_threshold(elegibles, percentil)
        excluida = elegibles["amplitud"] < umbral
        por_departamento = (
            excluida.groupby(elegibles["departamento"]).mean() * 100.0
        ).to_dict()
        clave = f"{percentil:g}"
        umbrales[clave] = {
            "umbral": umbral,
            "excluidas_pct_total": float(excluida.mean() * 100.0),
            "excluidas_pct_por_departamento": {
                str(k): float(v) for k, v in por_departamento.items()
            },
        }

    return {
        "n_celdas_cultivo": int(len(cultivo)),
        "n_elegibles": int(len(elegibles)),
        "campanas_entrenamiento": [int(c) for c in campanas_entrenamiento],
        "umbrales": umbrales,
    }


def summarize_cropland_altitude(
    celdas: pd.DataFrame, altitud_min_m: float, altitud_max_m: float
) -> pd.DataFrame:
    """Altitud mediana de la superficie de cultivo de cada provincia.

    Criterio complementario de la sección 4.5.3 para decidir si una provincia
    pertenece al ámbito altoandino. Si la mediana queda fuera del rango
    altitudinal de la máscara, el NDVI enmascarado se calcula sobre una
    fracción pequeña de la superficie cultivada y no representa a la provincia.

    Args:
        celdas: columnas `provincia_id`, `celda_id`, `cobertura_suelo`,
            `altitud_m`. Una celda repetida entre campañas se cuenta una vez.

    Returns:
        DataFrame con `provincia_id`, `n_celdas`, `altitud_mediana_m` y
        `frac_en_rango` (fracción de celdas dentro de [min, max]).
    """
    cultivo = celdas[celdas["cobertura_suelo"].isin(_COBERTURAS_AGRICOLAS)]
    cultivo = cultivo.drop_duplicates("celda_id")
    resumen = cultivo.groupby("provincia_id")["altitud_m"].agg(
        n_celdas="count",
        altitud_mediana_m="median",
        frac_en_rango=lambda s: float(s.between(altitud_min_m, altitud_max_m).mean()),
    )
    return resumen.reset_index()


def summarize_agricultural_surface_altitude(
    celdas: pd.DataFrame,
    altitud_min_m: float,
    altitud_max_m: float,
    pendiente_max_grados: float,
) -> pd.DataFrame:
    """Altitud MEDIA de la superficie agrícola de cada provincia (covariable estática, 4.10.3).

    «Superficie agrícola» es el cultivo que pasa los criterios de altitud y
    pendiente de la máscara: la misma superficie sobre la que se calcula el NDVI.
    A diferencia de `summarize_cropland_altitude` (todo el cultivo, mediana, para
    diagnóstico), esta es la covariable que entra al modelo.

    Args:
        celdas: columnas `provincia_id`, `celda_id`, `cobertura_suelo`,
            `altitud_m`, `pendiente_grados`. Una celda repetida entre campañas
            se cuenta una vez.

    Returns:
        DataFrame con `provincia_id`, `n_celdas` y `altitud_media_m`.
    """
    cultivo = celdas[celdas["cobertura_suelo"].isin(_COBERTURAS_AGRICOLAS)].drop_duplicates("celda_id")
    en_mascara = cultivo[
        cultivo["altitud_m"].between(altitud_min_m, altitud_max_m)
        & (cultivo["pendiente_grados"] <= pendiente_max_grados)
    ]
    resumen = en_mascara.groupby("provincia_id")["altitud_m"].agg(n_celdas="count", altitud_media_m="mean")
    return resumen.reset_index()
