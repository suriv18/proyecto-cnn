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
