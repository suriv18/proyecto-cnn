"""Estabilidad de las explicaciones SHAP entre campañas externas (HE2a, sección 3.3.3).

W de Kendall (coeficiente de concordancia) sobre el ordenamiento de
contribución por fase entre campañas — umbral W >= 0.70. Índice de Jaccard de
las 3 fases principales entre campañas — umbral J >= 0.50, que al comparar
conjuntos de 3 fases dentro de 5 preespecificadas equivale a que al menos 2 de
las 3 fases principales coincidan (sección 3.3.3).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import stats

UMBRAL_W_KENDALL = 0.70
UMBRAL_JACCARD = 0.50


def kendalls_w(contribuciones_por_campana: list[dict[str, float]]) -> float:
    """Calcula el coeficiente de concordancia W de Kendall.

    Args:
        contribuciones_por_campana: lista de diccionarios `{fase: contribución
            absoluta agregada}`, uno por campaña externa — mismas fases en
            cada diccionario (salida de `aggregate_shap_by_phase`, agregada
            por semilla dentro de cada campaña antes de llegar aquí, sección
            3.3.3).

    Returns:
        W en [0, 1]: 1 indica concordancia perfecta del ordenamiento de fases
        entre todas las campañas; valores bajos indican ordenamientos
        inconsistentes.

    Raises:
        ValueError: si se proveen menos de 2 campañas.
    """
    if len(contribuciones_por_campana) < 2:
        raise ValueError(
            "kendalls_w requiere al menos 2 campañas externas para evaluar "
            "concordancia entre ellas."
        )

    fases = sorted(contribuciones_por_campana[0].keys())
    m = len(contribuciones_por_campana)  # número de "jueces" (campañas)
    n = len(fases)  # número de "ítems" (fases)

    matriz_rangos = np.array(
        [
            stats.rankdata([campana[fase] for fase in fases])
            for campana in contribuciones_por_campana
        ]
    )
    suma_rangos_por_fase = matriz_rangos.sum(axis=0)
    media_suma_rangos = suma_rangos_por_fase.mean()
    s = float(np.sum((suma_rangos_por_fase - media_suma_rangos) ** 2))

    denominador = m**2 * (n**3 - n)
    if denominador == 0:
        return 1.0  # una sola fase: concordancia trivialmente perfecta

    return 12.0 * s / denominador


def jaccard_index(conjunto_a: set[str], conjunto_b: set[str]) -> float:
    """Índice de Jaccard entre dos conjuntos de fases: |A∩B| / |A∪B|."""
    union = conjunto_a | conjunto_b
    if not union:
        return 0.0
    interseccion = conjunto_a & conjunto_b
    return len(interseccion) / len(union)


def _top_n_fases(contribucion_por_fase: dict[str, float], n: int) -> set[str]:
    ordenadas = sorted(contribucion_por_fase, key=contribucion_por_fase.get, reverse=True)
    return set(ordenadas[:n])


@dataclass(frozen=True)
class StabilityResult:
    """Resultado de la evaluación de estabilidad (HE2a)."""

    w_kendall: float
    jaccard_promedio: float
    estabilidad_sostenida: bool


def evaluate_stability(
    contribuciones_por_campana: list[dict[str, float]], n_fases_principales: int
) -> StabilityResult:
    """Evalúa la estabilidad de las explicaciones entre campañas externas (HE2a).

    Args:
        contribuciones_por_campana: una entrada por campaña externa (ya
            agregada por semilla dentro de cada campaña, sección 3.3.3).
        n_fases_principales: número de fases de mayor contribución a comparar
            por Jaccard entre cada par de campañas (3 en la tesis).

    Returns:
        `StabilityResult` con W de Kendall, el promedio de Jaccard entre todos
        los pares de campañas, y si la estabilidad se sostiene (W >= 0.70 Y
        Jaccard promedio >= 0.50, sección 3.3.3).
    """
    w = kendalls_w(contribuciones_por_campana)

    conjuntos_top_n = [
        _top_n_fases(campana, n_fases_principales)
        for campana in contribuciones_por_campana
    ]
    jaccards_por_par = [
        jaccard_index(a, b) for a, b in combinations(conjuntos_top_n, 2)
    ]
    jaccard_promedio = float(np.mean(jaccards_por_par))

    return StabilityResult(
        w_kendall=w,
        jaccard_promedio=jaccard_promedio,
        estabilidad_sostenida=(w >= UMBRAL_W_KENDALL and jaccard_promedio >= UMBRAL_JACCARD),
    )
