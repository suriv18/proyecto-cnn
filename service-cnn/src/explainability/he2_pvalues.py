"""Valores p de los tres criterios de HE2 y decisión conjunta con Holm (sección 3.3.5).

Los criterios de HE2 (estabilidad, fidelidad, concordancia) son reglas con umbral. La corrección
de Holm de 3.3.5 necesita un valor p por criterio; este módulo propone uno para cada uno. La
unidad de análisis es la campaña externa, igual que en HE1. Son definiciones propuestas
(estado `propuesto` en E5) hasta que el asesor las ratifique.

  * Estabilidad: prueba de permutación. H0: en cada campaña el orden de las fases es
    arbitrario. Se permutan las fases dentro de cada campaña y se compara W de Kendall y el
    Jaccard medio de las fases principales. Se exigen ambos, así que p = max(p_W, p_Jaccard).
  * Fidelidad: prueba de aleatorización exacta. H0: la fase de mayor contribución daña tanto
    como cualquier otra al ocluirla. En cada campaña se toma el rango de su aumento de RMSE entre
    todas las fases (1 = el mayor; los empates cuentan en su contra) y se suman los rangos entre
    campañas; p = P(suma <= observada) con rangos uniformes independientes.
  * Concordancia: binomial exacta unilateral. H0: la fase máxima de cada campaña cae al azar
    entre las fases; p0 = (fases del referente agronómico) / (fases totales).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np
from scipy import stats

from src.evaluation.multiplicity import holm_adjust, holm_reject
from src.explainability.agronomic_concordance import _FASES_VALIDAS_POR_HORIZONTE

_TOLERANCIA = 1e-12


@dataclass(frozen=True)
class StabilityPvalues:
    w_kendall: float
    jaccard_promedio: float
    p_w: float
    p_jaccard: float
    p: float  # max(p_w, p_jaccard): la estabilidad exige ambos criterios


def _estadisticos(valores: np.ndarray, n_principales: int) -> tuple[float, float]:
    """W de Kendall y Jaccard medio de las `n_principales` fases mayores; filas = campañas."""
    m, n = valores.shape
    rangos = stats.rankdata(valores, axis=1)
    suma = rangos.sum(axis=0)
    w = 12.0 * float(np.sum((suma - suma.mean()) ** 2)) / (m**2 * (n**3 - n)) if n > 1 else 1.0
    tops = [set(np.argsort(-fila, kind="stable")[:n_principales]) for fila in valores]
    jaccard = float(np.mean([len(a & b) / len(a | b) for a, b in combinations(tops, 2)]))
    return w, jaccard


def stability_pvalue(
    contribuciones_por_campana: list[dict[str, float]], n_fases_principales: int,
    n_permutaciones: int = 10000, semilla: int = 0,
) -> StabilityPvalues:
    """Valor p de la estabilidad por permutación de las fases dentro de cada campaña."""
    if len(contribuciones_por_campana) < 2:
        raise ValueError("La estabilidad requiere al menos 2 campañas externas.")
    fases = sorted(contribuciones_por_campana[0])
    valores = np.array([[c[f] for f in fases] for c in contribuciones_por_campana], dtype=float)
    w_obs, j_obs = _estadisticos(valores, n_fases_principales)

    rng = np.random.default_rng(semilla)
    mayores_w = mayores_j = 0
    for _ in range(n_permutaciones):
        permutado = np.array([rng.permutation(fila) for fila in valores])
        w, j = _estadisticos(permutado, n_fases_principales)
        mayores_w += w >= w_obs - _TOLERANCIA
        mayores_j += j >= j_obs - _TOLERANCIA
    p_w = (1 + mayores_w) / (1 + n_permutaciones)
    p_j = (1 + mayores_j) / (1 + n_permutaciones)
    return StabilityPvalues(w_obs, j_obs, p_w, p_j, max(p_w, p_j))


def fidelity_pvalue(aumentos_rmse_por_campana: list[dict[str, float]], fase_alta_por_campana: list[str]) -> float:
    """Valor p exacto de la fidelidad; ver el encabezado del módulo."""
    if len(fase_alta_por_campana) != len(aumentos_rmse_por_campana):
        raise ValueError("Se requiere una fase alta por campaña.")
    rangos = []
    for aumentos, alta in zip(aumentos_rmse_por_campana, fase_alta_por_campana):
        otros = [v for f, v in aumentos.items() if f != alta]
        rangos.append(1 + sum(v >= aumentos[alta] for v in otros))
    n_fases = len(aumentos_rmse_por_campana[0])

    distribucion = {0: 1.0}  # suma de rangos uniformes en 1..n_fases, por convolución
    for _ in rangos:
        siguiente: dict[int, float] = {}
        for suma, prob in distribucion.items():
            for r in range(1, n_fases + 1):
                siguiente[suma + r] = siguiente.get(suma + r, 0.0) + prob / n_fases
        distribucion = siguiente
    observada = sum(rangos)
    return float(sum(prob for suma, prob in distribucion.items() if suma <= observada))


def concordance_pvalue(fases_maximas_por_campana: list[str], horizonte: str, n_fases: int) -> float:
    """Valor p binomial exacto de la concordancia agronómica."""
    if horizonte not in _FASES_VALIDAS_POR_HORIZONTE:
        raise ValueError(f"horizonte debe ser uno de {sorted(_FASES_VALIDAS_POR_HORIZONTE)}; recibido '{horizonte}'.")
    validas = _FASES_VALIDAS_POR_HORIZONTE[horizonte]
    p0 = len(validas) / n_fases
    n = len(fases_maximas_por_campana)
    aciertos = sum(f in validas for f in fases_maximas_por_campana)
    return float(sum(comb(n, k) * p0**k * (1 - p0) ** (n - k) for k in range(aciertos, n + 1)))


@dataclass(frozen=True)
class He2Decision:
    p_ajustados: dict[str, float]
    rechazos: dict[str, bool]
    he2_respaldada: bool


def he2_family_decision(
    p_estabilidad: float, p_fidelidad: float, p_concordancia: float, alfa: float = 0.05
) -> He2Decision:
    """Holm sobre la familia {estabilidad, fidelidad, concordancia}; HE2 exige las tres."""
    p = {"estabilidad": p_estabilidad, "fidelidad": p_fidelidad, "concordancia": p_concordancia}
    rechazos = holm_reject(p, alfa)
    return He2Decision(holm_adjust(p), rechazos, all(rechazos.values()))
