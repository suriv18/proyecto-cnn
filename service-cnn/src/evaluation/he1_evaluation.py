"""Evaluación de HE1 definitiva (tesis v3, secciones 3.3.1 y 3.3.2).

HE1a (confirmatoria): la CNN-1D no es inferior a XGBoost ni a la mejor línea base provincial B* (la
mejor entre B2 y B4, elegida con la validación interna antes de ver las campañas externas) cuando la
diferencia relativa del RMSE q_f(R) = RMSE(CNN-1D)/RMSE(R) − 1 es menor que el margen δ en TODAS las
campañas externas. Es una regla exacta de signos: bajo H0 (mediana de q_f >= δ) cada campaña cae
por debajo del margen con probabilidad <= 1/2, de modo que el nivel de significancia es 0,5^n
(0,031 con cinco campañas). Con tan pocos bloques el bootstrap no es fiable.

HE1b (exploratoria): ganancia de habilidad de Murphy (1988), SS_f = 1 − MSE(CNN-1D)/MSE(B*), con el
número de campañas con ganancia y su valor p de signos. No condiciona la decisión sobre HE1.

La unidad es la campaña externa; las semillas ya vienen resumidas por la mediana en
`experiment_runner.rmse_per_campaign` (4.12.4). Como apoyo se informan el contraste de Wilcoxon
pareado unilateral frente al margen y la probabilidad posterior bayesiana, con el modelo t de
Student de la media de q_f (previa no informativa), de que la diferencia quede por debajo del margen
y dentro de la región de equivalencia práctica (−δ, δ) (Benavoli et al., 2017).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

BASES_ADMITIDAS = ("B2", "B4")


@dataclass(frozen=True)
class He1Result:
    campanas: list[int]
    n_campanas: int
    b_star: str
    q_xgboost: list[float]
    q_bstar: list[float]
    no_inferior_a_xgboost: bool
    no_inferior_a_bstar: bool
    he1a_respaldada: bool
    he1_respaldada: bool
    nivel_regla_de_signos: float
    wilcoxon_p_xgboost: float
    wilcoxon_p_bstar: float
    prob_posterior_no_inferior_xgboost: float
    prob_posterior_no_inferior_bstar: float
    prob_posterior_equivalencia_xgboost: float
    prob_posterior_equivalencia_bstar: float
    diferencia_mediana_frente_a_xgboost_kg_ha: float
    diferencia_mediana_frente_a_bstar_kg_ha: float
    ss_por_campana: list[float]
    ss_mediana: float
    n_campanas_con_ganancia: int
    p_signos_ganancia: float


def select_best_baseline(rmse_validacion_interna: dict[str, float]) -> str:
    """Elige B* entre B2 y B4 por menor RMSE de validación interna; en un empate conserva B2."""
    if not all(b in rmse_validacion_interna for b in BASES_ADMITIDAS):
        raise ValueError("Se requieren las líneas base B2 y B4 para elegir B*.")
    return "B4" if rmse_validacion_interna["B4"] < rmse_validacion_interna["B2"] else "B2"


def _posterior(q: np.ndarray, margen: float) -> tuple[float, float]:
    """P(mu < margen) y P(-margen < mu < margen) para la media de q con previa no informativa."""
    n = len(q)
    if n < 2:
        return float("nan"), float("nan")
    escala = float(q.std(ddof=1)) / np.sqrt(n)
    if escala == 0.0:
        m = float(q.mean())
        return float(m < margen), float(-margen < m < margen)
    post = stats.t(df=n - 1, loc=float(q.mean()), scale=escala)
    return float(post.cdf(margen)), float(post.cdf(margen) - post.cdf(-margen))


def _wilcoxon_menor_que_margen(q: np.ndarray, margen: float) -> float:
    d = q - margen
    if len(q) < 2 or np.all(d == 0):
        return 1.0
    return float(stats.wilcoxon(d, alternative="less").pvalue)


def evaluate_he1(
    rmse_por_campana: pd.DataFrame,
    margen: float,
    b_star: str,
    modelo: str = "cnn1d",
    comparador: str = "xgboost",
) -> He1Result:
    """Contrasta HE1 sobre el RMSE por campaña externa.

    Args:
        rmse_por_campana: columnas `modelo`, `campana_prueba`, `rmse`
            (salida de `experiment_runner.rmse_per_campaign`).
        margen: margen de no inferioridad δ (0,05).
        b_star: línea base elegida antes de las campañas externas ("B2" o "B4").

    Raises:
        ValueError: si `b_star` no es B2 o B4, si falta algún modelo o si no tienen exactamente
            las mismas campañas (los contrastes son pareados por campaña).
    """
    if b_star not in BASES_ADMITIDAS:
        raise ValueError(f"B* debe ser B2 o B4; recibido '{b_star}'.")
    series = {}
    for nombre in (modelo, comparador, b_star):
        sub = rmse_por_campana[rmse_por_campana["modelo"] == nombre]
        if sub.empty:
            raise ValueError(f"Falta '{nombre}' en la tabla de RMSE por campaña.")
        series[nombre] = sub.set_index("campana_prueba")["rmse"].sort_index()
    campanas = [list(s.index) for s in series.values()]
    if not (campanas[0] == campanas[1] == campanas[2]):
        raise ValueError(
            "Los tres modelos deben tener exactamente las mismas campañas externas (contraste pareado); "
            f"recibido {campanas}."
        )

    cnn, xgb, base = (series[n].to_numpy(dtype=float) for n in (modelo, comparador, b_star))
    n = len(cnn)
    q_x, q_b = cnn / xgb - 1.0, cnn / base - 1.0
    ni_x, ni_b = bool(np.all(q_x < margen)), bool(np.all(q_b < margen))
    nox, eqx = _posterior(q_x, margen)
    nob, eqb = _posterior(q_b, margen)
    ss = 1.0 - (cnn / base) ** 2
    ganancias = int(np.sum(ss > 0))
    return He1Result(
        campanas=[int(c) for c in campanas[0]],
        n_campanas=n,
        b_star=b_star,
        q_xgboost=[float(v) for v in q_x],
        q_bstar=[float(v) for v in q_b],
        no_inferior_a_xgboost=ni_x,
        no_inferior_a_bstar=ni_b,
        he1a_respaldada=bool(ni_x and ni_b),
        he1_respaldada=bool(ni_x and ni_b),  # HE1b es exploratoria y no condiciona la decisión
        nivel_regla_de_signos=0.5**n,
        wilcoxon_p_xgboost=_wilcoxon_menor_que_margen(q_x, margen),
        wilcoxon_p_bstar=_wilcoxon_menor_que_margen(q_b, margen),
        prob_posterior_no_inferior_xgboost=nox,
        prob_posterior_no_inferior_bstar=nob,
        prob_posterior_equivalencia_xgboost=eqx,
        prob_posterior_equivalencia_bstar=eqb,
        diferencia_mediana_frente_a_xgboost_kg_ha=float(np.median(cnn - xgb)),
        diferencia_mediana_frente_a_bstar_kg_ha=float(np.median(cnn - base)),
        ss_por_campana=[float(v) for v in ss],
        ss_mediana=float(np.median(ss)),
        n_campanas_con_ganancia=ganancias,
        p_signos_ganancia=float(stats.binom.sf(ganancias - 1, n, 0.5)),
    )
