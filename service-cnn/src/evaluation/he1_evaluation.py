"""Evaluación conjunta de HE1 (tarea T5): no inferioridad frente a XGBoost y superioridad frente a B2.

HE1 se respalda solo si se cumplen HE1a y HE1b a la vez (3.3.2). La unidad de los
contrastes es la campaña externa; las semillas son réplicas computacionales y ya
vienen resumidas por la mediana en `experiment_runner.rmse_per_campaign` (4.12.4).
Con pocas campañas externas la potencia es limitada: se informa el menor valor p que
puede alcanzar el contraste de Wilcoxon unilateral exacto, 1 / 2^n.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.evaluation.hypothesis_tests import (
    NonInferiorityResult,
    SuperiorityResult,
    evaluate_non_inferiority,
    evaluate_superiority,
)


@dataclass(frozen=True)
class He1Result:
    campanas: list[int]
    n_campanas: int
    he1a: NonInferiorityResult
    he1b: SuperiorityResult
    he1_respaldada: bool
    p_minimo_posible: float
    diferencia_mediana_frente_a_xgboost_kg_ha: float
    diferencia_mediana_frente_a_b2_kg_ha: float


def evaluate_he1(
    rmse_por_campana: pd.DataFrame,
    margen: float,
    n_bootstrap: int,
    semilla: int,
    modelo: str = "cnn1d",
    comparador: str = "xgboost",
    benchmark: str = "B2",
) -> He1Result:
    """Contrasta HE1a y HE1b sobre el RMSE por campaña externa.

    Args:
        rmse_por_campana: columnas `modelo`, `campana_prueba`, `rmse`
            (salida de `experiment_runner.rmse_per_campaign`).
        margen: margen de no inferioridad δ (0,05).

    Raises:
        ValueError: si falta alguno de los tres modelos o si no tienen exactamente
            las mismas campañas (los contrastes son pareados por campaña).
    """
    series = {}
    for nombre in (modelo, comparador, benchmark):
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

    cnn, xgb, b2 = (series[n].to_numpy(dtype=float) for n in (modelo, comparador, benchmark))
    he1a = evaluate_non_inferiority(cnn, xgb, margen=margen, n_bootstrap=n_bootstrap, semilla=semilla)
    he1b = evaluate_superiority(cnn, b2, n_bootstrap=n_bootstrap, semilla=semilla)
    n = len(cnn)
    return He1Result(
        campanas=[int(c) for c in campanas[0]],
        n_campanas=n,
        he1a=he1a,
        he1b=he1b,
        he1_respaldada=bool(he1a.no_inferioridad_sostenida and he1b.superioridad_sostenida),
        p_minimo_posible=1.0 / 2**n,
        diferencia_mediana_frente_a_xgboost_kg_ha=float(np.median(cnn - xgb)),
        diferencia_mediana_frente_a_b2_kg_ha=float(np.median(cnn - b2)),
    )
