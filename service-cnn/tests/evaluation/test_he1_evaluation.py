"""Pruebas de la evaluación conjunta de HE1 (tarea T5 del plan del piloto).

HE1 se respalda solo si se cumplen HE1a (no inferioridad frente a XGBoost) y HE1b
(superioridad frente a B2) a la vez (3.3.2). La unidad es la campaña externa y las
semillas ya vienen resumidas por la mediana (`rmse_per_campaign`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.he1_evaluation import evaluate_he1

CAMPANAS = [2021, 2022, 2023, 2024, 2025]
KW = dict(margen=0.05, n_bootstrap=500, semilla=0)


def _tabla(cnn, xgb, b2, campanas=CAMPANAS):
    filas = []
    for nombre, valores in (("cnn1d", cnn), ("xgboost", xgb), ("B2", b2)):
        filas += [{"modelo": nombre, "campana_prueba": c, "rmse": v} for c, v in zip(campanas, valores)]
    return pd.DataFrame(filas)


XGB = [200.0, 210.0, 190.0, 220.0, 205.0]
B2 = [220.0, 230.0, 215.0, 240.0, 225.0]


class TestEvaluateHe1:
    def test_se_respalda_cuando_cumple_no_inferioridad_y_superioridad(self):
        cnn = [180.0, 190.0, 175.0, 200.0, 185.0]  # mejor que XGBoost y que B2

        r = evaluate_he1(_tabla(cnn, XGB, B2), **KW)

        assert r.he1a.no_inferioridad_sostenida and r.he1b.superioridad_sostenida
        assert r.he1_respaldada is True

    def test_no_se_respalda_si_falla_la_superioridad_aunque_haya_no_inferioridad(self):
        cnn = [x * 1.02 for x in XGB]  # no inferior a XGBoost (2 % < 5 %) pero peor que B2? B2 es peor que XGBoost

        r = evaluate_he1(_tabla(cnn, XGB, [x * 0.9 for x in XGB]), **KW)  # B2 mejor que la CNN

        assert r.he1a.no_inferioridad_sostenida is True
        assert r.he1b.superioridad_sostenida is False
        assert r.he1_respaldada is False

    def test_no_se_respalda_si_falla_la_no_inferioridad(self):
        cnn = [x * 1.5 for x in XGB]  # mucho peor que XGBoost

        r = evaluate_he1(_tabla(cnn, XGB, [x * 2.0 for x in XGB]), **KW)

        assert r.he1a.no_inferioridad_sostenida is False
        assert r.he1_respaldada is False

    def test_informa_el_numero_de_campanas_y_el_valor_p_minimo_posible(self):
        r = evaluate_he1(_tabla([180.0] * 5, XGB, B2), **KW)

        assert r.n_campanas == 5
        assert r.p_minimo_posible == pytest.approx(1 / 32)  # Wilcoxon unilateral exacto con n = 5

    def test_informa_las_diferencias_de_mediana_en_kg_ha(self):
        cnn = [180.0, 190.0, 175.0, 200.0, 185.0]

        r = evaluate_he1(_tabla(cnn, XGB, B2), **KW)

        assert r.diferencia_mediana_frente_a_xgboost_kg_ha == pytest.approx(np.median(np.array(cnn) - np.array(XGB)))
        assert r.diferencia_mediana_frente_a_b2_kg_ha == pytest.approx(np.median(np.array(cnn) - np.array(B2)))

    def test_exige_las_mismas_campanas_para_los_tres(self):
        tabla = _tabla([180.0] * 5, XGB, B2)
        tabla = tabla[~((tabla.modelo == "xgboost") & (tabla.campana_prueba == 2023))]

        with pytest.raises(ValueError, match="campañas"):
            evaluate_he1(tabla, **KW)

    def test_exige_los_tres_modelos(self):
        tabla = _tabla([180.0] * 5, XGB, B2)

        with pytest.raises(ValueError, match="B2"):
            evaluate_he1(tabla[tabla.modelo != "B2"], **KW)

    def test_las_unidades_son_las_campanas_en_orden_aunque_la_tabla_venga_desordenada(self):
        cnn = [180.0, 190.0, 175.0, 200.0, 185.0]
        ordenada = _tabla(cnn, XGB, B2)
        desordenada = ordenada.sample(frac=1.0, random_state=3)

        a, b = evaluate_he1(ordenada, **KW), evaluate_he1(desordenada, **KW)

        assert a.he1a.limite_superior_ic95 == b.he1a.limite_superior_ic95
