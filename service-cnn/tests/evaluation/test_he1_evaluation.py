"""Evaluación de HE1 definitiva (tesis v3, 3.3.1-3.3.2).

HE1a (confirmatoria): la CNN-1D no es inferior a XGBoost ni a la mejor línea base B* (la mejor
entre B2 y B4) cuando la diferencia relativa del RMSE es menor que el margen en TODAS las campañas
externas (regla exacta de signos, nivel 0,5^n). HE1b (exploratoria): ganancia de habilidad SS de
Murphy frente a B*, que no condiciona la decisión. La unidad es la campaña externa y las semillas
ya vienen resumidas por la mediana (`rmse_per_campaign`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.he1_evaluation import evaluate_he1, select_best_baseline

CAMPANAS = [2021, 2022, 2023, 2024, 2025]
XGB = [200.0, 210.0, 190.0, 220.0, 205.0]
B2 = [220.0, 230.0, 215.0, 240.0, 225.0]
B4 = [210.0, 215.0, 200.0, 230.0, 215.0]


def _tabla(cnn, xgb=XGB, b2=B2, b4=B4, campanas=CAMPANAS):
    filas = []
    for nombre, valores in (("cnn1d", cnn), ("xgboost", xgb), ("B2", b2), ("B4", b4)):
        filas += [{"modelo": nombre, "campana_prueba": c, "rmse": v} for c, v in zip(campanas, valores)]
    return pd.DataFrame(filas)


def _con(base, factor):
    return [x * factor for x in base]


class TestDecisionDeHe1a:
    def test_se_respalda_cuando_es_no_inferior_a_xgboost_y_a_bstar_en_todas_las_campanas(self):
        r = evaluate_he1(_tabla(_con(XGB, 1.02), b2=_con(XGB, 1.0)), margen=0.05, b_star="B2")

        assert r.no_inferior_a_xgboost and r.no_inferior_a_bstar
        assert r.he1a_respaldada is True and r.he1_respaldada is True

    def test_una_sola_campana_fuera_del_margen_frente_a_xgboost_basta_para_no_respaldar(self):
        cnn = _con(XGB, 1.02)
        cnn[3] = XGB[3] * 1.08  # 8 % peor que XGBoost en una campaña

        r = evaluate_he1(_tabla(cnn, b2=_con(XGB, 1.0)), margen=0.05, b_star="B2")

        assert r.no_inferior_a_xgboost is False
        assert r.he1_respaldada is False

    def test_una_sola_campana_fuera_del_margen_frente_a_bstar_basta_para_no_respaldar(self):
        cnn = _con(B2, 1.02)
        cnn[1] = B2[1] * 1.07

        r = evaluate_he1(_tabla(cnn, xgb=list(cnn)), margen=0.05, b_star="B2")  # XGBoost igual a la CNN: solo B* se viola

        assert r.no_inferior_a_xgboost is True
        assert r.no_inferior_a_bstar is False
        assert r.he1_respaldada is False

    def test_la_superioridad_no_es_necesaria_para_respaldar(self):
        # la CNN-1D es 3 % peor que B*: no es superior, pero sí no inferior (3 % < 5 %)
        r = evaluate_he1(_tabla(_con(B2, 1.03), xgb=_con(B2, 1.0)), margen=0.05, b_star="B2")

        assert r.he1_respaldada is True
        assert r.n_campanas_con_ganancia == 0

    def test_la_regla_es_estricta_con_el_margen(self):
        # exactamente 5 % no cumple q < 0,05
        r = evaluate_he1(_tabla(_con(B2, 1.05), xgb=_con(B2, 1.0)), margen=0.05, b_star="B2")

        assert r.no_inferior_a_bstar is False

    def test_usa_la_linea_base_indicada_como_bstar(self):
        cnn = _con(B4, 1.02)  # no inferior a B4, pero 2 % + 9 % peor que B2 no aplica: B2 es peor que B4

        r = evaluate_he1(_tabla(cnn, xgb=_con(B4, 1.0)), margen=0.05, b_star="B4")

        assert r.b_star == "B4" and r.he1_respaldada is True
        assert r.q_bstar == pytest.approx([0.02] * 5)


class TestNivelYApoyo:
    def test_el_nivel_de_la_regla_de_signos_es_medio_a_la_n(self):
        r = evaluate_he1(_tabla(_con(XGB, 1.0), b2=_con(XGB, 1.0)), margen=0.05, b_star="B2")

        assert r.n_campanas == 5
        assert r.nivel_regla_de_signos == pytest.approx(1 / 32)

    def test_la_probabilidad_posterior_de_no_inferioridad_es_alta_si_todo_queda_lejos_del_margen(self):
        r = evaluate_he1(_tabla([x * (1 + e) for x, e in zip(XGB, (0.0, 0.01, -0.01, 0.005, 0.0))], b2=_con(XGB, 1.0)), margen=0.05, b_star="B2")

        assert r.prob_posterior_no_inferior_xgboost > 0.95
        assert r.prob_posterior_equivalencia_xgboost > 0.9

    def test_la_probabilidad_posterior_es_baja_si_la_cnn_es_claramente_peor(self):
        r = evaluate_he1(_tabla(_con(XGB, 1.3), b2=_con(XGB, 1.0)), margen=0.05, b_star="B2")

        assert r.prob_posterior_no_inferior_xgboost < 0.05

    def test_informa_wilcoxon_y_las_diferencias_de_mediana_en_kg_ha(self):
        cnn = _con(XGB, 1.01)

        r = evaluate_he1(_tabla(cnn, b2=_con(XGB, 1.0)), margen=0.05, b_star="B2")

        assert 0.0 < r.wilcoxon_p_xgboost <= 1.0
        assert r.diferencia_mediana_frente_a_xgboost_kg_ha == pytest.approx(np.median(np.array(cnn) - np.array(XGB)))
        assert r.diferencia_mediana_frente_a_bstar_kg_ha == pytest.approx(np.median(np.array(cnn) - np.array(XGB)))


class TestHe1bExploratoria:
    def test_skill_score_de_murphy_por_campana(self):
        cnn = _con(B2, 0.9)

        r = evaluate_he1(_tabla(cnn, xgb=_con(B2, 1.0)), margen=0.05, b_star="B2")

        assert r.ss_por_campana == pytest.approx([1 - 0.9**2] * 5)
        assert r.ss_mediana == pytest.approx(1 - 0.9**2)

    def test_cuenta_las_campanas_con_ganancia_y_su_valor_p_de_signos(self):
        r = evaluate_he1(_tabla(_con(B2, 0.9), xgb=_con(B2, 1.0)), margen=0.05, b_star="B2")

        assert r.n_campanas_con_ganancia == 5
        assert r.p_signos_ganancia == pytest.approx(1 / 32)

    def test_con_tres_de_cinco_el_valor_p_de_signos_es_un_medio(self):
        cnn = [B2[0] * 0.95, B2[1] * 0.95, B2[2] * 0.95, B2[3] * 1.02, B2[4] * 1.02]

        r = evaluate_he1(_tabla(cnn, xgb=_con(B2, 1.0)), margen=0.05, b_star="B2")

        assert r.n_campanas_con_ganancia == 3
        assert r.p_signos_ganancia == pytest.approx(0.5)

    def test_una_ganancia_nula_no_cuenta_como_ganancia(self):
        r = evaluate_he1(_tabla(_con(B2, 1.0), xgb=_con(B2, 1.0)), margen=0.05, b_star="B2")

        assert r.n_campanas_con_ganancia == 0
        assert r.ss_mediana == pytest.approx(0.0)


class TestValidaciones:
    def test_exige_las_mismas_campanas_para_los_tres_modelos(self):
        tabla = _tabla(_con(XGB, 1.0))
        tabla = tabla[~((tabla.modelo == "xgboost") & (tabla.campana_prueba == 2023))]

        with pytest.raises(ValueError, match="campañas"):
            evaluate_he1(tabla, margen=0.05, b_star="B2")

    def test_exige_la_linea_base_indicada(self):
        tabla = _tabla(_con(XGB, 1.0))

        with pytest.raises(ValueError, match="B4"):
            evaluate_he1(tabla[tabla.modelo != "B4"], margen=0.05, b_star="B4")

    def test_solo_acepta_b2_o_b4_como_bstar(self):
        with pytest.raises(ValueError, match="B2 o B4"):
            evaluate_he1(_tabla(_con(XGB, 1.0)), margen=0.05, b_star="B3")

    def test_las_unidades_son_las_campanas_en_orden_aunque_la_tabla_venga_desordenada(self):
        ordenada = _tabla(_con(XGB, 1.02), b2=_con(XGB, 1.0))

        a = evaluate_he1(ordenada, margen=0.05, b_star="B2")
        b = evaluate_he1(ordenada.sample(frac=1.0, random_state=3), margen=0.05, b_star="B2")

        assert a.q_xgboost == b.q_xgboost and a.campanas == b.campanas == CAMPANAS


class TestSeleccionDeBstar:
    def test_elige_la_de_menor_rmse_de_validacion_interna(self):
        assert select_best_baseline({"B2": 210.0, "B4": 200.0}) == "B4"
        assert select_best_baseline({"B2": 190.0, "B4": 200.0}) == "B2"

    def test_en_un_empate_conserva_b2(self):
        assert select_best_baseline({"B2": 200.0, "B4": 200.0}) == "B2"

    def test_exige_ambas_lineas_base(self):
        with pytest.raises(ValueError, match="B2 y B4"):
            select_best_baseline({"B2": 200.0})
