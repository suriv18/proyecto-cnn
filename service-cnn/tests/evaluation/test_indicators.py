"""Indicadores de las dimensiones X1, X2 e Y1 del Anexo 01.

Y1 (exactitud): RMSE, MAE, rRMSE y R² por campaña y modelo. X1 (modelo CNN-1D): campo receptivo
temporal, porcentaje de predicciones válidas y de ejecuciones reproducibles. X2 (SHAP): porcentaje de
predicciones explicadas y valor SHAP absoluto medio y orden de importancia por predictor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.evaluation.indicators import (
    accuracy_indicators,
    accuracy_table,
    explained_fraction,
    receptive_field,
    reproducibility_rate,
    shap_predictor_importance,
    valid_prediction_rate,
)
from src.models.cnn1d import CNN1DConfig, QuinuaYieldCNN1D


class TestIndicadoresDeExactitud:
    Y = np.array([100.0, 200.0, 300.0])
    P = np.array([110.0, 190.0, 330.0])

    def test_valores_calculados_a_mano(self):
        r = accuracy_indicators(self.Y, self.P)

        assert r["rmse"] == pytest.approx(np.sqrt(1100 / 3))
        assert r["mae"] == pytest.approx(50 / 3)
        assert r["rrmse"] == pytest.approx(np.sqrt(1100 / 3) / 200 * 100)
        assert r["r2"] == pytest.approx(1 - 1100 / 20000)

    def test_el_rrmse_es_nan_si_la_media_observada_es_cercana_a_cero(self):
        # anomalías con media cero: la Tabla 9 prohíbe calcular el rRMSE
        r = accuracy_indicators(np.array([-5.0, 0.0, 5.0]), np.array([-4.0, 1.0, 4.0]))

        assert np.isnan(r["rrmse"]) and np.isfinite(r["rmse"])

    def test_tabla_por_modelo_y_campana_con_mediana_entre_semillas(self):
        filas = []
        for semilla, sesgo in enumerate((0.0, 10.0, 20.0)):
            for c in (2019, 2020):
                filas += [{"campana_prueba": c, "modelo": "cnn1d", "semilla": semilla, "y_real": y, "y_pred": y + sesgo}
                          for y in (100.0, 200.0, 300.0)]

        t = accuracy_table(pd.DataFrame(filas))

        assert set(t.columns) >= {"modelo", "campana_prueba", "rmse", "mae", "rrmse", "r2"}
        assert len(t) == 2
        assert t["rmse"].tolist() == pytest.approx([10.0, 10.0])  # mediana de sesgos 0, 10 y 20
        assert t["mae"].tolist() == pytest.approx([10.0, 10.0])


class TestCampoReceptivo:
    def test_dos_convoluciones_de_kernel_tres_dan_cinco_pasos(self):
        m = nn.Sequential(nn.Conv1d(1, 2, 3, padding=1), nn.Conv1d(2, 2, 3, padding=1))

        assert receptive_field(m) == 5

    def test_la_dilatacion_y_el_stride_se_acumulan(self):
        m = nn.Sequential(nn.Conv1d(1, 2, 3), nn.MaxPool1d(2), nn.Conv1d(2, 2, 3))

        assert receptive_field(m) == 8  # 3, luego 4 con el pooling (salto 2), luego 4 + 2*2

        d = nn.Sequential(nn.Conv1d(1, 2, 3, dilation=2))
        assert receptive_field(d) == 5

    def test_la_cnn_del_proyecto_tiene_campo_receptivo_cinco(self):
        modelo = QuinuaYieldCNN1D(CNN1DConfig(n_variables=6, n_pasos_temporales=5, n_covariables=2))

        assert receptive_field(modelo) == 5

    def test_un_modelo_sin_convoluciones_tiene_campo_uno(self):
        assert receptive_field(nn.Sequential(nn.Linear(3, 3))) == 1


class TestPrediccionesValidasYReproducibilidad:
    def test_porcentaje_de_predicciones_validas(self):
        assert valid_prediction_rate(np.array([1.0, np.nan, 2.0, np.inf])) == pytest.approx(0.5)

    def test_sin_predicciones_lanza_error(self):
        with pytest.raises(ValueError, match="predicciones"):
            valid_prediction_rate(np.array([]))

    def test_una_ejecucion_determinista_es_totalmente_reproducible(self):
        def ejecutar(semilla):
            g = np.random.default_rng(semilla)
            return g.normal(size=5)

        assert reproducibility_rate(ejecutar, semillas=[0, 1, 2]) == pytest.approx(1.0)

    def test_una_ejecucion_con_azar_no_controlado_no_es_reproducible(self):
        def ejecutar(_semilla):
            return np.random.default_rng().normal(size=5)

        assert reproducibility_rate(ejecutar, semillas=[0, 1, 2]) == pytest.approx(0.0)

    def test_una_cnn_entrenada_dos_veces_con_la_misma_semilla_coincide(self):
        x = torch.randn(12, 2, 4)
        z = torch.randn(12, 1)

        def ejecutar(semilla):
            torch.manual_seed(semilla)
            m = QuinuaYieldCNN1D(CNN1DConfig(n_variables=2, n_pasos_temporales=4, n_covariables=1, filtros_capa1=4, filtros_capa2=4))
            m.eval()
            with torch.no_grad():
                return m(x, z).numpy().ravel()

        assert reproducibility_rate(ejecutar, semillas=[0, 1]) == pytest.approx(1.0)


class TestIndicadoresDeShap:
    def test_fraccion_de_predicciones_explicadas(self):
        assert explained_fraction(30, 62) == pytest.approx(30 / 62)

    def test_no_admite_cero_predicciones_ni_mas_explicadas_que_el_total(self):
        with pytest.raises(ValueError):
            explained_fraction(1, 0)
        with pytest.raises(ValueError):
            explained_fraction(5, 3)

    def test_importancia_por_predictor_ordenada(self):
        # (n=2 observaciones, V=3 variables, T=2 pasos)
        valores = np.array([
            [[1.0, -1.0], [3.0, 3.0], [0.5, 0.5]],
            [[-1.0, 1.0], [-3.0, 3.0], [0.5, -0.5]],
        ])

        r = shap_predictor_importance(valores, ["precipitacion", "ndvi", "radiacion"])

        assert r["importancia"] == pytest.approx({"precipitacion": 2.0, "ndvi": 6.0, "radiacion": 1.0})
        assert r["orden"] == ["ndvi", "precipitacion", "radiacion"]

    def test_exige_un_nombre_por_variable(self):
        with pytest.raises(ValueError, match="nombres"):
            shap_predictor_importance(np.zeros((2, 3, 2)), ["a", "b"])

    def test_la_suma_de_las_importancias_coincide_con_la_de_las_fases(self):
        from src.explainability.shap_aggregation import aggregate_shap_by_phase

        rng = np.random.default_rng(0)
        valores = rng.normal(size=(7, 4, 5))
        fases = ["f1", "f2", "f3", "f4", "f5"]

        por_variable = shap_predictor_importance(valores, list("abcd"))["importancia"]
        por_fase = aggregate_shap_by_phase(valores, fases)

        assert sum(por_variable.values()) == pytest.approx(np.mean([sum(o.values()) for o in por_fase]))
