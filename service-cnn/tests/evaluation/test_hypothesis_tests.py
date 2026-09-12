"""Pruebas de los contrastes estadísticos confirmatorios HE1a y HE1b.

Referencia: docs/Proyecto_Tesis_Maestria_UNMSM_WMSR.docx, secciones 3.3.1-3.3.2,
4.14.

HE1a: no inferioridad de CNN-1D frente a XGBoost. Para cada campaña externa f,
q_f = RMSE(CNN-1D,f)/RMSE(XGBoost,f) - 1, margen δ=0.05. Se sostiene la no
inferioridad si el límite superior unilateral del IC 95% de la mediana de q_f
es menor que 0.05.

HE1b: superioridad de CNN-1D frente a B2. Δ_f = RMSE(CNN-1D,f) - RMSE(B2,f).
Se sostiene si el límite superior unilateral del IC 95% de la mediana de Δ_f
es menor que cero.

Ambos usan remuestreo por bloques temporales (bootstrap) y Wilcoxon pareado
unilateral como contraste de apoyo, más el estimador de Hodges-Lehmann y la
correlación biserial de rangos para datos pareados.
"""
import numpy as np
import pytest

from src.evaluation.hypothesis_tests import (
    NonInferiorityResult,
    SuperiorityResult,
    evaluate_non_inferiority,
    evaluate_superiority,
)


class TestNonInferiority:
    def test_no_inferioridad_se_sostiene_cuando_cnn_es_claramente_mejor(self):
        """Si CNN-1D es sistemáticamente mejor (RMSE menor) que XGBoost en
        todas las campañas, q_f es negativo en todas -> no inferioridad clara."""
        rng = np.random.default_rng(0)
        rmse_xgboost = rng.uniform(500, 700, size=15)
        rmse_cnn = rmse_xgboost * rng.uniform(0.85, 0.95, size=15)  # CNN 5-15% mejor

        resultado = evaluate_non_inferiority(
            rmse_cnn, rmse_xgboost, margen=0.05, n_bootstrap=2000, semilla=0
        )

        assert isinstance(resultado, NonInferiorityResult)
        assert resultado.no_inferioridad_sostenida is True
        assert resultado.limite_superior_ic95 < 0.05

    def test_no_inferioridad_se_rechaza_cuando_cnn_es_claramente_peor(self):
        """Si CNN-1D es sistemáticamente mucho peor (RMSE 50% mayor), q_f es
        grande y positivo -> no inferioridad rechazada."""
        rng = np.random.default_rng(1)
        rmse_xgboost = rng.uniform(500, 700, size=15)
        rmse_cnn = rmse_xgboost * 1.5  # CNN consistentemente 50% peor

        resultado = evaluate_non_inferiority(
            rmse_cnn, rmse_xgboost, margen=0.05, n_bootstrap=2000, semilla=0
        )

        assert resultado.no_inferioridad_sostenida is False
        assert resultado.limite_superior_ic95 >= 0.05

    def test_incluye_wilcoxon_pareado_de_apoyo(self):
        rng = np.random.default_rng(0)
        rmse_xgboost = rng.uniform(500, 700, size=15)
        rmse_cnn = rmse_xgboost * 0.9

        resultado = evaluate_non_inferiority(
            rmse_cnn, rmse_xgboost, margen=0.05, n_bootstrap=500, semilla=0
        )

        assert 0.0 <= resultado.wilcoxon_p_valor <= 1.0

    def test_incluye_hodges_lehmann_y_correlacion_biserial(self):
        rng = np.random.default_rng(0)
        rmse_xgboost = rng.uniform(500, 700, size=15)
        rmse_cnn = rmse_xgboost * 0.9

        resultado = evaluate_non_inferiority(
            rmse_cnn, rmse_xgboost, margen=0.05, n_bootstrap=500, semilla=0
        )

        assert isinstance(resultado.hodges_lehmann, float)
        assert -1.0 <= resultado.correlacion_biserial_rangos <= 1.0

    def test_falla_si_los_arreglos_tienen_longitudes_distintas(self):
        with pytest.raises(ValueError, match="misma longitud"):
            evaluate_non_inferiority(
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0, 3.0]),
                margen=0.05,
                n_bootstrap=100,
                semilla=0,
            )

    def test_reproducible_con_la_misma_semilla(self):
        rng = np.random.default_rng(0)
        rmse_xgboost = rng.uniform(500, 700, size=10)
        rmse_cnn = rmse_xgboost * 0.9

        resultado_a = evaluate_non_inferiority(
            rmse_cnn, rmse_xgboost, margen=0.05, n_bootstrap=500, semilla=42
        )
        resultado_b = evaluate_non_inferiority(
            rmse_cnn, rmse_xgboost, margen=0.05, n_bootstrap=500, semilla=42
        )
        assert resultado_a.limite_superior_ic95 == resultado_b.limite_superior_ic95


class TestSuperiority:
    def test_superioridad_se_sostiene_cuando_cnn_es_claramente_mejor_que_b2(self):
        rng = np.random.default_rng(0)
        rmse_b2 = rng.uniform(600, 800, size=15)
        rmse_cnn = rmse_b2 - rng.uniform(50, 100, size=15)  # CNN sistemáticamente menor

        resultado = evaluate_superiority(rmse_cnn, rmse_b2, n_bootstrap=2000, semilla=0)

        assert isinstance(resultado, SuperiorityResult)
        assert resultado.superioridad_sostenida is True
        assert resultado.limite_superior_ic95 < 0.0

    def test_superioridad_se_rechaza_cuando_cnn_no_es_mejor_que_b2(self):
        rng = np.random.default_rng(1)
        rmse_b2 = rng.uniform(600, 800, size=15)
        rmse_cnn = rmse_b2 + rng.uniform(10, 50, size=15)  # CNN peor que B2

        resultado = evaluate_superiority(rmse_cnn, rmse_b2, n_bootstrap=2000, semilla=0)

        assert resultado.superioridad_sostenida is False

    def test_falla_si_los_arreglos_tienen_longitudes_distintas(self):
        with pytest.raises(ValueError, match="misma longitud"):
            evaluate_superiority(
                np.array([1.0, 2.0]), np.array([1.0]), n_bootstrap=100, semilla=0
            )
