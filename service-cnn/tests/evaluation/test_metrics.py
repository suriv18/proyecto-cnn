"""Pruebas de las métricas de exactitud predictiva (Tabla 9, sección 4.13).

RMSE, MAE, rRMSE (RMSE relativo a la media observada) y R² fuera de muestra
(puede ser negativo cuando el modelo es peor que predecir la media).
"""
import numpy as np
import pytest

from src.evaluation.metrics import mae, r2_fuera_de_muestra, rmse, rrmse


class TestRmse:
    def test_rmse_es_cero_si_prediccion_es_perfecta(self):
        y_real = np.array([100.0, 200.0, 300.0])
        assert rmse(y_real, y_real) == pytest.approx(0.0)

    def test_rmse_penaliza_errores_grandes_mas_que_mae(self):
        y_real = np.array([100.0, 100.0])
        y_pred_error_uniforme = np.array([110.0, 90.0])  # error de 10 en ambos
        y_pred_error_concentrado = np.array([120.0, 100.0])  # error de 20 en uno
        # Mismo error absoluto total (20), pero RMSE penaliza más el
        # concentrado por elevar al cuadrado.
        assert rmse(y_real, y_pred_error_concentrado) > rmse(
            y_real, y_pred_error_uniforme
        )

    def test_valor_conocido(self):
        y_real = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        # errores: 1, 0, 1 -> MSE = (1+0+1)/3 = 0.667 -> RMSE = 0.8165
        assert rmse(y_real, y_pred) == pytest.approx(0.8165, abs=1e-3)


class TestMae:
    def test_mae_es_cero_si_prediccion_es_perfecta(self):
        y_real = np.array([100.0, 200.0])
        assert mae(y_real, y_real) == pytest.approx(0.0)

    def test_valor_conocido(self):
        y_real = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        # errores absolutos: 1, 0, 1 -> media = 0.667
        assert mae(y_real, y_pred) == pytest.approx(0.6667, abs=1e-3)


class TestRrmse:
    def test_rrmse_expresa_rmse_como_porcentaje_de_la_media_observada(self):
        y_real = np.array([100.0, 100.0, 100.0])
        y_pred = np.array([110.0, 110.0, 110.0])
        # RMSE = 10, media observada = 100 -> rRMSE = 10%
        assert rrmse(y_real, y_pred) == pytest.approx(10.0)

    def test_falla_si_la_media_observada_es_cercana_a_cero(self):
        """Tabla 9: rRMSE no se calculará sobre anomalías con media cercana a
        cero — debe fallar explícitamente en vez de devolver un valor
        engañosamente grande o infinito."""
        y_real = np.array([0.001, -0.001, 0.0005])
        y_pred = np.array([1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="cercana a cero"):
            rrmse(y_real, y_pred)


class TestR2FueraDeMuestra:
    def test_r2_es_uno_si_prediccion_es_perfecta(self):
        y_real = np.array([1.0, 2.0, 3.0, 4.0])
        assert r2_fuera_de_muestra(y_real, y_real) == pytest.approx(1.0)

    def test_r2_puede_ser_negativo_fuera_de_muestra(self):
        """Sección 2.4.10: R² fuera de muestra compara el error del modelo con
        la variabilidad del conjunto de prueba, por lo que puede ser negativo
        si el modelo predice peor que la media del conjunto de prueba."""
        y_real = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred_muy_mala = np.array([100.0, -100.0, 200.0, -200.0])
        assert r2_fuera_de_muestra(y_real, y_pred_muy_mala) < 0

    def test_r2_es_cero_si_prediccion_es_la_media(self):
        y_real = np.array([1.0, 2.0, 3.0, 4.0])
        media = np.full_like(y_real, y_real.mean())
        assert r2_fuera_de_muestra(y_real, media) == pytest.approx(0.0)
