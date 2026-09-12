"""Pruebas del benchmark B3: regresión simple con NDVI máximo (Tabla 8).

B3 es la cota mínima del aporte espectral: una regresión lineal univariada
entre el NDVI máximo de la campaña y el rendimiento, ajustada exclusivamente
con campañas de entrenamiento (misma disciplina anti-fuga que B2 en
target_transform.py). No se optimiza — es una regla de referencia fija, no
uno de los 5 modelos ajustables (sección 4.12.3).
"""
import numpy as np
import pandas as pd
import pytest

from src.models.benchmarks import fit_ndvi_benchmark, predict_ndvi_benchmark


@pytest.fixture
def historia_entrenamiento() -> pd.DataFrame:
    """Relación lineal clara: rendimiento = 500 + 1000 * ndvi_max + ruido leve."""
    rng = np.random.default_rng(0)
    ndvi_max = np.linspace(0.3, 0.8, 20)
    rendimiento = 500 + 1000 * ndvi_max + rng.normal(scale=5, size=20)
    return pd.DataFrame({"ndvi_max": ndvi_max, "rendimiento_kg_ha": rendimiento})


class TestFitNdviBenchmark:
    def test_ajusta_una_regresion_lineal_razonable(self, historia_entrenamiento):
        benchmark = fit_ndvi_benchmark(historia_entrenamiento)
        assert benchmark.pendiente == pytest.approx(1000.0, rel=0.1)
        assert benchmark.intercepto == pytest.approx(500.0, rel=0.1)

    def test_falla_si_el_historial_esta_vacio(self):
        vacio = pd.DataFrame({"ndvi_max": [], "rendimiento_kg_ha": []})
        with pytest.raises(ValueError, match="vac[ií]o"):
            fit_ndvi_benchmark(vacio)

    def test_falla_si_solo_hay_una_observacion(self):
        """Una regresión lineal requiere al menos 2 puntos distintos para
        estimar pendiente e intercepto de forma no trivial."""
        una_fila = pd.DataFrame({"ndvi_max": [0.5], "rendimiento_kg_ha": [700.0]})
        with pytest.raises(ValueError, match="al menos 2"):
            fit_ndvi_benchmark(una_fila)


class TestPredictNdviBenchmark:
    def test_predice_usando_pendiente_e_intercepto(self, historia_entrenamiento):
        benchmark = fit_ndvi_benchmark(historia_entrenamiento)
        prediccion = predict_ndvi_benchmark(benchmark, ndvi_max=0.6)
        esperado = benchmark.intercepto + benchmark.pendiente * 0.6
        assert prediccion == pytest.approx(esperado)

    def test_acepta_un_array_de_valores_ndvi(self, historia_entrenamiento):
        benchmark = fit_ndvi_benchmark(historia_entrenamiento)
        predicciones = predict_ndvi_benchmark(
            benchmark, ndvi_max=np.array([0.4, 0.6, 0.8])
        )
        assert predicciones.shape == (3,)
        assert np.all(np.diff(predicciones) > 0)  # monotónico si pendiente > 0
