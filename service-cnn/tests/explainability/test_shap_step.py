"""Pruebas del paso de SHAP con verificación de precisión local y contingencia (tarea T6).

Plan (4.15): DeepExplainer es el explicador principal; si su precisión local falla,
se migra a KernelExplainer, que cumple la propiedad de eficiencia por construcción.
El fondo de referencia sale solo de las campañas de entrenamiento y es reproducible.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.explainability import shap_step
from src.explainability.shap_explainer import LocalAccuracyError, SHAPExplanation
from src.explainability.shap_step import (
    campaign_phase_contribution,
    explain_with_contingency,
    explain_with_kernel_explainer,
    sample_background,
)
from src.models.cnn1d import CNN1DConfig, QuinuaYieldCNN1D


@pytest.fixture(scope="module")
def modelo():
    torch.manual_seed(0)
    config = CNN1DConfig(n_variables=3, n_pasos_temporales=4, n_covariables=2)
    m = QuinuaYieldCNN1D(config)
    m.eval()
    return m, config


def _entradas(config, n, semilla=0):
    g = torch.Generator().manual_seed(semilla)
    return (
        torch.randn(n, config.n_variables, config.n_pasos_temporales, generator=g),
        torch.randn(n, config.n_covariables, generator=g),
    )


class TestSampleBackground:
    def test_toma_a_lo_sumo_n_filas_y_son_filas_del_original(self):
        x = torch.arange(40, dtype=torch.float32).reshape(10, 2, 2)
        z = torch.arange(20, dtype=torch.float32).reshape(10, 2)

        xb, zb = sample_background(x, z, n_max=4, semilla=0)

        assert xb.shape[0] == zb.shape[0] == 4
        filas = {tuple(f.flatten().tolist()) for f in x}
        assert all(tuple(f.flatten().tolist()) in filas for f in xb)

    def test_es_reproducible_con_la_misma_semilla_y_distinto_con_otra(self):
        x = torch.randn(30, 2, 3)
        z = torch.randn(30, 2)

        a, _ = sample_background(x, z, 5, semilla=1)
        b, _ = sample_background(x, z, 5, semilla=1)
        c, _ = sample_background(x, z, 5, semilla=2)

        assert torch.equal(a, b) and not torch.equal(a, c)

    def test_si_hay_menos_filas_que_el_maximo_usa_todas(self):
        xb, zb = sample_background(torch.randn(3, 2, 2), torch.randn(3, 1), n_max=100, semilla=0)

        assert xb.shape[0] == zb.shape[0] == 3


class TestKernelExplainer:
    def test_devuelve_las_mismas_formas_que_deepexplainer(self, modelo):
        m, config = modelo
        xb, zb = _entradas(config, 6, 1)
        xp, zp = _entradas(config, 3, 2)

        e = explain_with_kernel_explainer(m, xb, zb, xp, zp)

        assert isinstance(e, SHAPExplanation)
        assert e.valores_secuenciales.shape == (3, config.n_variables, config.n_pasos_temporales)
        assert e.valores_covariables.shape == (3, config.n_covariables)
        assert e.valor_base.shape == (3,)

    def test_cumple_la_propiedad_de_eficiencia(self, modelo):
        """suma de SHAP + valor base = predicción del modelo, por construcción."""
        m, config = modelo
        xb, zb = _entradas(config, 6, 1)
        xp, zp = _entradas(config, 3, 2)

        e = explain_with_kernel_explainer(m, xb, zb, xp, zp)

        with torch.no_grad():
            pred = m(xp, zp).squeeze(1).numpy()
        reconstruida = e.valor_base + e.valores_secuenciales.sum(axis=(1, 2)) + e.valores_covariables.sum(axis=1)
        np.testing.assert_allclose(reconstruida, pred, atol=1e-3)


class TestKernelSinRegularizacion:
    """Con la regularización L1 automática de `shap` (`l1_reg="auto"`, activa cuando hay
    muchas variables y pocas muestras) KernelExplainer encoge las atribuciones: en el
    piloto técnico la suma de |SHAP| quedó a 1/2,2 de la de DeepExplainer y la correlación
    entre ambos fue 0,87. Con `l1_reg=0` la correlación sube a 0,95. La contingencia debe
    ser una estimación fiel de Shapley, así que no se regulariza."""

    def test_pasa_l1_reg_cero_a_shap_values(self, modelo, monkeypatch):
        m, config = modelo
        xb, zb = _entradas(config, 6, 1)
        xp, zp = _entradas(config, 2, 2)
        capturado = {}

        class FalsoKernel:
            def __init__(self, funcion, fondo):
                self.expected_value = 0.0
                self.n = fondo.shape[1]

            def shap_values(self, datos, **kwargs):
                capturado.update(kwargs)
                return np.zeros((len(datos), self.n))

        monkeypatch.setattr(shap_step.shap, "KernelExplainer", FalsoKernel)

        explain_with_kernel_explainer(m, xb, zb, xp, zp)

        assert capturado["l1_reg"] == 0


class TestContingencia:
    def test_usa_deepexplainer_si_su_precision_local_es_suficiente(self, modelo):
        m, config = modelo
        xb, zb = _entradas(config, 8, 1)
        xp, zp = _entradas(config, 3, 2)

        r = explain_with_contingency(m, xb, zb, xp, zp, tolerancia_fraccion=0.5)

        assert r.variante == "deep_explainer"

    def test_migra_a_kernelexplainer_si_deepexplainer_falla_la_precision_local(self, modelo, monkeypatch):
        m, config = modelo
        xb, zb = _entradas(config, 8, 1)
        xp, zp = _entradas(config, 3, 2)
        original = shap_step.explain_with_deep_explainer

        def estropeado(*args, **kwargs):
            e = original(*args, **kwargs)
            return SHAPExplanation(e.valores_secuenciales + 10.0, e.valores_covariables, e.valor_base)

        monkeypatch.setattr(shap_step, "explain_with_deep_explainer", estropeado)

        r = explain_with_contingency(m, xb, zb, xp, zp, tolerancia_fraccion=0.05)

        assert r.variante == "kernel_explainer"
        assert r.error_deep_explainer > r.tolerancia

    def test_si_tampoco_kernelexplainer_cumple_falla_con_un_error_claro(self, modelo, monkeypatch):
        m, config = modelo
        xb, zb = _entradas(config, 8, 1)
        xp, zp = _entradas(config, 3, 2)

        def estropeado(*args, **kwargs):
            return SHAPExplanation(np.full((3, 3, 4), 10.0), np.zeros((3, 2)), np.zeros(3))

        monkeypatch.setattr(shap_step, "explain_with_deep_explainer", estropeado)
        monkeypatch.setattr(shap_step, "explain_with_kernel_explainer", estropeado)

        with pytest.raises(LocalAccuracyError):
            explain_with_contingency(m, xb, zb, xp, zp, tolerancia_fraccion=0.05)

    def test_la_tolerancia_es_una_fraccion_de_la_desviacion_de_la_salida(self, modelo):
        m, config = modelo
        xb, zb = _entradas(config, 8, 1)
        xp, zp = _entradas(config, 3, 2)
        with torch.no_grad():
            desviacion = float(m(xb, zb).std())

        r = explain_with_contingency(m, xb, zb, xp, zp, tolerancia_fraccion=0.2)

        assert r.tolerancia == pytest.approx(0.2 * desviacion)


class TestContribucionPorFase:
    def test_promedia_la_contribucion_absoluta_de_las_observaciones(self):
        valores = np.zeros((2, 1, 4))
        valores[0, 0] = [1.0, -1.0, 2.0, 0.0]
        valores[1, 0] = [3.0, -3.0, 0.0, 0.0]
        e = SHAPExplanation(valores, np.zeros((2, 0)), np.zeros(2))

        c = campaign_phase_contribution(e, ["a", "a", "b", "b"])

        assert c == {"a": pytest.approx((2.0 + 6.0) / 2), "b": pytest.approx((2.0 + 0.0) / 2)}
