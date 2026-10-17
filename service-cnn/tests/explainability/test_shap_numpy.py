"""SHAP en NumPy (src/explainability/shap_numpy.py), sin PyTorch ni `shap`."""
from __future__ import annotations

import numpy as np
import pytest

from src.explainability.shap_numpy import _conv1d_transpuesta, _multiplicador_relu, deeplift_semilla
from src.models.inferencia import ModeloCNN1D, _conv1d
from tests.helpers_modelo import K, T, V, crear_artefacto


@pytest.fixture()
def modelo(tmp_path) -> ModeloCNN1D:
    return ModeloCNN1D.cargar(crear_artefacto(tmp_path / "m"))


def _entradas(modelo, n=4, semilla=3):
    rng = np.random.default_rng(semilla)
    x = rng.normal(10, 4, (n, V, T))
    z = np.column_stack([rng.normal(3800, 300, n), rng.uniform(0, 0.3, n)])
    return x, z


def test_la_suma_de_los_valores_shap_reproduce_la_salida_menos_el_valor_base(modelo):
    """Precisión local (Lundberg y Lee, 2017): se cumple por construcción en DeepLIFT con reescalado."""
    x, z = _entradas(modelo)
    x_z, z_z = modelo._estandarizar(x, z)
    for p in modelo.pesos:
        r = deeplift_semilla(p, x_z, z_z, modelo.fondo_x, modelo.fondo_z)
        reconstruida = r.base + r.tensor.sum(axis=(1, 2)) + r.covariables.sum(axis=1)
        np.testing.assert_allclose(reconstruida, r.salida, atol=1e-9)


def test_la_convolucion_transpuesta_es_la_adjunta_de_la_convolucion():
    rng = np.random.default_rng(0)
    peso = rng.normal(0, 1, (4, 3, 3))
    x = rng.normal(0, 1, (2, 3, 5))
    g = rng.normal(0, 1, (2, 4, 5))
    avance = _conv1d(x, peso, np.zeros(4))
    atras = _conv1d_transpuesta(g, peso)
    assert np.sum(avance * g) == pytest.approx(np.sum(x * atras))


def test_el_multiplicador_relu_usa_la_derivada_si_la_entrada_y_la_referencia_coinciden():
    z = np.array([2.0, -1.0])
    np.testing.assert_array_equal(_multiplicador_relu(z, z), [1.0, 0.0])
    # con diferencia: cociente de diferencias de la ReLU
    assert _multiplicador_relu(np.array([3.0]), np.array([-1.0]))[0] == pytest.approx(3.0 / 4.0)
    assert _multiplicador_relu(np.array([-2.0]), np.array([-1.0]))[0] == pytest.approx(0.0)


def test_explicar_devuelve_una_matriz_por_observacion_en_kg_ha_promediada_entre_semillas(modelo):
    x, z = _entradas(modelo, n=3)
    valores = modelo.explicar(x, z)
    assert valores.shape == (3, V, T)
    assert np.isfinite(valores).all()
    x_z, z_z = modelo._estandarizar(x, z)
    manual = np.mean(
        [deeplift_semilla(p, x_z, z_z, modelo.fondo_x, modelo.fondo_z).tensor * modelo.y_desviacion[i] for i, p in enumerate(modelo.pesos)],
        axis=0,
    )
    np.testing.assert_allclose(valores, manual)


def test_una_observacion_igual_a_todo_el_fondo_no_tiene_contribuciones(modelo):
    """Si la entrada es idéntica a cada referencia, (x - r) = 0 y todos los valores SHAP son 0."""
    fila = np.zeros((1, V, T))
    modelo_fijo = ModeloCNN1D(**{**modelo.__dict__, "fondo_x": np.repeat(modelo._estandarizar(fila + 5, np.array([[3800.0, 0.1]]))[0], 3, axis=0),
                                 "fondo_z": np.repeat(modelo._estandarizar(fila + 5, np.array([[3800.0, 0.1]]))[1], 3, axis=0)})
    np.testing.assert_allclose(modelo_fijo.explicar(fila + 5, np.array([[3800.0, 0.1]])), 0.0, atol=1e-12)


def test_sin_fondo_de_referencia_no_se_puede_explicar(tmp_path):
    sin_fondo = ModeloCNN1D.cargar(crear_artefacto(tmp_path / "s", con_fondo=False))
    assert not sin_fondo.puede_explicar
    with pytest.raises(RuntimeError, match="fondo"):
        sin_fondo.explicar(np.zeros((1, V, T)), np.zeros((1, K)))


def test_rechaza_formas_que_no_corresponden(modelo):
    with pytest.raises(ValueError):
        modelo.explicar(np.zeros((1, V, T + 1)), np.zeros((1, K)))
