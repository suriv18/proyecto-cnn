"""Inferencia de la CNN-1D serializada (src/models/inferencia.py), sin PyTorch."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.models.inferencia import EPS_BATCHNORM, LineaBase, ModeloCNN1D
from tests.helpers_modelo import K, T, V, crear_artefacto


@pytest.fixture()
def modelo(tmp_path) -> ModeloCNN1D:
    return ModeloCNN1D.cargar(crear_artefacto(tmp_path / "m"))


def _conv_con_bucles(x, peso, sesgo):
    """Convolución «same» escrita con bucles, independiente de la versión con einsum."""
    n, c, t = x.shape
    o, _, k = peso.shape
    xp = np.pad(x, ((0, 0), (0, 0), (k // 2, k // 2)))
    salida = np.zeros((n, o, t))
    for i in range(n):
        for j in range(o):
            for pos in range(t):
                salida[i, j, pos] = sesgo[j] + sum(peso[j, ch, kk] * xp[i, ch, pos + kk] for ch in range(c) for kk in range(k))
    return salida


def _referencia(p, x_z, z_z):
    def bn(h, prefijo):
        return (h - p[f"{prefijo}.running_mean"][None, :, None]) / np.sqrt(p[f"{prefijo}.running_var"][None, :, None] + EPS_BATCHNORM) \
            * p[f"{prefijo}.weight"][None, :, None] + p[f"{prefijo}.bias"][None, :, None]

    h = np.maximum(bn(_conv_con_bucles(x_z, p["bloque_convolucional.0.weight"], p["bloque_convolucional.0.bias"]), "bloque_convolucional.1"), 0)
    h = np.maximum(bn(_conv_con_bucles(h, p["bloque_convolucional.4.weight"], p["bloque_convolucional.4.bias"]), "bloque_convolucional.5"), 0)
    r = np.concatenate([h.mean(axis=2), z_z], axis=1)
    d = np.maximum(r @ p["cabeza_densa.0.weight"].T + p["cabeza_densa.0.bias"], 0)
    return (d @ p["cabeza_densa.2.weight"].T + p["cabeza_densa.2.bias"])[:, 0]


def test_el_forward_coincide_con_una_implementacion_independiente_con_bucles(modelo):
    rng = np.random.default_rng(1)
    x = rng.normal(10, 4, (3, V, T))
    z = np.column_stack([rng.normal(3800, 300, 3), rng.uniform(0, 0.3, 3)])
    x_z = (x - modelo.media) / modelo.desviacion
    z_z = (z - modelo.z_media) / modelo.z_desviacion
    esperado = np.stack([_referencia(p, x_z, z_z) * modelo.y_desviacion[i] + modelo.y_media[i] for i, p in enumerate(modelo.pesos)])
    np.testing.assert_allclose(modelo.predecir_anomalia(x, z), esperado, rtol=1e-10)


def test_devuelve_una_fila_por_semilla_y_una_columna_por_observacion(modelo):
    salida = modelo.predecir_anomalia(np.full((4, V, T), 10.0), np.tile([3800.0, 0.1], (4, 1)))
    assert salida.shape == (3, 4)
    assert np.isfinite(salida).all()


def test_una_celda_vacia_se_imputa_con_la_mediana_del_entrenamiento(modelo):
    x = np.full((1, V, T), 12.0)
    z = np.array([[3800.0, 0.1]])
    con_vacio = x.copy()
    con_vacio[0, 2, 3] = np.nan
    imputado = x.copy()
    imputado[0, 2, 3] = modelo.mediana[2, 3]
    np.testing.assert_allclose(modelo.predecir_anomalia(con_vacio, z), modelo.predecir_anomalia(imputado, z))


def test_es_determinista(modelo):
    x, z = np.full((2, V, T), 9.0), np.tile([3700.0, 0.2], (2, 1))
    np.testing.assert_array_equal(modelo.predecir_anomalia(x, z), modelo.predecir_anomalia(x, z))


@pytest.mark.parametrize("forma_x, forma_z", [((1, V - 1, T), (1, K)), ((1, V, T + 1), (1, K)), ((1, V, T), (1, K + 1)), ((2, V, T), (1, K))])
def test_rechaza_formas_que_no_corresponden_al_modelo(modelo, forma_x, forma_z):
    with pytest.raises(ValueError):
        modelo.predecir_anomalia(np.zeros(forma_x), np.zeros(forma_z))


def test_rechaza_un_formato_desconocido(tmp_path):
    carpeta = crear_artefacto(tmp_path / "m")
    tarjeta = json.loads((carpeta / "modelo.json").read_text(encoding="utf-8"))
    tarjeta["formato"] = 99
    (carpeta / "modelo.json").write_text(json.dumps(tarjeta), encoding="utf-8")
    with pytest.raises(ValueError, match="Formato"):
        ModeloCNN1D.cargar(carpeta)


def test_la_version_sale_de_la_huella_de_los_pesos(modelo):
    assert modelo.version == "cnn1d-abababab"


def test_la_linea_base_extrapola_la_tendencia_y_no_inventa_provincias(modelo):
    base: LineaBase = modelo.linea_base
    assert base.predecir("AAA", 2016) == pytest.approx(900.0)
    assert base.predecir("AAA", 2020) == pytest.approx(940.0)
    assert base.predecir("BBB", 2020) == pytest.approx(2000.0)  # respaldo: media constante
    assert base.predecir("NO EXISTE", 2020) is None
