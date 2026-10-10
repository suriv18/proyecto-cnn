"""Insumos de HE2 por campaña de prueba: contribución SHAP por fase y aumento de RMSE por oclusión.

Sobre datos sintéticos pequeños: se verifica la forma de las salidas, que cada campaña de
prueba se explica con un modelo entrenado solo con campañas anteriores y la reproducibilidad.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.fold_builder import build_folds
from src.evaluation.he2_runner import collect_he2_inputs

FASES = ["emergencia", "desarrollo_vegetativo", "floracion", "llenado_grano", "madurez"]
HP = {"filtros_capa1": 6, "filtros_capa2": 6, "dropout": 0.005, "tasa_aprendizaje": 0.01}


@pytest.fixture(scope="module")
def pliegos():
    rng = np.random.default_rng(0)
    provincias = [f"P{i}" for i in range(40)]  # con pocos datos la red no aprende la señal
    obs = pd.DataFrame([(p, c) for c in range(2016, 2020) for p in provincias], columns=["provincia_id", "campana_id"])
    X = rng.normal(5.0, 2.0, size=(len(obs), 6, 5))
    Z = rng.normal(0.0, 1.0, size=(len(obs), 2))
    nivel = {p: 800.0 + 120.0 * i for i, p in enumerate(provincias)}
    # la fase 'floracion' (paso 2) y la variable 4 gobiernan el rendimiento
    obs["rendimiento_kg_ha"] = obs.provincia_id.map(nivel) + 60.0 * X[:, 4, 2] + rng.normal(0, 10, len(obs))
    return build_folds(obs, X, Z, 2, range(2016, 2020))


def _recolectar(pliegos, **extra):
    base = dict(hp_cnn=HP, representacion="anomalia", fases=FASES, n_semillas=2, epocas_cnn=8,
                n_fondo=20, n_explicadas=8, semilla=0)
    base.update(extra)
    return collect_he2_inputs(pliegos, **base)


def test_devuelve_una_entrada_por_campana_de_prueba(pliegos):
    contribuciones, aumentos = _recolectar(pliegos)

    esperadas = {int(p.campana_prueba) for p in pliegos}
    assert set(contribuciones) == set(aumentos) == esperadas == {2018, 2019}


def test_cada_entrada_tiene_las_cinco_fases_con_valores_finitos(pliegos):
    contribuciones, aumentos = _recolectar(pliegos)

    for campana in contribuciones:
        assert set(contribuciones[campana]) == set(aumentos[campana]) == set(FASES)
        assert all(np.isfinite(v) and v >= 0.0 for v in contribuciones[campana].values())
        assert all(np.isfinite(v) for v in aumentos[campana].values())


def test_es_reproducible_con_la_misma_semilla(pliegos):
    a = _recolectar(pliegos)
    b = _recolectar(pliegos)

    assert a == b


def test_la_fase_que_gobierna_el_rendimiento_domina_la_contribucion_y_la_oclusion(pliegos):
    contribuciones, aumentos = _recolectar(pliegos, n_semillas=3, epocas_cnn=60, n_fondo=50, n_explicadas=30)

    assert {max(c, key=c.get) for c in contribuciones.values()} == {"floracion"}
    assert {max(a, key=a.get) for a in aumentos.values()} == {"floracion"}


def test_rechaza_fases_que_no_coinciden_con_los_pasos_temporales(pliegos):
    with pytest.raises(ValueError, match="fases"):
        _recolectar(pliegos, fases=FASES[:3])


def test_rechaza_una_lista_vacia_de_pliegos():
    with pytest.raises(ValueError, match="pliego"):
        _recolectar([])
