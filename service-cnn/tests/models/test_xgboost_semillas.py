"""Las semillas de XGBoost deben ser réplicas reales (4.12.4).

Con los valores por defecto (`subsample=1`, `colsample_bytree=1`) XGBoost es determinista y
las 10 semillas serían idénticas. El espacio de búsqueda debe incluir submuestreo.
"""
from __future__ import annotations

import numpy as np
import yaml

from src.models.classic_models import build_classic_model


def _espacio_xgboost() -> dict:
    return yaml.safe_load(open("configs/hyperparams.yaml", encoding="utf-8"))["xgboost"]


def test_el_espacio_de_xgboost_incluye_submuestreo():
    espacio = _espacio_xgboost()

    for nombre in ("subsample", "colsample_bytree"):
        tipo, minimo, maximo = espacio[nombre]
        assert tipo == "float"
        assert 0.0 < minimo < maximo <= 1.0


def test_con_el_limite_inferior_del_submuestreo_las_semillas_dan_predicciones_distintas():
    espacio = _espacio_xgboost()
    hp = {"n_estimators": 30, "max_depth": 3,
          "subsample": espacio["subsample"][1], "colsample_bytree": espacio["colsample_bytree"][1]}
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 30))
    y = X[:, 0] + rng.normal(size=120)

    pred = [build_classic_model("xgboost", dict(hp), s).fit(X, y).predict(X[:5]) for s in range(3)]

    assert not np.allclose(pred[0], pred[1]) or not np.allclose(pred[0], pred[2])
