"""Los 4 modelos clásicos comparadores (Tabla 8, sección 4.12.3-4.12.4).

Elastic-Net, Random Forest, XGBoost, SVR reciben entrada tabular (el tensor
V×T ya aplanado por `src/features/flatten.py`) y comparten una interfaz
uniforme (`ClassicModel`) para que el módulo de tuning (Optuna, presupuesto
equiparado) los trate de forma consistente, sin importar la librería
subyacente de cada uno.

Random Forest y XGBoost son estocásticos y reciben una semilla explícita para
las 10 réplicas de la sección 4.12.4; Elastic-Net y SVR (con kernel
determinista) se ejecutan deterministamente — reciben la semilla por
uniformidad de interfaz, pero la ignoran.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from xgboost import XGBRegressor

MODELOS_DISPONIBLES = ("elastic_net", "random_forest", "xgboost", "svr")


class ClassicModel(Protocol):
    """Interfaz uniforme que exponen los 4 wrappers de esta módulo."""

    modelo_subyacente: Any

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ClassicModel": ...

    def predict(self, x: np.ndarray) -> np.ndarray: ...


@dataclass
class _SklearnCompatibleModel:
    """Wrapper delgado sobre cualquier estimador con API scikit-learn
    (`fit`/`predict`) — cubre ElasticNet, RandomForestRegressor, SVR y
    XGBRegressor, que ya comparten esa firma."""

    modelo_subyacente: Any

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_SklearnCompatibleModel":
        self.modelo_subyacente.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.modelo_subyacente.predict(x)


def build_classic_model(
    nombre_modelo: str, hiperparametros: dict, semilla: int
) -> ClassicModel:
    """Construye uno de los 4 modelos clásicos con hiperparámetros configurables.

    Args:
        nombre_modelo: uno de `MODELOS_DISPONIBLES`.
        hiperparametros: kwargs pasados directamente al constructor del
            estimador subyacente (el espacio de búsqueda de Optuna, sección
            4.12.3, se define en `src/models/tuning.py`, no aquí).
        semilla: usada por Random Forest y XGBoost (estocásticos, sección
            4.12.4); ignorada por Elastic-Net y SVR (deterministas).

    Returns:
        Un `ClassicModel` con `fit`/`predict` uniformes.

    Raises:
        ValueError: si `nombre_modelo` no está en `MODELOS_DISPONIBLES`.
    """
    if nombre_modelo == "elastic_net":
        estimador = ElasticNet(**hiperparametros)
    elif nombre_modelo == "random_forest":
        estimador = RandomForestRegressor(random_state=semilla, **hiperparametros)
    elif nombre_modelo == "xgboost":
        estimador = XGBRegressor(random_state=semilla, **hiperparametros)
    elif nombre_modelo == "svr":
        estimador = SVR(**hiperparametros)
    else:
        raise ValueError(
            f"Modelo '{nombre_modelo}' no reconocido. "
            f"Modelos disponibles: {MODELOS_DISPONIBLES}"
        )

    return _SklearnCompatibleModel(modelo_subyacente=estimador)
