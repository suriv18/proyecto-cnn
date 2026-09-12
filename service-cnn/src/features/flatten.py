"""Aplanado del tensor V×T a vector tabular para los modelos clásicos.

Referencia: sección 4.3 (los algoritmos se aplican bajo condiciones
computacionales controladas sobre las mismas particiones) y Tabla 8. Los 4
modelos clásicos (Elastic-Net, Random Forest, XGBoost, SVR) no procesan
estructura secuencial nativamente, a diferencia de la CNN-1D; para que la
comparación algorítmica sea metodológicamente válida, deben recibir
exactamente la misma información que la CNN, aplanada a un vector.
"""
from __future__ import annotations

import numpy as np


def flatten_tensor(tensor: np.ndarray, covariables: np.ndarray) -> np.ndarray:
    """Aplana un tensor V×T y lo concatena con las covariables.

    Args:
        tensor: forma (V, T) — el tensor de una observación provincia-campaña
            (salida de `src/features/tensor_builder.build_observation_tensor`).
        covariables: vector 1D con las covariables estática y dinámica ya
            concatenadas (altitud media, proporción de superficie con quinua).

    Returns:
        Vector 1D de longitud V*T + len(covariables): primero el tensor
        aplanado en orden fila-mayor (variable por variable, cada una con
        todos sus pasos temporales consecutivos), luego las covariables.
        Los NaN del tensor (celdas sin observación, sección 4.10.3) se
        preservan — la imputación/estandarización es responsabilidad de un
        paso posterior del pipeline (sección 4.12.2), no de este aplanado.
    """
    return np.concatenate([tensor.flatten(order="C"), covariables])


def flattened_feature_names(
    orden_variables: list[str],
    orden_fases: list[str],
    nombres_covariables: list[str],
) -> list[str]:
    """Genera los nombres de columna correspondientes a `flatten_tensor`.

    El orden debe coincidir exactamente con `flatten_tensor` para que el
    diccionario de variables (sección 4.10.3) y las importancias de
    Random Forest / XGBoost (sección 4.15, comparación descriptiva de
    importancias) sean interpretables.
    """
    nombres_secuenciales = [
        f"{variable}__{fase}" for variable in orden_variables for fase in orden_fases
    ]
    return nombres_secuenciales + list(nombres_covariables)
