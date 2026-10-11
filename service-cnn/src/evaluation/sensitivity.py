"""Análisis de sensibilidad sobre las campañas de entrenamiento (sección 4.14).

Se ejecutan con el mismo protocolo del piloto (misma representación, mismo presupuesto de ajuste y
mismas semillas) sobre pliegos de las campañas permitidas; nunca cargan campañas externas. Son
exploratorios: informan cuánto cambia la conclusión al alterar una decisión, no la deciden.

Variantes sobre el conjunto procesado:
  * `sin_transicion`: sin las provincias de transición altoandina-valle interandino.
  * `tensor_completo`: solo observaciones sin ninguna celda imputada.
  * `horizonte_temprano`: horizonte secundario al cierre de la floración (solo las tres primeras
    fases, emergencia, desarrollo vegetativo y floración).
Las sensibilidades que cambian el preprocesamiento (arrastre de NDVI, exclusión por altitud) se
ejecutan sobre otro conjunto procesado con la misma función `run_sensitivity`.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.evaluation.experiment_runner import rmse_per_campaign, run_fold
from src.evaluation.fold_builder import build_folds

VARIANTES = ("base", "sin_transicion", "tensor_completo", "horizonte_temprano")
N_FASES_HORIZONTE_TEMPRANO = 3


def apply_variant(
    nombre: str, obs: pd.DataFrame, X: np.ndarray, Z: np.ndarray, imputado: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Aplica una variante al conjunto (observaciones, tensor y covariables alineados por fila).

    Raises:
        ValueError: si la variante no está declarada en `VARIANTES`.
    """
    if nombre not in VARIANTES:
        raise ValueError(f"variante desconocida '{nombre}'; use una de {VARIANTES}.")
    obs = obs.reset_index(drop=True)
    if nombre == "base":
        return obs, X, Z
    if nombre == "horizonte_temprano":
        return obs, X[:, :, :N_FASES_HORIZONTE_TEMPRANO], Z
    if nombre == "sin_transicion":
        conservar = (obs["clasificacion"] != "transicion").to_numpy()
    else:  # tensor_completo
        conservar = ~np.asarray(imputado).any(axis=(1, 2))
    return obs[conservar].reset_index(drop=True), X[conservar], Z[conservar]


def predict_sensitivity(
    obs: pd.DataFrame, X: np.ndarray, Z: np.ndarray, espacios: dict, permitidas: Iterable[int],
    ventana_piloto: int, ventana_interna: int, n_trials: int, n_semillas: int, epocas_cnn: int,
    indice_ndvi_max: int, semilla: int, representacion: str = "anomalia",
) -> pd.DataFrame:
    """Predicciones (todos los modelos y benchmarks) de los pliegos de las campañas permitidas.

    Raises:
        CampanaExternaError: si `obs` contiene campañas que no están en `permitidas`.
    """
    pliegos = build_folds(obs, X, Z, ventana_piloto, list(permitidas))
    resultados = [
        run_fold(f, espacios, n_trials, n_semillas, ventana_interna, representacion, indice_ndvi_max, epocas_cnn, semilla)
        for f in pliegos
    ]
    return pd.concat([r.predicciones for r in resultados], ignore_index=True)


def run_sensitivity(*args, **kwargs) -> pd.DataFrame:
    """RMSE por modelo (incluidos los benchmarks) y campaña de prueba con el protocolo del piloto."""
    return rmse_per_campaign(predict_sensitivity(*args, **kwargs))


def summarize_sensitivity(tablas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Une las variantes y calcula, por modelo y campaña, la diferencia relativa frente a B2 y B4 y el cambio frente a la base."""
    filas = []
    for variante, t in tablas.items():
        for campana, g in t.groupby("campana_prueba"):
            rmse = g.set_index("modelo")["rmse"]
            for modelo, valor in rmse.items():
                filas.append({
                    "variante": variante, "modelo": modelo, "campana_prueba": campana, "rmse": float(valor),
                    "q_frente_a_b2": float(valor / rmse["B2"] - 1.0) if "B2" in rmse else np.nan,
                    "q_frente_a_b4": float(valor / rmse["B4"] - 1.0) if "B4" in rmse else np.nan,
                })
    r = pd.DataFrame(filas)
    if "base" in tablas:
        base = r[r["variante"] == "base"].set_index(["modelo", "campana_prueba"])["rmse"]
        llave = list(zip(r["modelo"], r["campana_prueba"]))
        r["cambio_vs_base_pct"] = [(v / base.get(k, np.nan) - 1.0) * 100.0 for v, k in zip(r["rmse"], llave)]
    return r


def error_by_stratum(
    predicciones: pd.DataFrame, obs: pd.DataFrame, columna: str = "prop_quinua", n_estratos: int = 3,
) -> pd.DataFrame:
    """RMSE por estrato (terciles por defecto) de una característica de la observación.

    Args:
        predicciones: columnas `provincia_id`, `campana_prueba`, `modelo`, `y_real`, `y_pred`.
        obs: observaciones con `provincia_id`, `campana_id` y `columna`.
    """
    nombres = {3: ["bajo", "medio", "alto"]}.get(n_estratos, [f"estrato_{i + 1}" for i in range(n_estratos)])
    datos = obs[["provincia_id", "campana_id", columna]].copy()
    datos["estrato"] = pd.qcut(datos[columna].rank(method="first"), n_estratos, labels=nombres)
    m = predicciones.merge(datos, left_on=["provincia_id", "campana_prueba"], right_on=["provincia_id", "campana_id"], how="inner")
    m["_e2"] = (m["y_pred"] - m["y_real"]) ** 2
    por = m.groupby(["modelo", "estrato"], observed=True).agg(rmse=("_e2", lambda v: float(np.sqrt(v.mean()))), n=("_e2", "size")).reset_index()
    return por
