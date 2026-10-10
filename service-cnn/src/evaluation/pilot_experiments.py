"""Reglas de decisión del ensayo piloto (tarea T7; experimentos P4, P5, P6, P7 y P9).

Cada función implementa una regla preespecificada en `docs/crisp/F3-plan-del-piloto.md`,
fijada antes de ejecutar. Los empates se resuelven siempre hacia lo más simple (nivel,
media provincial), y ninguna regla usa campañas externas: el llamador solo pasa pliegos
del piloto (ver `pilot_guard`).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.evaluation.fold_builder import FoldData

TOLERANCIA_DE_EMPATE = 0.02  # diferencias relativas menores a 2 % se consideran empate


def decide_representation(
    rmse_nivel: pd.DataFrame, rmse_anomalia: pd.DataFrame, tolerancia_relativa: float = TOLERANCIA_DE_EMPATE
) -> dict:
    """P4: nivel o anomalía según el RMSE en kg/ha promediado entre modelos y pliegos.

    Se elige la anomalía solo si reduce el error en más de `tolerancia_relativa`; un
    empate se resuelve a favor del nivel (más simple). Reciben `modelo`, `campana_prueba`
    y `rmse` de los modelos ajustables (sin benchmarks), todos con el mismo peso.
    """
    media_nivel = float(rmse_nivel["rmse"].mean())
    media_anomalia = float(rmse_anomalia["rmse"].mean())
    relativa = (media_anomalia - media_nivel) / media_nivel
    decision = "anomalia" if (abs(relativa) >= tolerancia_relativa and media_anomalia < media_nivel) else "nivel"
    return {
        "decision": decision, "media_nivel": media_nivel, "media_anomalia": media_anomalia,
        "diferencia_relativa": float(relativa), "tolerancia_relativa": tolerancia_relativa,
    }


def baseline_forms(fold: FoldData) -> pd.DataFrame:
    """P5: RMSE en la campaña de prueba de tres formas de la línea base provincial B2.

    - `media`: media histórica de la provincia.
    - `lineal_min4`: tendencia lineal por campaña si hay al menos 4 puntos; si no, la media.
    - `media_ultimas3`: media de las 3 campañas más recientes de la provincia.
    Una provincia sin historia usa la media de entrenamiento. Solo usa el entrenamiento.
    """
    historia = pd.DataFrame({"p": fold.provincia_train, "c": fold.campana_train, "y": fold.y_train})
    media_global = float(fold.y_train.mean())
    predicciones = {"media": [], "lineal_min4": [], "media_ultimas3": []}
    for provincia, campana in zip(fold.provincia_test, fold.campana_test):
        h = historia[historia["p"] == provincia].sort_values("c")
        if h.empty:
            for lista in predicciones.values():
                lista.append(media_global)
            continue
        predicciones["media"].append(float(h["y"].mean()))
        if len(h) >= 4:
            pendiente, intercepto = np.polyfit(h["c"].to_numpy(dtype=float), h["y"].to_numpy(dtype=float), deg=1)
            predicciones["lineal_min4"].append(float(intercepto + pendiente * campana))
        else:
            predicciones["lineal_min4"].append(float(h["y"].mean()))
        predicciones["media_ultimas3"].append(float(h["y"].tail(3).mean()))
    return pd.DataFrame([
        {"forma": forma, "campana_prueba": fold.campana_prueba,
         "rmse": float(np.sqrt(np.mean((np.asarray(pred) - fold.y_test) ** 2)))}
        for forma, pred in predicciones.items()
    ])


def decide_baseline_form(tabla: pd.DataFrame, tolerancia_relativa: float = TOLERANCIA_DE_EMPATE) -> dict:
    """P5: la forma de B2 con menor RMSE medio; un empate (< 2 %) se resuelve a favor de la media.

    Con 4-5 puntos por provincia la media es la opción más estable, así que otra forma
    debe mejorarla de forma clara para elegirse.
    """
    medias = tabla.groupby("forma")["rmse"].mean()
    mejor = str(medias.idxmin())
    if mejor != "media" and "media" in medias and (medias["media"] - medias[mejor]) / medias["media"] < tolerancia_relativa:
        mejor = "media"
    return {"decision": mejor, "rmse_medio_por_forma": {k: float(v) for k, v in medias.items()}}


def seed_variability(predicciones: pd.DataFrame) -> pd.DataFrame:
    """P6: dispersión del RMSE entre semillas, por modelo y campaña (solo modelos con varias semillas)."""
    por_semilla = (
        predicciones.assign(_e=(predicciones["y_pred"] - predicciones["y_real"]) ** 2)
        .groupby(["modelo", "campana_prueba", "semilla"])["_e"].mean().pipe(np.sqrt)
        .rename("rmse").reset_index()
    )
    filas = []
    for (modelo, campana), g in por_semilla.groupby(["modelo", "campana_prueba"]):
        v = g["rmse"].to_numpy()
        if len(v) < 2:
            continue
        filas.append({
            "modelo": modelo, "campana_prueba": campana, "n_semillas": len(v), "mediana": float(np.median(v)),
            "q1": float(np.percentile(v, 25)), "q3": float(np.percentile(v, 75)),
            "minimo": float(v.min()), "maximo": float(v.max()),
            # error estándar asintótico de la mediana: 1,2533 · s / √n
            "error_estandar_mediana": float(1.2533 * np.std(v, ddof=1) / np.sqrt(len(v))),
        })
    return pd.DataFrame(filas, columns=[
        "modelo", "campana_prueba", "n_semillas", "mediana", "q1", "q3", "minimo", "maximo", "error_estandar_mediana",
    ])


def max_epochs_from_best(mejores_epocas: list[int], factor: float = 1.5) -> int:
    """P7: épocas máximas = factor × percentil 95 de la mejor época observada, redondeado hacia arriba."""
    if len(mejores_epocas) == 0:
        raise ValueError("Se necesita al menos una mejor época observada para fijar las épocas máximas.")
    return int(math.ceil(factor * float(np.percentile(mejores_epocas, 95))))


def border_check(
    ensayos: pd.DataFrame, espacio: dict, fraccion_top: float = 0.2, ancho_borde: float = 0.1, umbral: float = 0.5
) -> pd.DataFrame:
    """P9: señala los hiperparámetros cuyos mejores ensayos se agrupan en un borde del espacio.

    Toma el `fraccion_top` mejor de los ensayos (menor `valor`) y, por parámetro, mira qué
    fracción cae en el `ancho_borde` inferior o superior del rango. Si es al menos `umbral`,
    el óptimo probablemente está fuera del espacio y habría que ampliarlo (para los cinco
    modelos por igual).
    """
    k = max(1, math.ceil(fraccion_top * len(ensayos)))
    mejores = ensayos.nsmallest(k, "valor")
    filas = []
    for parametro, (_, minimo, maximo) in espacio.items():
        posicion = (mejores[parametro].to_numpy(dtype=float) - minimo) / (maximo - minimo)
        en_inferior, en_superior = float(np.mean(posicion <= ancho_borde)), float(np.mean(posicion >= 1 - ancho_borde))
        lado = "ninguno"
        if en_superior >= umbral and en_superior >= en_inferior:
            lado = "superior"
        elif en_inferior >= umbral:
            lado = "inferior"
        filas.append({"parametro": parametro, "lado": lado, "fraccion_en_borde": max(en_inferior, en_superior), "n_ensayos_evaluados": k})
    return pd.DataFrame(filas)
