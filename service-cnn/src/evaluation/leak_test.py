"""Prueba de fuga por permutación de las etiquetas de entrenamiento (tarea T8, experimento P1).

Se permutan las etiquetas de ENTRENAMIENTO de un pliego (se rompe la relación entre
entradas y rendimiento) y se evalúa cada modelo en la campaña de prueba. Sin relación
aprendible, el R² fuera de muestra debe quedar en torno a 0 o negativo. Si un modelo
sigue prediciendo bien, la verdad de la prueba le está llegando por otro camino, p. ej.
un ajuste o un preprocesamiento hechos con filas de prueba.

Se trabaja en la representación `nivel`, porque en `anomalia` la línea base provincial
aporta información legítima de la identidad de la provincia.

Limitación declarada: NO detecta una fuga entre entradas que sea coherente con las
etiquetas de entrenamiento. Por eso conviene acompañarla de un control positivo (un
"modelo" que vea la verdad de la prueba debe ser señalado) y de las pruebas de
construcción de pliegos (el entrenamiento es siempre anterior a la prueba).
"""
from __future__ import annotations

import dataclasses
from typing import Mapping

import numpy as np
import pandas as pd

from src.evaluation.experiment_runner import MODELOS, Adaptador, build_adapter
from src.evaluation.fold_builder import FoldData
from src.evaluation.metrics import r2_fuera_de_muestra


def permuted_training_fold(fold: FoldData, semilla: int) -> FoldData:
    """Copia del pliego con las etiquetas de entrenamiento permutadas; la prueba no se toca."""
    permutacion = np.random.default_rng(semilla).permutation(len(fold.y_train))
    return dataclasses.replace(fold, y_train=fold.y_train[permutacion])


def run_permutation_test(
    fold: FoldData,
    hiperparametros: Mapping[str, dict],
    modelos: tuple[str, ...] = MODELOS,
    n_permutaciones: int = 10,
    semilla: int = 0,
    umbral_r2: float = 0.05,
    epocas_cnn: int = 30,
    adaptadores: Mapping[str, Adaptador] | None = None,
) -> pd.DataFrame:
    """R² fuera de muestra de cada modelo entrenado con etiquetas permutadas.

    Args:
        hiperparametros: `{modelo: hp}` fijos (la prueba no ajusta; solo busca fugas).
        n_permutaciones: cuántas permutaciones independientes. El criterio es la MEDIANA
            de sus R², no el máximo: con pocas filas de prueba el R² de un modelo sin
            información oscila mucho (en un pliego de 8 observaciones entre -2,4 y +0,48),
            y el máximo de varias permutaciones supera cualquier umbral por azar.
        umbral_r2: un modelo "pasa" si la mediana de su R² es menor o igual a este valor.
        adaptadores: permite inyectar un adaptador propio (control positivo).

    Returns:
        DataFrame con `modelo`, `r2_mediano` (el criterio), `r2_maximo` (informativo),
        `r2_por_permutacion` y `pasa`.
    """
    filas = []
    for nombre in modelos:
        adaptador = (adaptadores or {}).get(nombre) or build_adapter(nombre, epocas_cnn)
        r2s = []
        for i in range(n_permutaciones):
            permutado = permuted_training_fold(fold, semilla=semilla * 1000 + i)
            pred = adaptador.ajustar_predecir(hiperparametros.get(nombre, {}), permutado, semilla, "nivel")
            r2s.append(r2_fuera_de_muestra(permutado.y_test, pred))
        filas.append({
            "modelo": nombre, "r2_mediano": float(np.median(r2s)), "r2_maximo": float(max(r2s)),
            "r2_por_permutacion": [round(float(r), 3) for r in r2s],
            "pasa": bool(np.median(r2s) <= umbral_r2),
        })
    return pd.DataFrame(filas)
