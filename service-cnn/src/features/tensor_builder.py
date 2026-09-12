"""Construcción del tensor V×T y del diccionario de variables (sección 4.10.3).

Cada observación provincia-campaña se representa como un tensor V×T, donde V
es el número de predictores secuenciales (precipitación, temperatura máx/mín,
radiación, NDVI) y T el número de pasos temporales (fases fenológicas)
disponibles hasta el horizonte. La altitud media de la superficie agrícola
(covariable estática) y la proporción de superficie sembrada con quinua
(covariable dinámica provincia-campaña) se mantienen SEPARADAS de este tensor:
según el diseño de la tesis, se concatenan después del bloque convolucional de
la CNN-1D, no como parte de la secuencia de entrada.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_ROLES_VALIDOS = {"secuencial", "estatica", "dinamica"}


class ObservacionIncompletaError(ValueError):
    """Falta por completo una variable declarada en `orden_variables` — a
    diferencia de una celda puntual faltante (que se completa con NaN), la
    ausencia total de una variable indica un problema de extracción previo,
    no un hueco de cobertura a tratar en el paso de completitud (sección 4.7).
    """


def build_observation_tensor(
    series_agregadas: pd.DataFrame,
    orden_variables: list[str],
    orden_fases: list[str],
) -> np.ndarray:
    """Ensambla el tensor V×T de una única observación provincia-campaña.

    Args:
        series_agregadas: salida de `gee_series_builder.aggregate_by_phase`
            (posiblemente concatenada entre variables), con columnas
            `provincia_id`, `campana_id`, `variable`, `fase`, `valor_agregado`,
            ya filtrada a una sola provincia-campaña.
        orden_variables: orden fijo de las V filas del tensor (debe declararse
            una vez y mantenerse igual en todo el pipeline — el orden en sí no
            tiene significado agronómico, pero debe ser consistente entre
            entrenamiento y prueba).
        orden_fases: orden fijo de las T columnas del tensor (orden
            cronológico de las fases fenológicas disponibles hasta el
            horizonte).

    Returns:
        Array de numpy de forma (V, T). Una celda sin observación en
        `series_agregadas` (ej. por nubosidad puntual) se completa con NaN —
        el criterio de completitud mínima del 80% (sección 4.7) se evalúa
        aparte, sobre el tensor ya construido.

    Raises:
        ObservacionIncompletaError: si una variable de `orden_variables` no
            aparece en absoluto en `series_agregadas` — señal de un problema
            de extracción previo (ej. la fuente de esa variable no se
            consultó), no un hueco de cobertura puntual.
    """
    combinaciones_provincia_campana = series_agregadas[
        ["provincia_id", "campana_id"]
    ].drop_duplicates()
    if len(combinaciones_provincia_campana) > 1:
        raise ValueError(
            "build_observation_tensor espera datos de una sola "
            "provincia-campaña, pero recibió "
            f"{len(combinaciones_provincia_campana)} combinaciones distintas. "
            "Filtre la serie antes de llamar a esta función."
        )

    variables_presentes = set(series_agregadas["variable"].unique())
    faltantes = [v for v in orden_variables if v not in variables_presentes]
    if faltantes:
        raise ObservacionIncompletaError(
            f"Las siguientes variables no tienen ninguna observación en la "
            f"serie agregada: {faltantes}. Verifique la extracción previa "
            "(gee_client / gee_series_builder) para esta provincia-campaña."
        )

    tabla_pivote = series_agregadas.pivot_table(
        index="variable", columns="fase", values="valor_agregado", aggfunc="first"
    )
    tabla_pivote = tabla_pivote.reindex(index=orden_variables, columns=orden_fases)

    return tabla_pivote.to_numpy(dtype=float)


def build_variable_dictionary(entradas: list[dict]) -> pd.DataFrame:
    """Construye el diccionario de variables exigido por la sección 4.10.3.

    Args:
        entradas: lista de diccionarios, cada uno con las claves `nombre`,
            `unidad`, `fuente`, `transformacion`, `rol`. `rol` debe ser uno de
            "secuencial" (parte del tensor V×T), "estatica" (ej. altitud
            media) o "dinamica" (ej. proporción de superficie con quinua) —
            concatenadas después del bloque convolucional.

    Returns:
        DataFrame con una fila por entrada y las columnas declaradas.

    Raises:
        ValueError: si alguna entrada declara un `rol` fuera de
            {"secuencial", "estatica", "dinamica"}.
    """
    roles_invalidos = {e["rol"] for e in entradas} - _ROLES_VALIDOS
    if roles_invalidos:
        raise ValueError(
            f"Rol(es) inválido(s) en el diccionario de variables: "
            f"{roles_invalidos}. Roles permitidos: {sorted(_ROLES_VALIDOS)}."
        )

    return pd.DataFrame(entradas)
