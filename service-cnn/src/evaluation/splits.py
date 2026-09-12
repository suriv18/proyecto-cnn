"""Esquemas de validación temporal anti-fuga (sección 4.12.1-4.12.2).

- `expanding_window_splits`: esquema confirmatorio principal, origen móvil —
  cada campaña externa se predice usando exclusivamente campañas anteriores.
- `leave_one_campaign_out_splits`: validación complementaria — el entrenamiento
  puede incluir campañas posteriores a la evaluada, por lo que se interpreta
  como secundaria, nunca como el contraste confirmatorio de HE1.
- `leave_one_department_out_splits`: análisis de robustez espacial.

Todos los splits agrupan por bloque completo (campaña o departamento) para que
ninguna observación de la unidad evaluada se filtre al entrenamiento — la regla
de anidamiento de la sección 4.12.2 exige que cada campaña externa permanezca
aislada durante la selección del modelo.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class VentanaInicialInsuficienteError(ValueError):
    """La ventana inicial mínima consume todas (o más) las campañas disponibles,
    dejando ninguna campaña externa que evaluar."""


@dataclass(frozen=True)
class Split:
    """Un pliegue de validación: partición de entrenamiento y prueba."""

    entrenamiento: pd.DataFrame
    prueba: pd.DataFrame


def expanding_window_splits(
    observaciones: pd.DataFrame,
    columna_campana: str,
    ventana_inicial_minima: int,
) -> list[Split]:
    """Genera los pliegues de origen móvil (esquema confirmatorio principal).

    Para cada campaña posterior a la ventana inicial, el entrenamiento incluye
    todas las campañas estrictamente anteriores a ella, y la prueba es
    exactamente esa campaña (con todas sus provincias).

    Args:
        observaciones: filas provincia-campaña, con una columna `columna_campana`.
        columna_campana: nombre de la columna que identifica la campaña (año de
            cosecha, sección 4.4).
        ventana_inicial_minima: número de las primeras campañas reservadas
            exclusivamente para entrenamiento antes de generar la primera
            campaña externa de prueba (definida en el piloto, sección 4.12.1).

    Returns:
        Lista de `Split`, uno por cada campaña externa evaluable, en orden
        cronológico ascendente.

    Raises:
        VentanaInicialInsuficienteError: si no queda ninguna campaña posterior
            a la ventana inicial para evaluar.
    """
    campanas_ordenadas = sorted(observaciones[columna_campana].unique())

    if ventana_inicial_minima >= len(campanas_ordenadas):
        raise VentanaInicialInsuficienteError(
            f"La ventana inicial mínima ({ventana_inicial_minima}) consume "
            f"todas las campañas disponibles ({len(campanas_ordenadas)}); no "
            "queda ninguna campaña externa que evaluar. Reduzca la ventana "
            "inicial o amplíe el periodo (sección 6.1, plan de contingencia)."
        )

    splits: list[Split] = []
    for campana_prueba in campanas_ordenadas[ventana_inicial_minima:]:
        entrenamiento = observaciones[
            observaciones[columna_campana] < campana_prueba
        ]
        prueba = observaciones[observaciones[columna_campana] == campana_prueba]
        splits.append(Split(entrenamiento=entrenamiento, prueba=prueba))

    return splits


def leave_one_campaign_out_splits(
    observaciones: pd.DataFrame, columna_campana: str
) -> list[Split]:
    """Genera los pliegues de leave-one-campaign-out (validación complementaria).

    Cada campaña se usa una vez como prueba; el entrenamiento es el resto de
    campañas, incluyendo las posteriores a la evaluada. Por eso este esquema
    es complementario y no confirmatorio (sección 4.12.1): algunos pliegues
    usan información futura respecto de la campaña de prueba.
    """
    campanas_unicas = sorted(observaciones[columna_campana].unique())

    splits: list[Split] = []
    for campana_prueba in campanas_unicas:
        entrenamiento = observaciones[
            observaciones[columna_campana] != campana_prueba
        ]
        prueba = observaciones[observaciones[columna_campana] == campana_prueba]
        splits.append(Split(entrenamiento=entrenamiento, prueba=prueba))

    return splits


def leave_one_department_out_splits(
    observaciones: pd.DataFrame, columna_departamento: str
) -> list[Split]:
    """Genera los pliegues de leave-one-department-out (robustez espacial).

    Cada departamento se excluye por completo del entrenamiento y se evalúa
    como prueba, con el resto de departamentos como entrenamiento. Análisis de
    sensibilidad, no el contraste confirmatorio de HE1 (sección 4.12.1).
    """
    departamentos_unicos = sorted(observaciones[columna_departamento].unique())

    splits: list[Split] = []
    for departamento_prueba in departamentos_unicos:
        entrenamiento = observaciones[
            observaciones[columna_departamento] != departamento_prueba
        ]
        prueba = observaciones[
            observaciones[columna_departamento] == departamento_prueba
        ]
        splits.append(Split(entrenamiento=entrenamiento, prueba=prueba))

    return splits
