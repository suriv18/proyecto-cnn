"""Agregación de valores SHAP por fase fenológica (sección 3.3.3, 4.15).

Unidad de atribución: predictor × paso temporal, agregada por fase
fenológica. Se usa contribución ABSOLUTA (no la suma con signo) para no
cancelar pasos temporales de signo opuesto dentro de la misma fase al
identificar qué fase concentra mayor contribución predictiva.
"""
from __future__ import annotations

import numpy as np


def aggregate_shap_by_phase(
    valores_shap: np.ndarray, mapeo_paso_a_fase: list[str]
) -> list[dict[str, float]]:
    """Agrega valores SHAP absolutos por fase fenológica, por observación.

    Args:
        valores_shap: forma (n_observaciones, n_variables, n_pasos_temporales)
            — salida de `explain_with_deep_explainer` (`valores_secuenciales`).
        mapeo_paso_a_fase: lista de longitud n_pasos_temporales, con la fase
            (ej. "emergencia", "floracion") de cada paso temporal (derivado de
            `src.preprocessing.phenology.build_phase_windows`).

    Returns:
        Lista de longitud n_observaciones; cada elemento es un diccionario
        `{fase: contribución_absoluta_agregada}`, sumando el valor absoluto de
        SHAP de todas las variables y todos los pasos que pertenecen a esa
        fase.

    Raises:
        ValueError: si `mapeo_paso_a_fase` no tiene un elemento por cada paso
            temporal de `valores_shap`.
    """
    n_pasos_temporales = valores_shap.shape[2]
    if len(mapeo_paso_a_fase) != n_pasos_temporales:
        raise ValueError(
            f"mapeo_paso_a_fase tiene {len(mapeo_paso_a_fase)} elementos, pero "
            f"valores_shap tiene {n_pasos_temporales} pasos temporales. "
            "Deben coincidir uno a uno."
        )

    fases_unicas = sorted(set(mapeo_paso_a_fase))
    indices_por_fase = {
        fase: [i for i, f in enumerate(mapeo_paso_a_fase) if f == fase]
        for fase in fases_unicas
    }

    valores_absolutos = np.abs(valores_shap)

    resultado: list[dict[str, float]] = []
    for observacion in valores_absolutos:
        contribucion_por_fase = {
            fase: float(observacion[:, indices].sum())
            for fase, indices in indices_por_fase.items()
        }
        resultado.append(contribucion_por_fase)

    return resultado


def fase_de_maxima_contribucion(contribucion_por_fase: dict[str, float]) -> str:
    """Identifica la fase con mayor contribución predictiva absoluta agregada.

    Es la "ventana de contribución predictiva" (sección 2.4.7): una propiedad
    de la función predictiva aprendida, no una ventana causal del cultivo.

    Raises:
        ValueError: si el diccionario está vacío.
    """
    if not contribucion_por_fase:
        raise ValueError(
            "No se puede determinar la fase de máxima contribución: el "
            "diccionario de contribuciones por fase está vacío."
        )
    return max(contribucion_por_fase, key=contribucion_por_fase.get)
