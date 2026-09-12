"""Auditoría de cobertura y elegibilidad provincia-campaña (Actividad 1 / Hito H1).

Reproduce la cascada escalonada N0-N5 de la Tabla 5 del proyecto de tesis
(docs/Proyecto_Tesis_Maestria_UNMSM_WMSR.docx, secciones 4.5.2-4.5.7), sin asumir
tamaños poblacionales mediante tasas de ocupación no observadas (regla explícita
de la sección 4.5.6): N4 y N5 quedan en None hasta que existan datos reales.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

import pandas as pd


class ClasificacionTerritorial(str, Enum):
    """Estados de la clasificación territorial preliminar (Tabla 4)."""

    ALTOANDINA = "altoandina"
    SELVA = "selva"
    COSTA_RIEGO = "costa_riego"
    TRANSICION = "transicion"


_EXCLUIDAS = {ClasificacionTerritorial.SELVA.value, ClasificacionTerritorial.COSTA_RIEGO.value}


def classify_provinces(provincias: pd.DataFrame) -> pd.DataFrame:
    """Aplica la clasificación territorial preliminar (Tabla 4) a cada provincia.

    Espera una columna `clasificacion` con los valores de `ClasificacionTerritorial`.
    Devuelve el DataFrame con una columna adicional `clasificacion_final` tipada.

    Nota: `clasificacion_final` se valida como `ClasificacionTerritorial` pero se
    almacena como su valor `str` — comparar instancias de Enum dentro de una Serie
    de pandas (dtype object) con `==`/`!=`/`isin` no es fiable de forma vectorizada.
    """
    resultado = provincias.copy()
    resultado["clasificacion_final"] = resultado["clasificacion"].apply(
        lambda v: ClasificacionTerritorial(v).value
    )
    return resultado


def build_coverage_cascade(
    provincias: pd.DataFrame,
    n_campanas: int,
    produccion_documentada: Optional[pd.DataFrame],
    celdas_con_produccion: Optional[pd.DataFrame],
    celdas_con_calidad: Optional[pd.DataFrame],
) -> dict[str, Optional[int]]:
    """Calcula la cascada N0-N5 de celdas potencialmente elegibles (Tabla 5).

    Args:
        provincias: columnas `provincia_id`, `departamento`, `clasificacion`
            (valores de `ClasificacionTerritorial`). Representa los 8 departamentos
            altoandinos delimitados en la sección 4.5.
        n_campanas: número de campañas del periodo delimitado (19 para 2006-2024).
        produccion_documentada: columnas `provincia_id`, `produccion_documentada`
            (bool). Si es None, N3 se omite (queda igual a N2) y se marca como
            pendiente de verificación, nunca asumido.
        celdas_con_produccion: columnas `provincia_id`, `campana_id`,
            `produccion_ton`. Si es None, N4 queda en None (sección 4.5.6: no se
            estima con tasas supuestas).
        celdas_con_calidad: columnas `provincia_id`, `campana_id`,
            `cumple_calidad` (bool, completitud >= 80% de predictores, sección 4.7).
            Si es None, N5 queda en None.

    Returns:
        Diccionario con las claves N0..N5. N4/N5 son None cuando no hay datos
        reales todavía, nunca un número estimado.
    """
    clasificadas = classify_provinces(provincias)

    n0 = len(clasificadas) * n_campanas

    provincias_no_selva = clasificadas[
        clasificadas["clasificacion_final"] != ClasificacionTerritorial.SELVA.value
    ]
    n1 = len(provincias_no_selva) * n_campanas

    provincias_altoandinas_o_transicion = provincias_no_selva[
        ~provincias_no_selva["clasificacion_final"].isin(_EXCLUIDAS)
    ]
    n2 = len(provincias_altoandinas_o_transicion) * n_campanas

    if produccion_documentada is not None:
        provincias_con_produccion = provincias_altoandinas_o_transicion.merge(
            produccion_documentada, on="provincia_id", how="inner"
        )
        provincias_con_produccion = provincias_con_produccion[
            provincias_con_produccion["produccion_documentada"]
        ]
        n3 = len(provincias_con_produccion) * n_campanas
    else:
        n3 = n2

    n4: Optional[int]
    if celdas_con_produccion is not None:
        n4 = int((celdas_con_produccion["produccion_ton"] > 0).sum())
    else:
        n4 = None

    n5: Optional[int]
    if celdas_con_calidad is not None:
        n5 = int(celdas_con_calidad["cumple_calidad"].sum())
    else:
        n5 = None

    return {"N0": n0, "N1": n1, "N2": n2, "N3": n3, "N4": n4, "N5": n5}
