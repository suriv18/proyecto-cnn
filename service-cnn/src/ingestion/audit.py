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
    """Estados de la clasificación territorial preliminar (Tabla 3-4).

    ALTOANDINA/TRANSICION cuentan dentro de las "preliminares de sierra" de
    su departamento (columna 3 de la Tabla 3) y por tanto permanecen en N2
    (ej. Carabaya y Sandia en Puno, La Mar y Huanta en Ayacucho — controladas
    por la máscara agrícola, sección 4.5.3, pero incluidas en el total
    preliminar). PENDIENTE_VERIFICACION identifica provincias marcadas como
    "requiere verificación" que la propia Tabla 3 NO cuenta dentro del total
    preliminar de su departamento (ej. Gran Chimú en La Libertad: 12
    provincias totales, 6 preliminares explícitas, Gran Chimú fuera de esas
    6) — se excluyen de N2 igual que SELVA/COSTA_RIEGO.
    """

    ALTOANDINA = "altoandina"
    SELVA = "selva"
    COSTA_RIEGO = "costa_riego"
    TRANSICION = "transicion"
    PENDIENTE_VERIFICACION = "pendiente_verificacion"


_EXCLUIDAS_N1 = {ClasificacionTerritorial.SELVA.value}
_EXCLUIDAS_N2 = {
    ClasificacionTerritorial.SELVA.value,
    ClasificacionTerritorial.COSTA_RIEGO.value,
    ClasificacionTerritorial.PENDIENTE_VERIFICACION.value,
}


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
    campanas_validas: Optional[set[int]] = None,
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
        campanas_validas: conjunto de años de cosecha dentro del periodo
            delimitado (ej. `set(range(2016, 2026))` tras la enmienda de
            periodo, ver data/manifest/midagri_sisagri.yaml). Si se provee,
            `celdas_con_produccion` y `celdas_con_calidad` se filtran a este
            rango antes de contar N4/N5 — un archivo real de origen puede
            contener campañas fuera del periodo delimitado (ej. datos
            parciales de años en los bordes de la cobertura disponible) que
            no deben inflar N4/N5 por encima de N3 (sección 4.5.6: la
            cascada es de exclusiones sucesivas). Si es None, no se aplica
            ningún filtro de campaña (uso con datos ya pre-filtrados).

    Returns:
        Diccionario con las claves N0..N5. N4/N5 son None cuando no hay datos
        reales todavía, nunca un número estimado.
    """
    clasificadas = classify_provinces(provincias)

    n0 = len(clasificadas) * n_campanas

    provincias_no_selva = clasificadas[
        ~clasificadas["clasificacion_final"].isin(_EXCLUIDAS_N1)
    ]
    n1 = len(provincias_no_selva) * n_campanas

    provincias_altoandinas_o_transicion = clasificadas[
        ~clasificadas["clasificacion_final"].isin(_EXCLUIDAS_N2)
    ]
    n2 = len(provincias_altoandinas_o_transicion) * n_campanas

    if produccion_documentada is not None:
        provincias_elegibles_n3 = provincias_altoandinas_o_transicion.merge(
            produccion_documentada, on="provincia_id", how="inner"
        )
        provincias_elegibles_n3 = provincias_elegibles_n3[
            provincias_elegibles_n3["produccion_documentada"]
        ]
        n3 = len(provincias_elegibles_n3) * n_campanas
    else:
        provincias_elegibles_n3 = provincias_altoandinas_o_transicion
        n3 = n2

    n4: Optional[int]
    if celdas_con_produccion is not None:
        # N4 debe contarse solo sobre provincias que ya pasaron N3 (sección
        # 4.5.6: la cascada es de exclusiones sucesivas, N4 <= N3). Sin este
        # filtro, una celda con producción registrada en una provincia ya
        # excluida (selva, costa, pendiente_verificación, sin producción
        # documentada) inflaría N4 por encima de N3 — bug real detectado al
        # ejecutar la auditoría contra el archivo completo de MIDAGRI.
        ids_elegibles_n3 = set(provincias_elegibles_n3["provincia_id"])
        celdas_elegibles = celdas_con_produccion[
            celdas_con_produccion["provincia_id"].isin(ids_elegibles_n3)
        ]
        if campanas_validas is not None:
            # El archivo de origen puede contener campañas fuera del periodo
            # delimitado (ej. campañas parciales en los bordes de cobertura,
            # sección 6.1 enmienda de periodo) que no deben inflar N4 — bug
            # real detectado con el mismo archivo.
            celdas_elegibles = celdas_elegibles[
                celdas_elegibles["campana_id"].isin(campanas_validas)
            ]
        n4 = int((celdas_elegibles["produccion_ton"] > 0).sum())
    else:
        n4 = None

    n5: Optional[int]
    if celdas_con_calidad is not None:
        celdas_calidad_elegibles = celdas_con_calidad
        if campanas_validas is not None:
            celdas_calidad_elegibles = celdas_calidad_elegibles[
                celdas_calidad_elegibles["campana_id"].isin(campanas_validas)
            ]
        n5 = int(celdas_calidad_elegibles["cumple_calidad"].sum())
    else:
        n5 = None

    return {"N0": n0, "N1": n1, "N2": n2, "N3": n3, "N4": n4, "N5": n5}
