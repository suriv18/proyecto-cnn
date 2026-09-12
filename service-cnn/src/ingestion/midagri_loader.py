"""Cargador configurable de datos de producción agrícola de MIDAGRI/SIEA.

Referencia: docs/Proyecto_Tesis_Maestria_UNMSM_WMSR.docx, sección 4.9 y Tabla 7.

El esquema real del archivo público de datosabiertos.gob.pe (nombres de columna,
formato exacto) no se conoce todavía de forma verificada: el acceso programático
al portal (CKAN API / descarga directa) requiere navegación manual autenticada.
Por eso este módulo NO asume nombres de columna fijos — recibe un
`MidagriColumnMapping` configurable que se ajusta una vez confirmado el archivo
real durante la Actividad 1 (auditoría H1). Fallar de forma explícita ante un
mapeo incorrecto es preferible a asumir una estructura no verificada (sección
4.5.2 de la tesis: "la disponibilidad... no garantiza su existencia...").
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

_ESQUEMA_INTERNO = [
    "departamento",
    "provincia",
    "campana_id",
    "cultivo",
    "superficie_sembrada_ha",
    "superficie_cosechada_ha",
    "produccion_ton",
]

_CULTIVO_OBJETIVO = "quinua"


class ColumnMappingError(ValueError):
    """El mapeo de columnas configurado no coincide con el archivo real."""


@dataclass(frozen=True)
class MidagriColumnMapping:
    """Mapeo de nombres de columna del archivo real de MIDAGRI al esquema interno.

    `rendimiento_kg_ha` es opcional: si el archivo no lo publica directamente,
    se reconstruye como producción / superficie cosechada (sección 4.5.2).
    """

    departamento: str
    provincia: str
    campana_id: str
    cultivo: str
    superficie_sembrada_ha: str
    superficie_cosechada_ha: str
    produccion_ton: str
    rendimiento_kg_ha: Optional[str] = None

    def columnas_requeridas(self) -> list[str]:
        columnas = [
            self.departamento,
            self.provincia,
            self.campana_id,
            self.cultivo,
            self.superficie_sembrada_ha,
            self.superficie_cosechada_ha,
            self.produccion_ton,
        ]
        if self.rendimiento_kg_ha is not None:
            columnas.append(self.rendimiento_kg_ha)
        return columnas

    def a_esquema_interno(self) -> dict[str, str]:
        renombrado = {
            self.departamento: "departamento",
            self.provincia: "provincia",
            self.campana_id: "campana_id",
            self.cultivo: "cultivo",
            self.superficie_sembrada_ha: "superficie_sembrada_ha",
            self.superficie_cosechada_ha: "superficie_cosechada_ha",
            self.produccion_ton: "produccion_ton",
        }
        if self.rendimiento_kg_ha is not None:
            renombrado[self.rendimiento_kg_ha] = "rendimiento_kg_ha"
        return renombrado


def _normalizar_texto(valor: str) -> str:
    """Normaliza texto para comparación robusta ante mayúsculas/tildes/paréntesis
    (ej. "Quinua", "QUINUA (grano seco)" deben identificarse como el mismo cultivo)."""
    sin_tildes = "".join(
        c for c in unicodedata.normalize("NFKD", valor) if not unicodedata.combining(c)
    )
    return sin_tildes.strip().lower()


def load_column_mapping(
    ruta_yaml: str | Path, seccion: str = "midagri_produccion"
) -> MidagriColumnMapping:
    """Carga un `MidagriColumnMapping` desde `configs/column_mapping.yaml`.

    Args:
        ruta_yaml: ruta al archivo YAML de configuración.
        seccion: clave de nivel superior dentro del YAML (permite añadir otras
            secciones de mapeo en el mismo archivo en el futuro, ej. siembra
            mensual — Tabla 7).
    """
    contenido = yaml.safe_load(Path(ruta_yaml).read_text(encoding="utf-8"))
    valores = contenido[seccion]
    return MidagriColumnMapping(**valores)


def load_midagri_production(
    datos_crudos: pd.DataFrame, mapeo: MidagriColumnMapping
) -> pd.DataFrame:
    """Normaliza un extracto crudo de MIDAGRI al esquema interno y filtra quinua.

    Args:
        datos_crudos: DataFrame tal como se lee del archivo real (columnas con
            los nombres originales del portal, aún no normalizados).
        mapeo: correspondencia entre esos nombres reales y el esquema interno.

    Returns:
        DataFrame con columnas del esquema interno más `rendimiento_kg_ha` y
        `rendimiento_reconstruido` (bool), filtrado a filas de quinua con
        producción y superficie cosechada positivas (sección 4.5.1, criterio de
        contenido: "producción de quinua mayor que cero y superficie cosechada
        positiva").

    Raises:
        ColumnMappingError: si alguna columna del mapeo no existe en los datos
            crudos — señal de que el mapeo configurado no coincide con el
            archivo real descargado.
    """
    faltantes = [
        columna
        for columna in mapeo.columnas_requeridas()
        if columna not in datos_crudos.columns
    ]
    if faltantes:
        raise ColumnMappingError(
            "El mapeo de columnas no coincide con el archivo real. "
            f"Columnas configuradas pero ausentes en los datos: {faltantes}. "
            "Verifique el esquema real del archivo descargado de MIDAGRI/SIEA "
            "(datosabiertos.gob.pe) y actualice configs/column_mapping.yaml."
        )

    datos = datos_crudos.rename(columns=mapeo.a_esquema_interno()).copy()

    es_quinua = datos["cultivo"].apply(_normalizar_texto).str.contains(
        _CULTIVO_OBJETIVO
    )
    datos = datos[es_quinua].copy()

    if "rendimiento_kg_ha" not in datos.columns:
        datos["rendimiento_kg_ha"] = (
            datos["produccion_ton"] * 1000.0 / datos["superficie_cosechada_ha"]
        )
        datos["rendimiento_reconstruido"] = True
    else:
        datos["rendimiento_reconstruido"] = False

    contenido_valido = (datos["produccion_ton"] > 0) & (
        datos["superficie_cosechada_ha"] > 0
    )
    datos = datos[contenido_valido].reset_index(drop=True)

    return datos
