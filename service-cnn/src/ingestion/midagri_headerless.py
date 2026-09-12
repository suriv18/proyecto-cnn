"""Lectura del archivo real de MIDAGRI (SISAGRI.xlsx) con cabecera real.

Referencia: configs/column_mapping.yaml — el archivo real (descargado del
dashboard vigente de SIEA-MIDAGRI, ver ese archivo para la URL, la cobertura
temporal confirmada y la brecha respecto a la delimitación de la tesis) SÍ
tiene fila de cabecera con nombres reales de columna. Reporta MES calendario,
no campaña agrícola directamente — estas funciones derivan `campana_id`
(sección 4.4) y adaptan las columnas al esquema que espera
`load_midagri_production`.

Nota técnica: un metadato `<dimension ref="A1"/>` corrupto en el XML de la
mayoría de las hojas del archivo hace que `openpyxl.load_workbook(read_only=
True)` (y cualquier lectura que dependa de streaming basado en ese metadato)
crea que esas hojas están vacías. Leer con `pandas.read_excel(..., header=0)`
en modo no read_only evita ese problema.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import pandas as pd
import yaml

from src.ingestion.midagri_loader import MidagriColumnMapping, load_midagri_production
from src.preprocessing.campaign_calendar import derive_campana_from_month


def select_and_rename_columns(
    datos_crudos: pd.DataFrame, mapeo: dict[str, str]
) -> pd.DataFrame:
    """Selecciona y renombra columnas del archivo real por nombre.

    Args:
        datos_crudos: DataFrame leído con `header=0` (columnas con nombre real).
        mapeo: `{nombre_columna_real: nombre_deseado}`. Solo las columnas
            listadas se conservan en el resultado.

    Returns:
        DataFrame con las columnas seleccionadas, renombradas.

    Raises:
        ValueError: si alguna columna del mapeo no existe en `datos_crudos`
            — señal de que el archivo real cambió de esquema respecto a lo
            confirmado en `configs/column_mapping.yaml`.
    """
    faltantes = [
        columna for columna in mapeo if columna not in datos_crudos.columns
    ]
    if faltantes:
        raise ValueError(
            f"columna(s) {faltantes} no existen en los datos crudos "
            f"(columnas disponibles: {list(datos_crudos.columns)}). El "
            "esquema real puede haber cambiado; verifique "
            "configs/column_mapping.yaml contra el archivo actual."
        )

    return datos_crudos[list(mapeo.keys())].rename(columns=mapeo)


def derive_campana_column(
    datos: pd.DataFrame, columna_anio: str, columna_mes: str, nombre_salida: str
) -> pd.DataFrame:
    """Agrega una columna de campaña agrícola derivada de año y mes calendario.

    Args:
        datos: DataFrame con columnas de año y mes calendario ya nombradas.
        columna_anio, columna_mes: nombres de esas columnas en `datos`.
        nombre_salida: nombre de la nueva columna de campaña a agregar.

    Returns:
        Copia de `datos` con la columna `nombre_salida` añadida; las columnas
        originales de año y mes se conservan sin modificar.
    """
    resultado = datos.copy()
    resultado[nombre_salida] = [
        derive_campana_from_month(anio=int(anio), mes=int(mes))
        for anio, mes in zip(datos[columna_anio], datos[columna_mes])
    ]
    return resultado


@dataclass(frozen=True)
class SisagriHeaderlessConfig:
    """Mapeo de nombres de columna reales del archivo SISAGRI.xlsx,
    confirmado en configs/column_mapping.yaml. `distrito` no forma parte del
    esquema interno de `MidagriColumnMapping` (la unidad de análisis de la
    tesis es provincia, sección 4.4), pero se conserva aquí para un futuro
    paso de agregación distrito->provincia (pendiente de confirmar en H1)."""

    anio: str
    mes: str
    departamento: str
    provincia: str
    distrito: str
    cultivo: str
    superficie_sembrada_ha: str
    superficie_cosechada_ha: str
    produccion_ton: str


def load_sisagri_headerless_config(ruta_yaml: str | Path) -> SisagriHeaderlessConfig:
    """Carga `SisagriHeaderlessConfig` desde la sección `sisagri_headerless`
    de `configs/column_mapping.yaml`."""
    contenido = yaml.safe_load(Path(ruta_yaml).read_text(encoding="utf-8"))
    valores = contenido["sisagri_headerless"]
    campos_validos = {f.name for f in fields(SisagriHeaderlessConfig)}
    return SisagriHeaderlessConfig(**{k: v for k, v in valores.items() if k in campos_validos})


def load_sisagri_headerless(
    datos_crudos: pd.DataFrame, config: SisagriHeaderlessConfig
) -> pd.DataFrame:
    """Orquesta la lectura completa del archivo real SISAGRI.xlsx.

    Selecciona y renombra las columnas reales, deriva la campaña agrícola
    desde año+mes calendario (sección 4.4), y aplica
    `load_midagri_production` para filtrar quinua y reconstruir rendimiento
    (sección 4.5.2).

    Args:
        datos_crudos: DataFrame leído con `pandas.read_excel(header=0)`.
        config: mapeo de columnas confirmado (ver `SisagriHeaderlessConfig`).

    Returns:
        DataFrame en el esquema interno estándar (mismo formato que
        `load_midagri_production`), con `campana_id` ya derivado.
    """
    mapeo = {
        config.anio: "col_anio",
        config.mes: "col_mes",
        config.departamento: "departamento",
        config.provincia: "provincia",
        config.cultivo: "cultivo",
        config.superficie_sembrada_ha: "superficie_sembrada_ha",
        config.superficie_cosechada_ha: "superficie_cosechada_ha",
        config.produccion_ton: "produccion_ton",
    }
    renombrado = select_and_rename_columns(datos_crudos, mapeo)
    con_campana = derive_campana_column(
        renombrado,
        columna_anio="col_anio",
        columna_mes="col_mes",
        nombre_salida="campana_id",
    )

    mapeo_final = MidagriColumnMapping(
        departamento="departamento",
        provincia="provincia",
        campana_id="campana_id",
        cultivo="cultivo",
        superficie_sembrada_ha="superficie_sembrada_ha",
        superficie_cosechada_ha="superficie_cosechada_ha",
        produccion_ton="produccion_ton",
        rendimiento_kg_ha=None,
    )
    return load_midagri_production(con_campana, mapeo_final)
