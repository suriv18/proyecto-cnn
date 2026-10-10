"""Orquestación del preprocesamiento: alineación fenológica + tensor V×T + objetivo.

Referencia: secciones 4.10.2-4.10.4 y 4.12.2 de la tesis; fase F2 de CRISP-ML(Q)
(docs/crisp/F2-preparacion-datos.md).

Garantías metodológicas:
  - El respaldo departamental de la alineación (nivel 3) se calcula SOLO con
    `campanas_entrenamiento`. Sin ellas el resultado se marca
    `metadata["exploratorio"] = True` y no es válido para la evaluación
    definitiva (regla anti-fuga, sección 4.12.2).
  - Un parquet crudo con el esquema NDVI anterior (sin `valor_mean`/
    `valor_max`) se rechaza con un error claro en vez de fallar más adelante
    con un `KeyError`.
  - Toda observación descartada se registra con su motivo (sección 4.7).

La máscara agrícola NO se aplica aquí: se aplica en la extracción de Earth
Engine (`gee_mask_pipeline`), por lo que este módulo solo consume el NDVI ya
enmascarado. La covariable estática (altitud) y la dinámica (proporción de
quinua) del tensor (4.10.3) quedan pendientes: necesitan el MDE y la superficie
agrícola total, aún no extraídos.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.features.dataset_builder import build_all_observation_tensors, resolve_variable
from src.features.tensor_builder import build_variable_dictionary
from src.ingestion.gee_config import get_collection_spec
from src.preprocessing.phenology_pipeline import (
    build_all_phase_windows,
    estimate_departmental_modal_dates,
    estimate_departmental_modal_dates_exploratory,
)

ORDEN_VARIABLES_POR_DEFECTO = [
    "precipitacion",
    "temperatura_maxima",
    "temperatura_minima",
    "radiacion_solar",
    "ndvi_mean",
    "ndvi_max",
]

_UNIDADES = {
    "precipitacion": "mm",
    "temperatura_maxima": "°C",
    "temperatura_minima": "°C",
    "radiacion_solar": "J/m²",
    "ndvi": "adimensional",
}


# Límites físicos por variable (valor diario promedio provincial), amplios a
# propósito: detectan errores de agregación (ej. sumar píxeles en vez de
# promediarlos), no valores atípicos. Radiación: J/m2 por día (ERA5-Land).
_LIMITES_FISICOS = {
    "precipitacion": (0.0, 500.0),
    "temperatura_maxima": (-30.0, 50.0),
    "temperatura_minima": (-50.0, 40.0),
    "radiacion_solar": (0.0, 4.5e7),
}
_LIMITES_NDVI = (-0.2, 1.0)


class RawSchemaDesactualizadaError(ValueError):
    """El parquet crudo usa un esquema anterior al vigente (NDVI sin `valor_mean`/`valor_max`)."""


class RawDatosImplausiblesError(ValueError):
    """Hay valores fuera de los límites físicos de la variable (posible error de agregación)."""


@dataclass
class ProcessedDataset:
    X: np.ndarray
    observaciones: pd.DataFrame
    fenologia: pd.DataFrame
    diccionario: pd.DataFrame
    exclusiones: pd.DataFrame
    metadata: dict


def validate_raw_schema(series_crudas: pd.DataFrame) -> None:
    """Verifica que las filas de NDVI tengan `valor_mean` y `valor_max` (Tabla 7).

    Raises:
        RawSchemaDesactualizadaError: si el NDVI conserva el esquema viejo de
            una sola columna `valor`; hay que re-extraer el NDVI de Earth
            Engine antes de preprocesar.
    """
    ndvi = series_crudas[series_crudas["variable"] == "ndvi"]
    if ndvi.empty:
        return
    for columna in ("valor_mean", "valor_max"):
        if columna not in ndvi.columns or ndvi[columna].isna().all():
            raise RawSchemaDesactualizadaError(
                f"Las filas de NDVI no tienen la columna '{columna}': el parquet "
                "crudo se generó con el esquema anterior (NDVI sin filtro de "
                "calidad, solo media). Debe re-extraerse el NDVI de Earth "
                "Engine (scripts/run_ingestion.py) antes de preprocesar."
            )


def validate_raw_ranges(series_crudas: pd.DataFrame) -> None:
    """Rechaza series crudas con valores físicamente imposibles.

    Control de calidad de datos (CRISP-ML(Q), F2): habría detectado la
    regresión del reductor espacial (commit de73679), que dejó la precipitación
    diaria de una provincia como la suma de ~177 píxeles CHIRPS.

    Raises:
        RawDatosImplausiblesError: indicando la variable, el rango esperado y el
            mínimo/máximo observado.
    """
    problemas = []
    for variable, (minimo, maximo) in _LIMITES_FISICOS.items():
        valores = series_crudas.loc[series_crudas["variable"] == variable, "valor"].dropna()
        if len(valores) and (valores.min() < minimo or valores.max() > maximo):
            problemas.append(
                f"{variable}: observado [{valores.min():.4g}, {valores.max():.4g}], "
                f"esperado [{minimo:g}, {maximo:g}]"
            )
    ndvi = series_crudas[series_crudas["variable"] == "ndvi"]
    for columna in ("valor_mean", "valor_max"):
        if columna in ndvi.columns:
            valores = ndvi[columna].dropna()
            if len(valores) and (valores.min() < _LIMITES_NDVI[0] or valores.max() > _LIMITES_NDVI[1]):
                problemas.append(
                    f"ndvi.{columna}: observado [{valores.min():.4g}, {valores.max():.4g}], "
                    f"esperado [{_LIMITES_NDVI[0]:g}, {_LIMITES_NDVI[1]:g}]"
                )
    if problemas:
        raise RawDatosImplausiblesError(
            "Valores fuera de los límites físicos (posible error de agregación "
            "espacial al extraer de Earth Engine): " + "; ".join(problemas)
        )


def _aplanar_ventanas(resultados: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    filas = []
    fallos = []
    for r in resultados:
        if r["error"] is not None:
            fallos.append(r)
            continue
        for ventana in r["ventanas"]:
            filas.append(
                {
                    "provincia_id": r["provincia_id"],
                    "campana_id": r["campana_id"],
                    "fuente": r["fuente"].value,
                    "fecha_siembra": r["fecha_siembra"],
                    "es_imputado": r["es_imputado"],
                    "fase": ventana.fase,
                    "inicio": ventana.inicio,
                    "fin": ventana.fin,
                }
            )
    columnas = [
        "provincia_id", "campana_id", "fuente", "fecha_siembra",
        "es_imputado", "fase", "inicio", "fin",
    ]
    return pd.DataFrame(filas, columns=columnas), fallos


def _diccionario(orden_variables: list[str]) -> pd.DataFrame:
    entradas = []
    for entrada in orden_variables:
        variable, columna, regla = resolve_variable(entrada)
        spec = get_collection_spec(variable)
        entradas.append(
            {
                "nombre": entrada,
                "unidad": _UNIDADES.get(variable, ""),
                "fuente": f"{spec.collection_id} ({spec.band})",
                "transformacion": f"{regla.value} por fase fenológica ({columna})",
                "rol": "secuencial",
            }
        )
    return build_variable_dictionary(entradas)


def build_processed_dataset(
    series_crudas: pd.DataFrame,
    siembra_mensual: pd.DataFrame,
    rendimiento: pd.DataFrame,
    provincia_a_departamento: dict[str, str],
    duracion_dias_por_fase: dict[str, int],
    orden_variables: Optional[list[str]] = None,
    campanas_entrenamiento: Optional[list[int]] = None,
    dias_antes_de_cosecha: Optional[int] = None,
) -> ProcessedDataset:
    """Construye el conjunto de datos procesado (tensor V×T + objetivo).

    Args:
        series_crudas: `data/raw/gee_series_crudo.parquet` ya cargado.
        siembra_mensual: columnas `provincia_id`, `campana_id`, `mes`,
            `superficie_ha` (nivel primario de la alineación).
        rendimiento: columnas `provincia_id`, `campana_id`,
            `rendimiento_kg_ha`; define el universo de observaciones (celdas
            con producción, N4 de la sección 4.5.6).
        provincia_a_departamento: `{provincia_id: departamento}`.
        duracion_dias_por_fase: duración de cada fase BBCH, en orden; también
            fija el orden y el número T de columnas del tensor.
        orden_variables: filas del tensor (por defecto las 5 variables de la
            Tabla 7 con NDVI en media y máximo).
        campanas_entrenamiento: campañas con las que se estima el respaldo
            departamental. `None` => exploratorio (usa todo el histórico).
        dias_antes_de_cosecha: horizonte principal (4.10.4); `None` no trunca.

    Returns:
        `ProcessedDataset`. Las observaciones descartadas quedan en
        `exclusiones` con su `motivo` (`sin_alineacion_fenologica` o
        `tensor_incompleto`).

    Raises:
        RawSchemaDesactualizadaError: si el NDVI crudo usa el esquema viejo.
    """
    orden_variables = orden_variables or ORDEN_VARIABLES_POR_DEFECTO
    validate_raw_schema(series_crudas)
    validate_raw_ranges(series_crudas)

    if campanas_entrenamiento is None:
        fechas_respaldo = estimate_departmental_modal_dates_exploratory(
            siembra_mensual=siembra_mensual,
            provincia_a_departamento=provincia_a_departamento,
        )
    else:
        fechas_respaldo = estimate_departmental_modal_dates(
            siembra_mensual=siembra_mensual,
            provincia_a_departamento=provincia_a_departamento,
            campanas_entrenamiento=campanas_entrenamiento,
        )

    celdas = [
        (str(p), int(c))
        for p, c in rendimiento[["provincia_id", "campana_id"]].itertuples(index=False)
    ]
    resultados = build_all_phase_windows(
        provincia_campanas=celdas,
        siembra_mensual=siembra_mensual,
        serie_ndvi_cruda=series_crudas,
        fechas_modales_departamentales=fechas_respaldo,
        duracion_dias_por_fase=duracion_dias_por_fase,
        provincia_a_departamento=provincia_a_departamento,
        dias_antes_de_cosecha=dias_antes_de_cosecha,
    )
    fenologia, fallos_alineacion = _aplanar_ventanas(resultados)

    exclusiones = [
        {"provincia_id": f["provincia_id"], "campana_id": f["campana_id"],
         "motivo": "sin_alineacion_fenologica", "detalle": f["error"]}
        for f in fallos_alineacion
    ]

    orden_fases = list(duracion_dias_por_fase)
    tensores = build_all_observation_tensors(
        series_crudas=series_crudas,
        fenologia_alineada=fenologia,
        orden_variables=orden_variables,
        orden_fases=orden_fases,
    )

    filas_validas = []
    arreglos = []
    alineacion = fenologia.drop_duplicates(["provincia_id", "campana_id"]).set_index(
        ["provincia_id", "campana_id"]
    )
    objetivo = rendimiento.set_index(["provincia_id", "campana_id"])["rendimiento_kg_ha"]
    for t in tensores:
        clave = (t["provincia_id"], t["campana_id"])
        if t["error"] is not None:
            exclusiones.append(
                {"provincia_id": clave[0], "campana_id": clave[1],
                 "motivo": "tensor_incompleto", "detalle": t["error"]}
            )
            continue
        fila_alineacion = alineacion.loc[clave]
        filas_validas.append(
            {
                "provincia_id": clave[0],
                "campana_id": clave[1],
                "rendimiento_kg_ha": float(objetivo.loc[clave]),
                "fuente_alineacion": fila_alineacion["fuente"],
                "es_imputado": bool(fila_alineacion["es_imputado"]),
                "fecha_siembra": fila_alineacion["fecha_siembra"],
            }
        )
        arreglos.append(t["tensor"])

    observaciones = pd.DataFrame(
        filas_validas,
        columns=[
            "provincia_id", "campana_id", "rendimiento_kg_ha",
            "fuente_alineacion", "es_imputado", "fecha_siembra",
        ],
    )
    X = (
        np.stack(arreglos)
        if arreglos
        else np.empty((0, len(orden_variables), len(orden_fases)))
    )
    df_exclusiones = pd.DataFrame(
        exclusiones, columns=["provincia_id", "campana_id", "motivo", "detalle"]
    )

    metadata = {
        "exploratorio": campanas_entrenamiento is None,
        "campanas_entrenamiento": (
            None if campanas_entrenamiento is None else [int(c) for c in campanas_entrenamiento]
        ),
        "dias_antes_de_cosecha": dias_antes_de_cosecha,
        "orden_variables": list(orden_variables),
        "orden_fases": orden_fases,
        "n_observaciones": int(len(observaciones)),
        "n_exclusiones": int(len(df_exclusiones)),
        "fuentes_alineacion": {
            str(k): int(v)
            for k, v in observaciones["fuente_alineacion"].value_counts().items()
        },
    }
    return ProcessedDataset(
        X=X,
        observaciones=observaciones,
        fenologia=fenologia,
        diccionario=_diccionario(list(orden_variables)),
        exclusiones=df_exclusiones,
        metadata=metadata,
    )


def save_processed_dataset(dataset: ProcessedDataset, directorio: str | Path) -> None:
    """Escribe `tensores.npz`, `observaciones.parquet`, `fenologia.parquet`,
    `exclusiones.csv`, `diccionario_variables.csv` y `metadata.json`."""
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directorio / "tensores.npz", X=dataset.X)
    dataset.observaciones.to_parquet(directorio / "observaciones.parquet", index=False)
    dataset.fenologia.to_parquet(directorio / "fenologia.parquet", index=False)
    dataset.exclusiones.to_csv(directorio / "exclusiones.csv", index=False)
    dataset.diccionario.to_csv(directorio / "diccionario_variables.csv", index=False)
    (directorio / "metadata.json").write_text(
        json.dumps(dataset.metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
