"""Orquestador de alineación fenológica sobre el conjunto de datos real.

Referencia: docs/Proyecto_Tesis_Maestria_UNMSM_WMSR.docx, sección 4.10.2.

Conecta las funciones puras y ya probadas de `preprocessing.phenology`
(`estimate_sowing_date`, `build_phase_windows`) con las fuentes de datos
reales:
  - Nivel primario: `ingestion.midagri_siembra_mensual` (superficie sembrada
    mensual por provincia).
  - Nivel secundario: la serie NDVI cruda ya extraída de Earth Engine
    (`data/raw/gee_series_crudo.parquet`, ver `gee_extraction_pipeline`).
  - Nivel de respaldo: recibido ya calculado por el llamador — depende de
    qué campañas son "entrenamiento" en cada pliegue de validación
    (`evaluation.splits`), lógica que este módulo no conoce para mantener la
    separación de responsabilidades ya establecida en el proyecto.
"""
from __future__ import annotations

import calendar
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from src.preprocessing.campaign_calendar import derive_campana_from_month
from src.preprocessing.phenology import (
    FuenteAlineacion,
    build_phase_windows,
    estimate_sowing_date,
)

_MES_INICIO_SEGUNDA_MITAD_AGRICOLA = 7


def _fecha_de_respaldo_para_campana(campana_id: int, mes: int, dia: int) -> date:
    """Combina el mes/día modal de respaldo con el año calendario correcto.

    Sección 4.4: jul-dic pertenece al año calendario `campana_id - 1`
    (siembra); ene-jun pertenece al año calendario `campana_id` — misma
    convención que `campaign_calendar.derive_campana_from_month`, aplicada
    en sentido inverso.
    """
    anio = campana_id - 1 if mes >= _MES_INICIO_SEGUNDA_MITAD_AGRICOLA else campana_id
    return date(anio, mes, dia)


def load_fenologia_bbch(ruta_yaml: str | Path) -> dict[str, int]:
    """Carga la duración en días de cada fase BBCH desde `configs/hyperparams.yaml`.

    Ver la sección `fenologia_bbch` de ese archivo para las fuentes (INIA
    Perú para los días, Sosa-Zuniga et al. 2017 para la nomenclatura BBCH) y
    la limitación de variabilidad varietal declarada.
    """
    contenido = yaml.safe_load(Path(ruta_yaml).read_text(encoding="utf-8"))
    return dict(contenido["fenologia_bbch"])


def _serie_ndvi_de_provincia_campana(
    serie_ndvi_cruda: pd.DataFrame, provincia_id: str, campana_id: int
) -> Optional[pd.DataFrame]:
    filtro = serie_ndvi_cruda[
        (serie_ndvi_cruda["provincia_id"] == provincia_id)
        & (serie_ndvi_cruda["campana_id"] == campana_id)
        & (serie_ndvi_cruda["variable"] == "ndvi")
    ]
    if filtro.empty:
        return None
    return pd.DataFrame(
        {
            "fecha": pd.to_datetime(filtro["fecha"]).dt.date,
            "ndvi": filtro["valor"],
        }
    )


def estimate_departmental_modal_dates_exploratory(
    siembra_mensual: pd.DataFrame,
    provincia_a_departamento: dict[str, str],
) -> dict[str, tuple[int, int]]:
    """Mes/día modal de siembra por departamento, usando TODO el histórico disponible.

    ADVERTENCIA — aproximación EXPLORATORIA, no válida para H2/evaluación
    definitiva: mezcla campañas de prueba y entrenamiento en un mismo
    cálculo, lo que violaría la regla anti-fuga temporal (sección 4.12.2) si
    se usa tal cual dentro de un pliegue de validación. Sirve solo para una
    primera pasada de alineación fenológica exploratoria, mientras el
    esquema de validación real (`evaluation.splits`) no está conectado a
    este pipeline — para la evaluación definitiva, la fecha modal
    departamental de respaldo debe calcularse restringida a las campañas de
    ENTRENAMIENTO de cada pliegue específico.

    Args:
        siembra_mensual: salida de `ingestion.midagri_siembra_mensual.
            load_siembra_mensual_por_provincia` (todas las provincias y
            campañas disponibles, sin restringir a un pliegue).
        provincia_a_departamento: `{provincia_id: departamento}` (ej.
            `configs/provincias.csv`).

    Returns:
        `{departamento: (mes, día)}` — el mes de mayor superficie sembrada
        agregada (sobre todas las provincias y campañas de ese
        departamento) y el punto medio de ese mes. Deliberadamente SIN año
        fijo: `build_all_phase_windows` combina este mes/día con el año de
        CADA campaña que cae al nivel de respaldo — usar aquí un año fijo
        (ej. el máximo histórico del departamento) produciría fechas de
        respaldo con el año equivocado para campañas de otros años (bug
        real detectado: una campaña 2017 terminaba con fecha de respaldo en
        2026). Un departamento sin ninguna fila en `siembra_mensual` no
        aparece en el resultado, para que el llamador use
        `.get(departamento)` de forma uniforme con "sin dato disponible".
    """
    con_departamento = siembra_mensual.copy()
    con_departamento["departamento"] = con_departamento["provincia_id"].map(
        provincia_a_departamento
    )

    resultado: dict[str, tuple[int, int]] = {}
    for departamento, grupo in con_departamento.groupby("departamento"):
        por_mes = grupo.groupby("mes")["superficie_ha"].sum()
        mes_modal = int(por_mes.idxmax())
        # Año bisiesto de referencia solo para calcular monthrange (afecta
        # únicamente a febrero, que no es un mes de siembra de quinua) — el
        # año real se aplica después, por campaña, en build_all_phase_windows.
        ultimo_dia_mes = calendar.monthrange(2000, mes_modal)[1]
        punto_medio = (ultimo_dia_mes + 1) // 2
        resultado[departamento] = (mes_modal, punto_medio)

    return resultado


def build_all_phase_windows(
    provincia_campanas: list[tuple[str, int]],
    siembra_mensual: pd.DataFrame,
    serie_ndvi_cruda: pd.DataFrame,
    fechas_modales_departamentales: dict[str, tuple[int, int]],
    duracion_dias_por_fase: dict[str, int],
    provincia_a_departamento: Optional[dict[str, str]] = None,
    umbral_ascenso_ndvi: float = 0.3,
    fecha_corte: Optional[date] = None,
) -> list[dict]:
    """Alinea fenológicamente cada provincia-campaña del lote dado.

    Args:
        provincia_campanas: pares `(provincia_id, campana_id)` a alinear.
        siembra_mensual: salida de `ingestion.midagri_siembra_mensual.
            load_siembra_mensual_por_provincia` (columnas `provincia_id`,
            `campana_id`, `mes`, `superficie_ha`).
        serie_ndvi_cruda: `data/raw/gee_series_crudo.parquet` ya cargado
            (columnas `provincia_id`, `fecha`, `valor`, `variable`,
            `campana_id`) — se filtra internamente a `variable == "ndvi"`.
        fechas_modales_departamentales: `{departamento: (mes, día)}` ya
            calculado por el llamador para el nivel de respaldo (nivel 3) —
            ver `estimate_departmental_modal_dates_exploratory`. El año se
            toma de la `campana_id` de cada provincia-campaña en curso, no
            del cálculo de respaldo, para no producir fechas con el año
            equivocado.
        duracion_dias_por_fase: duración en días de cada fase BBCH, en el
            orden en que deben aparecer (ver `phenology.build_phase_windows`)
            — calibrada con respaldo agronómico antes del análisis definitivo
            (Sosa-Zuniga et al., 2017).
        provincia_a_departamento: mapeo necesario solo si alguna
            provincia-campaña cae al nivel de respaldo (para saber a qué
            fecha modal departamental recurrir).
        umbral_ascenso_ndvi: ver `phenology.estimate_sowing_date`.
        fecha_corte: ver `phenology.build_phase_windows` (regla anti-fuga,
            sección 4.10.4).

    Returns:
        Lista de dicts, uno por cada `(provincia_id, campana_id)` de
        entrada, cada uno con `provincia_id`, `campana_id`, y:
          - en éxito: `fuente`, `fecha_siembra`, `es_imputado`, `ventanas`
            (lista de `VentanaFase`), `error: None`.
          - en fallo (ningún nivel de la jerarquía disponible): `error` con
            el mensaje, y las demás claves en `None` — no se propaga la
            excepción, para no descartar el resto del lote.
    """
    provincia_a_departamento = provincia_a_departamento or {}
    resultados: list[dict] = []

    for provincia_id, campana_id in provincia_campanas:
        filtro_siembra = siembra_mensual[
            (siembra_mensual["provincia_id"] == provincia_id)
            & (siembra_mensual["campana_id"] == campana_id)
        ]
        siembra_mensual_provincia = filtro_siembra if not filtro_siembra.empty else None

        serie_ndvi_provincia = _serie_ndvi_de_provincia_campana(
            serie_ndvi_cruda, provincia_id, campana_id
        )

        departamento = provincia_a_departamento.get(provincia_id)
        mes_dia_respaldo = (
            fechas_modales_departamentales.get(departamento)
            if departamento is not None
            else None
        )
        fecha_respaldo = (
            _fecha_de_respaldo_para_campana(campana_id, *mes_dia_respaldo)
            if mes_dia_respaldo is not None
            else None
        )

        try:
            fecha_estimada = estimate_sowing_date(
                provincia_id=provincia_id,
                campana_id=campana_id,
                siembra_mensual=siembra_mensual_provincia,
                serie_ndvi=serie_ndvi_provincia,
                fecha_modal_departamental=fecha_respaldo,
                umbral_ascenso_ndvi=umbral_ascenso_ndvi,
            )
        except ValueError as exc:
            resultados.append(
                {
                    "provincia_id": provincia_id,
                    "campana_id": campana_id,
                    "fuente": None,
                    "fecha_siembra": None,
                    "es_imputado": None,
                    "ventanas": None,
                    "error": str(exc),
                }
            )
            continue

        ventanas = build_phase_windows(
            fecha_siembra=fecha_estimada.fecha,
            duracion_dias_por_fase=duracion_dias_por_fase,
            fecha_corte=fecha_corte,
        )

        resultados.append(
            {
                "provincia_id": provincia_id,
                "campana_id": campana_id,
                "fuente": fecha_estimada.fuente,
                "fecha_siembra": fecha_estimada.fecha,
                "es_imputado": fecha_estimada.es_imputado,
                "ventanas": ventanas,
                "error": None,
            }
        )

    return resultados
