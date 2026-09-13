"""Alineación fenológica de las series agroclimáticas y espectrales.

Referencia: docs/Proyecto_Tesis_Maestria_UNMSM_WMSR.docx, sección 4.10.2.

Jerarquía de 3 niveles para estimar la fecha de siembra:
  1. Primario: fecha modal de siembra con superficie sembrada mensual por
     provincia-campaña (punto medio del mes de mayor superficie).
  2. Secundario: si el nivel 1 no está disponible, inicio de estación estimado
     mediante fenometría NDVI.
  3. Respaldo: fecha modal departamental estimada con campañas de entrenamiento,
     marcada explícitamente como imputada.

A partir del inicio estimado, las series se organizan en las fases de la escala
BBCH de la quinua (Sosa-Zuniga et al., 2017): emergencia, desarrollo_vegetativo,
floracion, llenado_grano, madurez.
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from typing import Optional

import pandas as pd


class FuenteAlineacion(str, Enum):
    """De qué nivel de la jerarquía (sección 4.10.2) proviene la fecha estimada."""

    PRIMARIO = "primario"
    SECUNDARIO = "secundario"
    RESPALDO = "respaldo"


@dataclass(frozen=True)
class FechaSiembraEstimada:
    """Resultado de `estimate_sowing_date`, con trazabilidad de su origen."""

    fecha: date
    fuente: FuenteAlineacion
    es_imputado: bool


@dataclass(frozen=True)
class VentanaFase:
    """Ventana temporal de una fase fenológica (escala BBCH)."""

    fase: str
    inicio: date
    fin: date


_MES_INICIO_SEGUNDA_MITAD_AGRICOLA = 7


def _fecha_modal_desde_siembra_mensual(
    provincia_id: str, campana_id: int, siembra_mensual: pd.DataFrame
) -> Optional[date]:
    filtro = siembra_mensual[
        (siembra_mensual["provincia_id"] == provincia_id)
        & (siembra_mensual["campana_id"] == campana_id)
    ]
    if filtro.empty:
        return None

    mes_modal = int(filtro.loc[filtro["superficie_ha"].idxmax(), "mes"])
    # El mes modal pertenece al año calendario ANTERIOR al año de cosecha
    # que identifica la campaña cuando cae en jul-dic (siembra set-nov,
    # sección 4.4/4.10.2), y al mismo año calendario cuando cae en ene-jun
    # — misma convención que campaign_calendar.derive_campana_from_month,
    # aplicada en sentido inverso. Bug real corregido: la versión anterior
    # asumía `anio = campana_id` directamente, produciendo el año de
    # cosecha en vez del año de siembra real para los meses jul-dic (el
    # caso típico de siembra de quinua, set-nov).
    anio = (
        campana_id - 1
        if mes_modal >= _MES_INICIO_SEGUNDA_MITAD_AGRICOLA
        else campana_id
    )
    ultimo_dia = calendar.monthrange(anio, mes_modal)[1]
    punto_medio = (ultimo_dia + 1) // 2
    return date(anio, mes_modal, punto_medio)


def _inicio_estacion_desde_ndvi(
    serie_ndvi: pd.DataFrame, umbral_ascenso_ndvi: float
) -> Optional[date]:
    serie_ordenada = serie_ndvi.sort_values("fecha")
    sobre_umbral = serie_ordenada[serie_ordenada["ndvi"] >= umbral_ascenso_ndvi]
    if sobre_umbral.empty:
        return None
    primera_fecha = sobre_umbral.iloc[0]["fecha"]
    return primera_fecha if isinstance(primera_fecha, date) else primera_fecha.date()


def estimate_sowing_date(
    provincia_id: str,
    campana_id: int,
    siembra_mensual: Optional[pd.DataFrame],
    serie_ndvi: Optional[pd.DataFrame],
    fecha_modal_departamental: Optional[date],
    umbral_ascenso_ndvi: float = 0.3,
) -> FechaSiembraEstimada:
    """Aplica la jerarquía de 3 niveles para estimar la fecha de siembra.

    Args:
        provincia_id, campana_id: identifican la observación provincia-campaña.
        siembra_mensual: columnas `provincia_id`, `campana_id`, `mes`,
            `superficie_ha`. None o sin filas para esta provincia-campaña hace
            caer al nivel secundario.
        serie_ndvi: columnas `fecha`, `ndvi` de la provincia-campaña, ya
            recortada a información disponible hasta el horizonte (evita fuga).
            Se usa solo si el nivel primario no está disponible.
        fecha_modal_departamental: fecha de respaldo estimada con campañas de
            entrenamiento (nivel 3), o None si tampoco está disponible.
        umbral_ascenso_ndvi: umbral de NDVI que marca el inicio de la estación
            de crecimiento (calibrado en el piloto, sección 4.10.2).

    Returns:
        `FechaSiembraEstimada` con la fecha, el nivel de origen y si fue
        imputada (solo el nivel de respaldo se marca como imputado).

    Raises:
        ValueError: si ningún nivel de la jerarquía produce una fecha — nunca
            se asume una fecha arbitraria (disciplina de la sección 4.5.2).
    """
    if siembra_mensual is not None:
        fecha_primaria = _fecha_modal_desde_siembra_mensual(
            provincia_id, campana_id, siembra_mensual
        )
        if fecha_primaria is not None:
            return FechaSiembraEstimada(
                fecha=fecha_primaria, fuente=FuenteAlineacion.PRIMARIO, es_imputado=False
            )

    if serie_ndvi is not None:
        fecha_secundaria = _inicio_estacion_desde_ndvi(serie_ndvi, umbral_ascenso_ndvi)
        if fecha_secundaria is not None:
            return FechaSiembraEstimada(
                fecha=fecha_secundaria,
                fuente=FuenteAlineacion.SECUNDARIO,
                es_imputado=False,
            )

    if fecha_modal_departamental is not None:
        return FechaSiembraEstimada(
            fecha=fecha_modal_departamental,
            fuente=FuenteAlineacion.RESPALDO,
            es_imputado=True,
        )

    raise ValueError(
        f"No fue posible estimar la fecha de siembra de {provincia_id} "
        f"campaña {campana_id}: ningún nivel de la jerarquía de alineación "
        "(primario, secundario, respaldo) tiene datos disponibles."
    )


def build_phase_windows(
    fecha_siembra: date,
    duracion_dias_por_fase: dict[str, int],
    fecha_corte: Optional[date] = None,
) -> list[VentanaFase]:
    """Construye las ventanas de fase fenológica a partir de la fecha de siembra.

    Args:
        fecha_siembra: inicio de la primera fase (nivel de `estimate_sowing_date`).
        duracion_dias_por_fase: duración en días de cada fase, en el orden en
            que deben aparecer (ej. emergencia, desarrollo_vegetativo,
            floracion, llenado_grano, madurez — escala BBCH, Sosa-Zuniga et
            al., 2017). La duración exacta se calibra con respaldo agronómico
            antes del análisis (sección 4.10.2).
        fecha_corte: si se especifica, ninguna ventana puede extenderse más
            allá de esta fecha (sección 4.10.4): la fase que la contiene se
            trunca, y las fases posteriores se excluyen por completo.

    Returns:
        Lista de `VentanaFase` contiguas y sin solapamiento, en el orden dado.
    """
    ventanas: list[VentanaFase] = []
    cursor = fecha_siembra
    for fase, dias in duracion_dias_por_fase.items():
        inicio = cursor
        fin = inicio + timedelta(days=dias - 1)

        # Si `fin` excede el corte, se trunca la fase actual y el bucle
        # termina de inmediato: cualquier fase siguiente empezaría en o
        # después de `fecha_corte`, así que detenerse aquí ya implementa
        # "las fases posteriores se excluyen por completo" (sección 4.10.4)
        # sin necesitar evaluar esas fases una a una.
        if fecha_corte is not None and fin > fecha_corte:
            fin = fecha_corte
            ventanas.append(VentanaFase(fase=fase, inicio=inicio, fin=fin))
            break

        ventanas.append(VentanaFase(fase=fase, inicio=inicio, fin=fin))
        cursor = fin + timedelta(days=1)

    return ventanas
