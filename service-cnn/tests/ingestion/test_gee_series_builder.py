"""Pruebas de agregación de series GEE por fase fenológica.

Referencia: docs/Proyecto_Tesis_Maestria_UNMSM_WMSR.docx, secciones 4.10.2-4.10.3,
Tabla 7. Esta lógica opera sobre datos YA exportados de Earth Engine (una tabla
provincia x fecha x valor), sin depender de `ee`/`geemap` — 100% testeable sin
credenciales de Google Earth Engine.
"""
from datetime import date

import pandas as pd
import pytest

from src.ingestion.gee_config import AggregationRule
from src.ingestion.gee_series_builder import (
    HorizonteExcedidoError,
    aggregate_by_phase,
)


@pytest.fixture
def serie_diaria_precipitacion() -> pd.DataFrame:
    """Serie diaria de precipitación de una provincia-campaña, ya recortada por
    la ventana [siembra, horizonte] (sección 4.10.4)."""
    fechas = pd.date_range("2020-09-01", "2020-11-30", freq="D")
    return pd.DataFrame(
        {
            "provincia_id": "PUN-AZA",
            "campana_id": 2021,
            "fecha": fechas,
            "valor": [5.0] * len(fechas),  # 5mm/día constante para verificar suma
        }
    )


@pytest.fixture
def fases_fenologicas() -> pd.DataFrame:
    """Ventanas de fase fenológica para la misma provincia-campaña (sección
    4.10.2): emergencia y desarrollo_vegetativo dentro del rango de la serie."""
    return pd.DataFrame(
        [
            {
                "provincia_id": "PUN-AZA",
                "campana_id": 2021,
                "fase": "emergencia",
                "inicio": date(2020, 9, 1),
                "fin": date(2020, 9, 30),
            },
            {
                "provincia_id": "PUN-AZA",
                "campana_id": 2021,
                "fase": "desarrollo_vegetativo",
                "inicio": date(2020, 10, 1),
                "fin": date(2020, 11, 30),
            },
        ]
    )


class TestAggregateByPhase:
    def test_suma_precipitacion_por_fase(
        self, serie_diaria_precipitacion, fases_fenologicas
    ):
        resultado = aggregate_by_phase(
            serie_diaria_precipitacion, fases_fenologicas, AggregationRule.SUM
        )
        fila_emergencia = resultado[resultado["fase"] == "emergencia"].iloc[0]
        # Septiembre tiene 30 días * 5mm = 150mm
        assert fila_emergencia["valor_agregado"] == pytest.approx(150.0)

    def test_suma_precipitacion_fase_de_dos_meses(
        self, serie_diaria_precipitacion, fases_fenologicas
    ):
        resultado = aggregate_by_phase(
            serie_diaria_precipitacion, fases_fenologicas, AggregationRule.SUM
        )
        fila_desarrollo = resultado[
            resultado["fase"] == "desarrollo_vegetativo"
        ].iloc[0]
        # Octubre (31) + Noviembre (30) = 61 días * 5mm = 305mm
        assert fila_desarrollo["valor_agregado"] == pytest.approx(305.0)

    def test_media_para_temperatura(self, fases_fenologicas):
        fechas = pd.date_range("2020-09-01", "2020-09-30", freq="D")
        serie = pd.DataFrame(
            {
                "provincia_id": "PUN-AZA",
                "campana_id": 2021,
                "fecha": fechas,
                "valor": list(range(1, len(fechas) + 1)),  # 1..30
            }
        )
        resultado = aggregate_by_phase(
            serie, fases_fenologicas[:1], AggregationRule.MEAN
        )
        # media de 1..30 = 15.5
        assert resultado.iloc[0]["valor_agregado"] == pytest.approx(15.5)

    def test_preserva_columnas_de_identificacion(
        self, serie_diaria_precipitacion, fases_fenologicas
    ):
        resultado = aggregate_by_phase(
            serie_diaria_precipitacion, fases_fenologicas, AggregationRule.SUM
        )
        assert set(["provincia_id", "campana_id", "fase", "valor_agregado"]).issubset(
            resultado.columns
        )
        assert len(resultado) == 2  # una fila por fase

    def test_fase_sin_datos_en_la_serie_produce_nan_no_cero(
        self, serie_diaria_precipitacion
    ):
        """Una fase sin observaciones en la serie (ej. cobertura de nubes total)
        debe registrarse como NaN, nunca como 0 — un cero falso alteraría la
        suma de precipitación y ocultaría el hueco de datos (sección 4.7,
        criterio de completitud mínima)."""
        fase_sin_datos = pd.DataFrame(
            [
                {
                    "provincia_id": "PUN-AZA",
                    "campana_id": 2021,
                    "fase": "floracion",
                    "inicio": date(2021, 3, 1),
                    "fin": date(2021, 3, 31),
                }
            ]
        )
        resultado = aggregate_by_phase(
            serie_diaria_precipitacion, fase_sin_datos, AggregationRule.SUM
        )
        assert pd.isna(resultado.iloc[0]["valor_agregado"])

    def test_rechaza_datos_posteriores_al_horizonte_de_corte(
        self, fases_fenologicas
    ):
        """Regla anti-fuga (sección 4.10.4 y 4.12.2): ninguna observación
        posterior al punto de corte del horizonte puede usarse. La función debe
        rechazar explícitamente una serie que contenga fechas posteriores al
        `fecha_corte` declarado, en vez de agregarlas silenciosamente."""
        fechas = pd.date_range("2020-09-01", "2021-01-15", freq="D")  # excede corte
        serie_con_fuga = pd.DataFrame(
            {
                "provincia_id": "PUN-AZA",
                "campana_id": 2021,
                "fecha": fechas,
                "valor": [1.0] * len(fechas),
            }
        )
        with pytest.raises(HorizonteExcedidoError):
            aggregate_by_phase(
                serie_con_fuga,
                fases_fenologicas,
                AggregationRule.SUM,
                fecha_corte=date(2020, 11, 30),
            )

    def test_permite_datos_hasta_el_horizonte_inclusive(
        self, serie_diaria_precipitacion, fases_fenologicas
    ):
        # serie_diaria_precipitacion termina exactamente en 2020-11-30
        resultado = aggregate_by_phase(
            serie_diaria_precipitacion,
            fases_fenologicas,
            AggregationRule.SUM,
            fecha_corte=date(2020, 11, 30),
        )
        assert len(resultado) == 2
