"""Pruebas del orquestador de extracción real de series GEE por campaña.

Referencia: docs/02-arquitectura-tecnica.md §2; docs/Proyecto_Tesis_Maestria_
UNMSM_WMSR.docx sección 4.9 (Tabla 7), 4.10.2 (alineación fenológica).

Este módulo conecta `GeeClient` + `province_geometries` + una lista de
provincias/departamentos para extraer, por cada campaña, una ventana amplia
de datos crudos (año calendario del ciclo agrícola, jul(t-1)-jun(t)) — la
alineación fenológica fina a fases BBCH ocurre después en preprocessing/
phenology.py, que requiere estas series ya extraídas (incluyendo NDVI, usado
como respaldo de nivel 2 para estimar la fecha de siembra). No depende de
`ee` a nivel de módulo: recibe un `GeeClient` ya inyectado con su doble de
prueba (mismo patrón que el resto de `ingestion/`).
"""
from __future__ import annotations

import threading
import time
from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.ingestion.gee_extraction_pipeline import (
    campana_a_ventana_extraccion,
    dividir_en_meses,
    extract_all_provinces_all_variables,
)


class TestCampanaAVentanaExtraccion:
    def test_campana_2020_produce_ventana_jul_2019_jun_2020(self):
        ventana = campana_a_ventana_extraccion(2020)
        assert ventana == (date(2019, 7, 1), date(2020, 6, 30))

    def test_campana_2016_primera_del_periodo_enmendado(self):
        ventana = campana_a_ventana_extraccion(2016)
        assert ventana == (date(2015, 7, 1), date(2016, 6, 30))


class TestDividirEnMeses:
    """`FeatureCollection.getInfo()` aborta con 'Collection query aborted
    after accumulating over 5000 elements' cuando una sola llamada junta
    demasiadas filas (80 provincias x 365 días diarios = 29 200 > 5000,
    verificado contra la API real). Dividir en trozos mensuales (80 x 31 =
    2480 filas como máximo) mantiene cada llamada bajo el límite."""

    def test_ventana_de_un_ano_produce_12_meses(self):
        meses = dividir_en_meses(date(2019, 7, 1), date(2020, 6, 30))
        assert len(meses) == 12
        assert meses[0] == (date(2019, 7, 1), date(2019, 7, 31))
        assert meses[-1] == (date(2020, 6, 1), date(2020, 6, 30))

    def test_meses_son_contiguos_sin_huecos_ni_solapes(self):
        meses = dividir_en_meses(date(2019, 7, 1), date(2020, 6, 30))
        for (_, fin_actual), (inicio_siguiente, _) in zip(meses, meses[1:]):
            assert (inicio_siguiente - fin_actual).days == 1

    def test_respeta_el_limite_de_fecha_fin_en_el_ultimo_mes(self):
        meses = dividir_en_meses(date(2020, 1, 1), date(2020, 1, 15))
        assert meses == [(date(2020, 1, 1), date(2020, 1, 15))]

    def test_ventana_de_un_solo_dia(self):
        meses = dividir_en_meses(date(2020, 9, 1), date(2020, 9, 1))
        assert meses == [(date(2020, 9, 1), date(2020, 9, 1))]

    def test_rechaza_rango_invertido(self):
        with pytest.raises(ValueError, match="fecha_inicio"):
            dividir_en_meses(date(2020, 9, 30), date(2020, 9, 1))


class TestExtractAllProvincesAllVariables:
    @pytest.fixture
    def cliente_fake(self):
        cliente = MagicMock()
        cliente.extract_daily_series.return_value = pd.DataFrame(
            {
                "provincia_id": ["ABANCAY", "ABANCAY"],
                "fecha": ["2019-07-01", "2019-07-02"],
                "valor": [1.0, 2.0],
            }
        )
        return cliente

    @pytest.fixture
    def geometrias_fake(self):
        return object()

    def test_llama_extract_daily_series_una_vez_por_mes_de_cada_campana(
        self, cliente_fake, geometrias_fake
    ):
        """Cada campaña (año calendario del ciclo agrícola) se pagina en 12
        llamadas mensuales — necesario para no exceder el límite real de
        Earth Engine de 5000 elementos por FeatureCollection.getInfo()
        (80 provincias x 365 días diarios = 29 200, verificado en producción)."""
        resultado = extract_all_provinces_all_variables(
            cliente=cliente_fake,
            geometrias_por_provincia=geometrias_fake,
            variables=["precipitacion", "ndvi"],
            campanas=[2016, 2017],
        )

        # 2 variables x 2 campañas x 12 meses = 48 llamadas a extract_daily_series
        assert cliente_fake.extract_daily_series.call_count == 48

    def test_agrega_columnas_variable_y_campana_id_al_resultado(
        self, cliente_fake, geometrias_fake
    ):
        resultado = extract_all_provinces_all_variables(
            cliente=cliente_fake,
            geometrias_por_provincia=geometrias_fake,
            variables=["precipitacion"],
            campanas=[2020],
        )

        assert set(resultado.columns) == {
            "provincia_id",
            "fecha",
            "valor",
            "variable",
            "campana_id",
        }
        assert (resultado["variable"] == "precipitacion").all()
        assert (resultado["campana_id"] == 2020).all()

    def test_concatena_resultados_de_multiples_variables_y_campanas(
        self, cliente_fake, geometrias_fake
    ):
        resultado = extract_all_provinces_all_variables(
            cliente=cliente_fake,
            geometrias_por_provincia=geometrias_fake,
            variables=["precipitacion", "ndvi"],
            campanas=[2016, 2017],
        )

        # cada llamada fake devuelve 2 filas; 2 variables x 2 campañas x 12
        # meses = 48 llamadas -> 96 filas totales
        assert len(resultado) == 96

    def test_pagina_por_mes_dentro_de_la_ventana_de_campana(
        self, cliente_fake, geometrias_fake
    ):
        extract_all_provinces_all_variables(
            cliente=cliente_fake,
            geometrias_por_provincia=geometrias_fake,
            variables=["precipitacion"],
            campanas=[2020],
        )

        llamadas = cliente_fake.extract_daily_series.call_args_list
        assert len(llamadas) == 12
        primera = llamadas[0].kwargs
        ultima = llamadas[-1].kwargs
        assert primera["fecha_inicio"] == date(2019, 7, 1)
        assert primera["fecha_fin"] == date(2019, 7, 31)
        assert primera["variable"] == "precipitacion"
        assert ultima["fecha_inicio"] == date(2020, 6, 1)
        assert ultima["fecha_fin"] == date(2020, 6, 30)

    def test_continua_si_una_llamada_falla_y_reporta_los_fallos(
        self, geometrias_fake
    ):
        """Un fallo en un mes puntual (ej. cuota excedida) no debe descartar
        los demás meses ya extraídos exitosamente de esa misma variable."""
        cliente = MagicMock()

        def _side_effect(variable, geometrias_por_provincia, fecha_inicio, fecha_fin):
            if variable == "ndvi" and fecha_inicio == date(2019, 7, 1):
                raise RuntimeError("EEException simulada")
            return pd.DataFrame(
                {"provincia_id": ["ABANCAY"], "fecha": ["2019-07-01"], "valor": [1.0]}
            )

        cliente.extract_daily_series.side_effect = _side_effect

        resultado, fallos = extract_all_provinces_all_variables(
            cliente=cliente,
            geometrias_por_provincia=geometrias_fake,
            variables=["precipitacion", "ndvi"],
            campanas=[2020],
            devolver_fallos=True,
        )

        # precipitacion: 12 meses ok; ndvi: 11 de 12 meses ok (1 falla)
        assert len(resultado) == 23
        assert len(fallos) == 1
        assert fallos[0]["variable"] == "ndvi"
        assert fallos[0]["campana_id"] == 2020
        assert fallos[0]["fecha_inicio"] == "2019-07-01"

    def test_ejecuta_las_llamadas_en_paralelo_con_max_workers(self, geometrias_fake):
        """El cuello de botella real es espera de red hacia la API de Earth
        Engine (~78s por variable-mes, verificado en producción): con
        max_workers > 1, varias llamadas deben solaparse en el tiempo, no
        ejecutarse una tras otra."""
        marcas_inicio: list[float] = []
        lock = threading.Lock()

        def _side_effect(variable, geometrias_por_provincia, fecha_inicio, fecha_fin):
            with lock:
                marcas_inicio.append(time.monotonic())
            time.sleep(0.2)
            return pd.DataFrame(
                {"provincia_id": ["ABANCAY"], "fecha": ["2019-07-01"], "valor": [1.0]}
            )

        cliente = MagicMock()
        cliente.extract_daily_series.side_effect = _side_effect

        inicio = time.monotonic()
        extract_all_provinces_all_variables(
            cliente=cliente,
            geometrias_por_provincia=geometrias_fake,
            variables=["precipitacion"],
            campanas=[2020],
            max_workers=6,
        )
        duracion = time.monotonic() - inicio

        # 12 llamadas x 0.2s: secuencial tardaría >=2.4s; en paralelo con 6
        # workers, <=3 tandas de 0.2s (~0.6s) más margen de overhead.
        assert duracion < 2.0
        assert len(marcas_inicio) == 12

    def test_max_workers_1_preserva_comportamiento_secuencial(
        self, cliente_fake, geometrias_fake
    ):
        resultado = extract_all_provinces_all_variables(
            cliente=cliente_fake,
            geometrias_por_provincia=geometrias_fake,
            variables=["precipitacion"],
            campanas=[2020],
            max_workers=1,
        )
        assert len(resultado) == 24
