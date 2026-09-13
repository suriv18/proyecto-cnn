"""Pruebas de la extracción de superficie sembrada mensual por provincia.

Referencia: docs/Proyecto_Tesis_Maestria_UNMSM_WMSR.docx, sección 4.10.2
(jerarquía de alineación fenológica, nivel primario: fecha modal de siembra
con superficie sembrada mensual por provincia-campaña).

A diferencia de `midagri_aggregation.aggregate_to_provincia_campana` (que
colapsa TODOS los meses de la campaña en un solo total, para el cálculo de
rendimiento), esta función preserva el detalle mensual — necesario para que
`preprocessing.phenology.estimate_sowing_date` encuentre el mes de mayor
superficie sembrada. También agrega DISTRITO -> PROVINCIA (el archivo real
de SISAGRI reporta a nivel distrito; la unidad de análisis de la tesis es
provincia, sección 4.4).
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.midagri_siembra_mensual import load_siembra_mensual_por_provincia


class TestLoadSiembraMensualPorProvincia:
    def test_agrega_distritos_de_la_misma_provincia_y_mes(self):
        datos_crudos = pd.DataFrame(
            [
                {
                    "AÑO": 2020,
                    "MES": 10,
                    "DEPARTAMENTO": "PUNO",
                    "PROVINCIA": "AZANGARO",
                    "DISTRITO": "AZANGARO",
                    "PRODUCTO": "QUINUA",
                    "SIEMBRA": 100.0,
                    "COSECHA": 0.0,
                    "PRODUCCION": 0.0,
                },
                {
                    "AÑO": 2020,
                    "MES": 10,
                    "DEPARTAMENTO": "PUNO",
                    "PROVINCIA": "AZANGARO",
                    "DISTRITO": "SAN JOSE",
                    "PRODUCTO": "QUINUA",
                    "SIEMBRA": 50.0,
                    "COSECHA": 0.0,
                    "PRODUCCION": 0.0,
                },
            ]
        )
        resultado = load_siembra_mensual_por_provincia(datos_crudos)

        assert len(resultado) == 1
        fila = resultado.iloc[0]
        assert fila["provincia_id"] == "AZANGARO"
        assert fila["mes"] == 10
        assert fila["superficie_ha"] == 150.0

    def test_deriva_campana_id_desde_anio_y_mes(self):
        """Octubre pertenece al segundo semestre -> campaña = año + 1
        (sección 4.4: siembra set-nov del año t, cosecha abr-jun del año t+1)."""
        datos_crudos = pd.DataFrame(
            [
                {
                    "AÑO": 2020,
                    "MES": 10,
                    "DEPARTAMENTO": "PUNO",
                    "PROVINCIA": "AZANGARO",
                    "DISTRITO": "AZANGARO",
                    "PRODUCTO": "QUINUA",
                    "SIEMBRA": 100.0,
                    "COSECHA": 0.0,
                    "PRODUCCION": 0.0,
                },
            ]
        )
        resultado = load_siembra_mensual_por_provincia(datos_crudos)
        assert resultado.iloc[0]["campana_id"] == 2021

    def test_filtra_solo_registros_de_quinua(self):
        datos_crudos = pd.DataFrame(
            [
                {
                    "AÑO": 2020,
                    "MES": 10,
                    "DEPARTAMENTO": "PUNO",
                    "PROVINCIA": "AZANGARO",
                    "DISTRITO": "AZANGARO",
                    "PRODUCTO": "PAPA",
                    "SIEMBRA": 500.0,
                    "COSECHA": 0.0,
                    "PRODUCCION": 0.0,
                },
                {
                    "AÑO": 2020,
                    "MES": 10,
                    "DEPARTAMENTO": "PUNO",
                    "PROVINCIA": "AZANGARO",
                    "DISTRITO": "AZANGARO",
                    "PRODUCTO": "QUINUA (grano seco)",
                    "SIEMBRA": 100.0,
                    "COSECHA": 0.0,
                    "PRODUCCION": 0.0,
                },
            ]
        )
        resultado = load_siembra_mensual_por_provincia(datos_crudos)
        assert len(resultado) == 1
        assert resultado.iloc[0]["superficie_ha"] == 100.0

    def test_no_mezcla_provincias_ni_meses_distintos(self):
        datos_crudos = pd.DataFrame(
            [
                {
                    "AÑO": 2020,
                    "MES": 9,
                    "DEPARTAMENTO": "PUNO",
                    "PROVINCIA": "AZANGARO",
                    "DISTRITO": "AZANGARO",
                    "PRODUCTO": "QUINUA",
                    "SIEMBRA": 50.0,
                    "COSECHA": 0.0,
                    "PRODUCCION": 0.0,
                },
                {
                    "AÑO": 2020,
                    "MES": 10,
                    "DEPARTAMENTO": "PUNO",
                    "PROVINCIA": "AZANGARO",
                    "DISTRITO": "AZANGARO",
                    "PRODUCTO": "QUINUA",
                    "SIEMBRA": 300.0,
                    "COSECHA": 0.0,
                    "PRODUCCION": 0.0,
                },
                {
                    "AÑO": 2020,
                    "MES": 10,
                    "DEPARTAMENTO": "PUNO",
                    "PROVINCIA": "LAMPA",
                    "DISTRITO": "LAMPA",
                    "PRODUCTO": "QUINUA",
                    "SIEMBRA": 999.0,
                    "COSECHA": 0.0,
                    "PRODUCCION": 0.0,
                },
            ]
        )
        resultado = load_siembra_mensual_por_provincia(datos_crudos)
        assert len(resultado) == 3
        azangaro_oct = resultado[
            (resultado["provincia_id"] == "AZANGARO") & (resultado["mes"] == 10)
        ]
        assert azangaro_oct.iloc[0]["superficie_ha"] == 300.0

    def test_columnas_del_resultado(self):
        datos_crudos = pd.DataFrame(
            [
                {
                    "AÑO": 2020,
                    "MES": 10,
                    "DEPARTAMENTO": "PUNO",
                    "PROVINCIA": "AZANGARO",
                    "DISTRITO": "AZANGARO",
                    "PRODUCTO": "QUINUA",
                    "SIEMBRA": 100.0,
                    "COSECHA": 0.0,
                    "PRODUCCION": 0.0,
                },
            ]
        )
        resultado = load_siembra_mensual_por_provincia(datos_crudos)
        assert set(resultado.columns) == {
            "provincia_id",
            "campana_id",
            "mes",
            "superficie_ha",
        }

    def test_devuelve_dataframe_vacio_si_no_hay_quinua(self):
        datos_crudos = pd.DataFrame(
            [
                {
                    "AÑO": 2020,
                    "MES": 10,
                    "DEPARTAMENTO": "PUNO",
                    "PROVINCIA": "AZANGARO",
                    "DISTRITO": "AZANGARO",
                    "PRODUCTO": "PAPA",
                    "SIEMBRA": 500.0,
                    "COSECHA": 0.0,
                    "PRODUCCION": 0.0,
                },
            ]
        )
        resultado = load_siembra_mensual_por_provincia(datos_crudos)
        assert len(resultado) == 0
