"""Pruebas de la construcción del tensor V×T por provincia-campaña (sección 4.10.3).

V = predictores secuenciales (precipitación, temperatura máx/mín, radiación,
NDVI), T = pasos temporales (fases fenológicas) hasta el horizonte. La altitud
media (estática) y la proporción de superficie sembrada con quinua (dinámica
provincia-campaña) se mantienen SEPARADAS del tensor V×T: se concatenan después
del bloque convolucional, no como parte de la secuencia de entrada.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.tensor_builder import (
    ObservacionIncompletaError,
    build_observation_tensor,
    build_variable_dictionary,
)


@pytest.fixture
def series_agregadas_por_fase() -> pd.DataFrame:
    """Salida típica de gee_series_builder.aggregate_by_phase para 4 variables
    y 3 fases, ya para una sola provincia-campaña."""
    filas = []
    variables = ["precipitacion", "temperatura_maxima", "temperatura_minima", "ndvi"]
    fases = ["emergencia", "desarrollo_vegetativo", "floracion"]
    valor = 1.0
    for variable in variables:
        for fase in fases:
            filas.append(
                {
                    "provincia_id": "PUN-AZA",
                    "campana_id": 2021,
                    "variable": variable,
                    "fase": fase,
                    "valor_agregado": valor,
                }
            )
            valor += 1.0
    return pd.DataFrame(filas)


@pytest.fixture
def orden_variables() -> list[str]:
    return ["precipitacion", "temperatura_maxima", "temperatura_minima", "ndvi"]


@pytest.fixture
def orden_fases() -> list[str]:
    return ["emergencia", "desarrollo_vegetativo", "floracion"]


class TestBuildObservationTensor:
    def test_forma_del_tensor_es_v_por_t(
        self, series_agregadas_por_fase, orden_variables, orden_fases
    ):
        tensor = build_observation_tensor(
            series_agregadas_por_fase, orden_variables, orden_fases
        )
        assert tensor.shape == (len(orden_variables), len(orden_fases))

    def test_respeta_el_orden_de_variables_declarado(
        self, series_agregadas_por_fase, orden_variables, orden_fases
    ):
        tensor = build_observation_tensor(
            series_agregadas_por_fase, orden_variables, orden_fases
        )
        # precipitacion, fase emergencia -> primer valor insertado (1.0)
        assert tensor[0, 0] == 1.0

    def test_respeta_el_orden_de_fases_declarado(
        self, series_agregadas_por_fase, orden_variables, orden_fases
    ):
        tensor = build_observation_tensor(
            series_agregadas_por_fase, orden_variables, orden_fases
        )
        # precipitacion: emergencia=1.0, desarrollo=2.0, floracion=3.0
        assert list(tensor[0, :]) == [1.0, 2.0, 3.0]

    def test_falla_si_falta_una_variable_completa(
        self, series_agregadas_por_fase, orden_variables, orden_fases
    ):
        sin_ndvi = series_agregadas_por_fase[
            series_agregadas_por_fase["variable"] != "ndvi"
        ]
        with pytest.raises(ObservacionIncompletaError, match="ndvi"):
            build_observation_tensor(sin_ndvi, orden_variables, orden_fases)

    def test_celda_faltante_puntual_se_completa_con_nan_no_con_error(
        self, series_agregadas_por_fase, orden_variables, orden_fases
    ):
        """Una fase sin observación para una variable (ej. nubosidad en NDVI)
        debe registrarse como NaN dentro del tensor, no provocar una excepción
        — el criterio de completitud mínima (80%, sección 4.7) se evalúa
        aparte, sobre el tensor ya construido, no aquí."""
        sin_una_celda = series_agregadas_por_fase[
            ~(
                (series_agregadas_por_fase["variable"] == "ndvi")
                & (series_agregadas_por_fase["fase"] == "floracion")
            )
        ]
        tensor = build_observation_tensor(sin_una_celda, orden_variables, orden_fases)
        assert np.isnan(tensor[orden_variables.index("ndvi"), orden_fases.index("floracion")])

    def test_falla_si_recibe_mas_de_una_provincia_campana_sin_filtrar(
        self, series_agregadas_por_fase, orden_variables, orden_fases
    ):
        """La función espera una única provincia-campaña ya filtrada (el
        llamador es responsable de filtrar, igual que en
        calibrate_ndvi_amplitude_threshold); si recibe una mezcla, debe fallar
        explícitamente en vez de promediar/mezclar silenciosamente valores de
        distintas provincias en la misma celda del tensor."""
        otra_provincia = series_agregadas_por_fase.copy()
        otra_provincia["provincia_id"] = "PUN-OTRA"
        otra_provincia["valor_agregado"] = 100.0
        mezcla = pd.concat([series_agregadas_por_fase, otra_provincia])

        with pytest.raises(ValueError, match="provincia-campa"):
            build_observation_tensor(mezcla, orden_variables, orden_fases)


class TestBuildVariableDictionary:
    def test_registra_forma_unidad_fuente_transformacion_y_rol(self):
        """Sección 4.10.3: el diccionario de variables debe registrar forma,
        unidad, fuente, transformación y rol de cada entrada."""
        entradas = [
            {
                "nombre": "precipitacion",
                "unidad": "mm",
                "fuente": "CHIRPS v2.0",
                "transformacion": "suma por fase",
                "rol": "secuencial",
            },
            {
                "nombre": "altitud_media_agricola",
                "unidad": "m",
                "fuente": "MDE",
                "transformacion": "media sobre mascara agricola",
                "rol": "estatica",
            },
            {
                "nombre": "prop_superficie_quinua",
                "unidad": "proporcion",
                "fuente": "SIEA-MIDAGRI",
                "transformacion": "superficie_quinua / superficie_agricola_total",
                "rol": "dinamica",
            },
        ]
        diccionario = build_variable_dictionary(entradas)

        assert set(diccionario["rol"]) == {"secuencial", "estatica", "dinamica"}
        columnas_requeridas = {"nombre", "unidad", "fuente", "transformacion", "rol"}
        assert columnas_requeridas.issubset(set(diccionario.columns))

    def test_falla_si_una_entrada_no_tiene_rol_valido(self):
        entradas = [
            {
                "nombre": "x",
                "unidad": "u",
                "fuente": "f",
                "transformacion": "t",
                "rol": "rol_invalido",
            }
        ]
        with pytest.raises(ValueError, match="rol_invalido"):
            build_variable_dictionary(entradas)
