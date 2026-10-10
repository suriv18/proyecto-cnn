"""Pruebas de la orquestación del tensor V×T sobre el conjunto de datos real
(sección 4.10.3).

Conecta las funciones puras y ya probadas de `ingestion.gee_series_builder`
(`aggregate_by_phase`) y `features.tensor_builder` (`build_observation_tensor`)
con las fuentes de datos reales:
  - `data/raw/gee_series_crudo.parquet` (series diarias por variable).
  - `data/interim/fenologia_alineada.parquet` (ventanas de fase por
    provincia-campaña, salida de `preprocessing.phenology_pipeline`).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.features.dataset_builder import build_all_observation_tensors
from src.features.tensor_builder import ObservacionIncompletaError


@pytest.fixture
def series_crudas() -> pd.DataFrame:
    """Formato de gee_series_crudo.parquet: provincia_id, fecha, valor,
    variable, campana_id."""
    filas = []
    for dia in range(1, 11):
        filas.append(
            {
                "provincia_id": "AZANGARO",
                "fecha": f"2020-10-{dia:02d}",
                "valor": float(dia),
                "variable": "precipitacion",
                "campana_id": 2021,
            }
        )
        filas.append(
            {
                "provincia_id": "AZANGARO",
                "fecha": f"2020-10-{dia:02d}",
                "valor": 10.0 + dia,
                "variable": "temperatura_maxima",
                "campana_id": 2021,
            }
        )
    return pd.DataFrame(filas)


@pytest.fixture
def fenologia_alineada() -> pd.DataFrame:
    """Formato de fenologia_alineada.parquet: una fila por fase."""
    return pd.DataFrame(
        [
            {
                "provincia_id": "AZANGARO",
                "campana_id": 2021,
                "fase": "emergencia",
                "inicio": date(2020, 10, 1),
                "fin": date(2020, 10, 5),
            },
            {
                "provincia_id": "AZANGARO",
                "campana_id": 2021,
                "fase": "desarrollo_vegetativo",
                "inicio": date(2020, 10, 6),
                "fin": date(2020, 10, 10),
            },
        ]
    )


class TestBuildAllObservationTensors:
    def test_devuelve_un_tensor_por_provincia_campana(
        self, series_crudas, fenologia_alineada
    ):
        resultado = build_all_observation_tensors(
            series_crudas=series_crudas,
            fenologia_alineada=fenologia_alineada,
            orden_variables=["precipitacion", "temperatura_maxima"],
        )
        assert len(resultado) == 1
        entrada = resultado[0]
        assert entrada["provincia_id"] == "AZANGARO"
        assert entrada["campana_id"] == 2021
        assert entrada["error"] is None

    def test_el_tensor_tiene_forma_v_por_t(self, series_crudas, fenologia_alineada):
        resultado = build_all_observation_tensors(
            series_crudas=series_crudas,
            fenologia_alineada=fenologia_alineada,
            orden_variables=["precipitacion", "temperatura_maxima"],
        )
        tensor = resultado[0]["tensor"]
        assert tensor.shape == (2, 2)  # 2 variables x 2 fases

    def test_aplica_la_regla_de_agregacion_correcta_por_variable(
        self, series_crudas, fenologia_alineada
    ):
        """precipitacion se SUMA por fase (Tabla 7); temperatura_maxima toma
        el MAX. Fase emergencia: días 1-5 -> precipitacion suma 1+2+3+4+5=15;
        temperatura_maxima max(11..15)=15."""
        resultado = build_all_observation_tensors(
            series_crudas=series_crudas,
            fenologia_alineada=fenologia_alineada,
            orden_variables=["precipitacion", "temperatura_maxima"],
        )
        tensor = resultado[0]["tensor"]
        assert tensor[0, 0] == 15.0  # precipitacion, emergencia (suma)
        assert tensor[1, 0] == 15.0  # temperatura_maxima, emergencia (max)

    def test_registra_fallo_explicito_si_falta_una_variable_completa(
        self, series_crudas, fenologia_alineada
    ):
        resultado = build_all_observation_tensors(
            series_crudas=series_crudas,
            fenologia_alineada=fenologia_alineada,
            orden_variables=["precipitacion", "ndvi"],
        )
        entrada = resultado[0]
        assert entrada["tensor"] is None
        assert entrada["error"] is not None
        assert "ndvi" in entrada["error"]

    def test_ndvi_expone_media_y_maximo_como_filas_separadas(
        self, series_crudas, fenologia_alineada
    ):
        """Tabla 7: NDVI se agrega con media Y máximo por fase; en el tensor
        son las variables `ndvi_mean` y `ndvi_max`."""
        ndvi = pd.DataFrame(
            [
                {
                    "provincia_id": "AZANGARO",
                    "fecha": f"2020-10-{dia:02d}",
                    "valor_mean": float(dia),
                    "valor_max": 100.0 + dia,
                    "variable": "ndvi",
                    "campana_id": 2021,
                }
                for dia in range(1, 11)
            ]
        )
        resultado = build_all_observation_tensors(
            series_crudas=pd.concat([series_crudas, ndvi]),
            fenologia_alineada=fenologia_alineada,
            orden_variables=["ndvi_mean", "ndvi_max"],
        )
        tensor = resultado[0]["tensor"]
        assert tensor[0, 0] == 3.0  # media de 1..5
        assert tensor[1, 0] == 105.0  # máximo de 101..105

    def test_ndvi_a_secas_equivale_a_la_media(self, series_crudas, fenologia_alineada):
        ndvi = pd.DataFrame(
            [
                {
                    "provincia_id": "AZANGARO",
                    "fecha": f"2020-10-{dia:02d}",
                    "valor_mean": float(dia),
                    "valor_max": 100.0 + dia,
                    "variable": "ndvi",
                    "campana_id": 2021,
                }
                for dia in range(1, 11)
            ]
        )
        resultado = build_all_observation_tensors(
            series_crudas=ndvi,
            fenologia_alineada=fenologia_alineada,
            orden_variables=["ndvi"],
        )
        assert resultado[0]["tensor"][0, 0] == 3.0

    def test_fases_excluidas_por_el_horizonte_quedan_como_nan_con_t_fijo(
        self, series_crudas, fenologia_alineada
    ):
        """Si el horizonte trunca la segunda observación, su tensor conserva
        las T columnas del lote y deja NaN en la fase excluida."""
        lampa_series = series_crudas.copy()
        lampa_series["provincia_id"] = "LAMPA"
        lampa_fenologia = fenologia_alineada.iloc[[0]].copy()  # solo emergencia
        lampa_fenologia["provincia_id"] = "LAMPA"

        resultado = build_all_observation_tensors(
            series_crudas=pd.concat([series_crudas, lampa_series]),
            fenologia_alineada=pd.concat([fenologia_alineada, lampa_fenologia]),
            orden_variables=["precipitacion"],
        )
        por_provincia = {r["provincia_id"]: r["tensor"] for r in resultado}
        assert por_provincia["LAMPA"].shape == (1, 2)
        assert por_provincia["LAMPA"][0, 0] == 15.0
        assert np.isnan(por_provincia["LAMPA"][0, 1])

    def test_orden_fases_explicito_define_las_columnas(
        self, series_crudas, fenologia_alineada
    ):
        resultado = build_all_observation_tensors(
            series_crudas=series_crudas,
            fenologia_alineada=fenologia_alineada,
            orden_variables=["precipitacion"],
            orden_fases=["desarrollo_vegetativo", "emergencia"],
        )
        tensor = resultado[0]["tensor"]
        assert tensor[0, 0] == 40.0  # días 6-10: 6+7+8+9+10
        assert tensor[0, 1] == 15.0

    def test_procesa_varias_provincia_campanas_independientemente(
        self, series_crudas, fenologia_alineada
    ):
        otra_provincia_series = series_crudas.copy()
        otra_provincia_series["provincia_id"] = "LAMPA"
        otra_provincia_fenologia = fenologia_alineada.copy()
        otra_provincia_fenologia["provincia_id"] = "LAMPA"

        resultado = build_all_observation_tensors(
            series_crudas=pd.concat([series_crudas, otra_provincia_series]),
            fenologia_alineada=pd.concat(
                [fenologia_alineada, otra_provincia_fenologia]
            ),
            orden_variables=["precipitacion", "temperatura_maxima"],
        )
        provincias = {r["provincia_id"] for r in resultado}
        assert provincias == {"AZANGARO", "LAMPA"}
