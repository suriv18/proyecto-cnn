"""Pruebas de la capa delgada de acceso a Google Earth Engine.

No requieren `earthengine-api` instalado ni credenciales: `GeeClient` recibe el
módulo `ee` inyectado (o un doble de prueba), en vez de importarlo a nivel de
módulo — patrón necesario porque `ee.Initialize()` requiere autenticación real
que no está disponible en este entorno de desarrollo/CI.
"""
from datetime import date

import pandas as pd
import pytest

from src.ingestion.gee_client import GeeClient
from src.ingestion.gee_config import get_collection_spec


class _FakeFeatureCollection:
    """Doble de prueba de una ee.FeatureCollection ya reducida a tabla."""

    def __init__(self, filas: list[dict]):
        self._filas = filas

    def getInfo(self) -> dict:
        return {
            "features": [{"properties": fila} for fila in self._filas]
        }


class _FakeEeModule:
    """Doble de prueba del módulo `ee`: registra las llamadas realizadas y
    devuelve una FeatureCollection fija, sin tocar la red."""

    def __init__(self, filas_resultado: list[dict]):
        self.initialized = False
        self.filas_resultado = filas_resultado
        self.llamadas: list[tuple] = []

    def Initialize(self, project=None):
        self.initialized = True
        self.llamadas.append(("Initialize", project))

    def ImageCollection(self, collection_id: str):
        self.llamadas.append(("ImageCollection", collection_id))
        return self

    def filterDate(self, inicio, fin):
        self.llamadas.append(("filterDate", inicio, fin))
        return self

    def select(self, band: str):
        self.llamadas.append(("select", band))
        return self

    def toBands(self):
        self.llamadas.append(("toBands",))
        return self

    def reduceRegions(self, collection, reducer, scale):
        self.llamadas.append(("reduceRegions", scale))
        return _FakeFeatureCollection(self.filas_resultado)


@pytest.fixture
def geometria_provincias_fake():
    return object()  # opaco: el cliente no debe inspeccionarlo, solo pasarlo


class TestGeeClientInitialize:
    def test_initialize_pasa_el_project_id(self):
        fake_ee = _FakeEeModule(filas_resultado=[])
        cliente = GeeClient(ee_module=fake_ee, project_id="mi-proyecto-gee")
        cliente.initialize()
        assert fake_ee.initialized is True
        assert ("Initialize", "mi-proyecto-gee") in fake_ee.llamadas


class TestGeeClientExtractSeries:
    def test_extrae_serie_diaria_por_provincia(self, geometria_provincias_fake):
        filas_esperadas = [
            {"provincia_id": "PUN-AZA", "fecha": "2020-09-01", "valor": 5.0},
            {"provincia_id": "PUN-AZA", "fecha": "2020-09-02", "valor": 3.2},
        ]
        fake_ee = _FakeEeModule(filas_resultado=filas_esperadas)
        cliente = GeeClient(ee_module=fake_ee, project_id="mi-proyecto-gee")

        resultado = cliente.extract_daily_series(
            variable="precipitacion",
            geometrias_por_provincia=geometria_provincias_fake,
            fecha_inicio=date(2020, 9, 1),
            fecha_fin=date(2020, 9, 2),
        )

        assert isinstance(resultado, pd.DataFrame)
        assert list(resultado["provincia_id"]) == ["PUN-AZA", "PUN-AZA"]
        assert list(resultado["valor"]) == [5.0, 3.2]

    def test_usa_la_coleccion_correcta_segun_la_variable(
        self, geometria_provincias_fake
    ):
        fake_ee = _FakeEeModule(filas_resultado=[])
        cliente = GeeClient(ee_module=fake_ee, project_id="mi-proyecto-gee")

        cliente.extract_daily_series(
            variable="ndvi",
            geometrias_por_provincia=geometria_provincias_fake,
            fecha_inicio=date(2020, 9, 1),
            fecha_fin=date(2020, 9, 30),
        )

        spec = get_collection_spec("ndvi")
        assert ("ImageCollection", spec.collection_id) in fake_ee.llamadas
        assert ("select", spec.band) in fake_ee.llamadas

    def test_usa_la_escala_espacial_de_la_coleccion(self, geometria_provincias_fake):
        fake_ee = _FakeEeModule(filas_resultado=[])
        cliente = GeeClient(ee_module=fake_ee, project_id="mi-proyecto-gee")

        cliente.extract_daily_series(
            variable="precipitacion",
            geometrias_por_provincia=geometria_provincias_fake,
            fecha_inicio=date(2020, 9, 1),
            fecha_fin=date(2020, 9, 2),
        )

        spec = get_collection_spec("precipitacion")
        assert ("reduceRegions", spec.spatial_resolution_m) in fake_ee.llamadas

    def test_rechaza_rango_de_fechas_invertido(self, geometria_provincias_fake):
        fake_ee = _FakeEeModule(filas_resultado=[])
        cliente = GeeClient(ee_module=fake_ee, project_id="mi-proyecto-gee")

        with pytest.raises(ValueError, match="fecha_inicio"):
            cliente.extract_daily_series(
                variable="precipitacion",
                geometrias_por_provincia=geometria_provincias_fake,
                fecha_inicio=date(2020, 9, 30),
                fecha_fin=date(2020, 9, 1),
            )


class TestGeeClientImportPerezoso:
    def test_no_requiere_earthengine_api_instalado_para_importar_el_modulo(self):
        """El módulo debe poder importarse (para usar GeeClient con un doble de
        prueba) incluso si `earthengine-api` no está instalado en el entorno."""
        import importlib

        modulo = importlib.import_module("src.ingestion.gee_client")
        assert hasattr(modulo, "GeeClient")

    def test_from_default_falla_con_mensaje_claro_si_ee_no_instalado(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "ee":
                raise ModuleNotFoundError("No module named 'ee'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        from src.ingestion.gee_client import GeeClient

        with pytest.raises(ImportError, match="earthengine-api"):
            GeeClient.from_default(project_id="mi-proyecto-gee")
