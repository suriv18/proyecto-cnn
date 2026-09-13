"""Pruebas de la capa delgada de acceso a Google Earth Engine.

No requieren `earthengine-api` instalado ni credenciales: `GeeClient` recibe el
módulo `ee` inyectado (o un doble de prueba), en vez de importarlo a nivel de
módulo — patrón necesario porque `ee.Initialize()` requiere autenticación real
que no está disponible en este entorno de desarrollo/CI.

Diseño de extracción (verificado contra la API real, proyecto cnn-sentinel):
`ImageCollection.map()` reduce cada imagen por separado con `reduceRegions` y
le agrega la fecha como propiedad, luego `.flatten()` aplana el resultado a
una FeatureCollection en formato largo (una fila por provincia × fecha). El
enfoque anterior (`.toBands().reduceRegions(reducer=None, ...)`) fallaba
contra la API real con "Parameter 'reducer' is required and may not be
null." — el doble de prueba anterior no lo detectaba porque ignoraba ese
parámetro.
"""
from datetime import date

import pandas as pd
import pytest

from src.ingestion.gee_client import GeeClient
from src.ingestion.gee_config import AggregationRule, get_collection_spec


class _FakeFeature:
    """Doble de prueba de una ee.Feature: propiedades + .set(clave, valor)."""

    def __init__(self, propiedades: dict):
        self._propiedades = dict(propiedades)

    def set(self, clave: str, valor):
        nuevas = dict(self._propiedades)
        nuevas[clave] = valor
        return _FakeFeature(nuevas)

    def getInfo(self) -> dict:
        return {"properties": self._propiedades}


class _FakeFeatureCollection:
    """Doble de prueba de una ee.FeatureCollection: soporta .map() y .getInfo()."""

    def __init__(self, features: list[_FakeFeature]):
        self._features = features

    def map(self, funcion):
        return _FakeFeatureCollection([funcion(f) for f in self._features])

    def flatten(self):
        return self

    def getInfo(self) -> dict:
        return {"features": [f.getInfo() for f in self._features]}


class _FakeImage:
    """Doble de prueba de una ee.Image individual dentro de la colección."""

    def __init__(self, ee_module: "_FakeEeModule", fecha_iso: str, filas: list[dict]):
        self._ee = ee_module
        self._fecha_iso = fecha_iso
        self._filas = filas

    def date(self):
        return self

    def format(self, formato: str):
        self._ee.llamadas.append(("date.format", formato))
        return self._fecha_iso

    def reduceRegions(self, collection, reducer, scale):
        if reducer is None:
            raise ValueError(
                "Parameter 'reducer' is required and may not be null."
            )
        self._ee.llamadas.append(("Image.reduceRegions", reducer, scale))
        return _FakeFeatureCollection(
            [_FakeFeature(fila) for fila in self._filas]
        )


class _FakeImageCollection:
    """Doble de prueba de una ee.ImageCollection: una imagen fake por fecha."""

    def __init__(self, ee_module: "_FakeEeModule", filas_por_fecha: dict[str, list[dict]]):
        self._ee = ee_module
        self._filas_por_fecha = filas_por_fecha

    def filterDate(self, inicio, fin):
        self._ee.llamadas.append(("filterDate", inicio, fin))
        return self

    def select(self, band: str):
        self._ee.llamadas.append(("select", band))
        return self

    def map(self, funcion):
        self._ee.llamadas.append(("map",))
        resultados = [
            funcion(_FakeImage(self._ee, fecha, filas))
            for fecha, filas in self._filas_por_fecha.items()
        ]
        features = [f for coleccion in resultados for f in coleccion._features]
        return _FakeFeatureCollection(features)


class _FakeEeModule:
    """Doble de prueba del módulo `ee`: registra las llamadas realizadas y
    devuelve una FeatureCollection fija, sin tocar la red."""

    def __init__(self, filas_resultado: list[dict]):
        self.initialized = False
        self.filas_resultado = filas_resultado
        self.llamadas: list[tuple] = []
        # agrupar filas esperadas por fecha, para simular reduceRegions por imagen
        self._filas_por_fecha: dict[str, list[dict]] = {}
        for fila in filas_resultado:
            self._filas_por_fecha.setdefault(fila.get("fecha", ""), []).append(fila)

    def Initialize(self, project=None):
        self.initialized = True
        self.llamadas.append(("Initialize", project))

    def ImageCollection(self, collection_id: str):
        self.llamadas.append(("ImageCollection", collection_id))
        return _FakeImageCollection(self, self._filas_por_fecha)

    class Reducer:
        @staticmethod
        def mean():
            return "MEAN_REDUCER"


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

    def test_ee_module_expone_el_modulo_inyectado(self):
        """Otros módulos (ej. province_geometries.build_province_feature_
        collection) necesitan el módulo `ee` real para construir geometrías
        con la misma sesión ya inicializada por GeeClient."""
        fake_ee = _FakeEeModule(filas_resultado=[])
        cliente = GeeClient(ee_module=fake_ee, project_id="mi-proyecto-gee")
        assert cliente.ee_module is fake_ee


class TestGeeClientExtractSeries:
    def test_extrae_serie_diaria_por_provincia(self, geometria_provincias_fake):
        filas_esperadas = [
            {"provincia_id": "PUN-AZA", "fecha": "2020-09-01", "mean": 5.0},
            {"provincia_id": "PUN-AZA", "fecha": "2020-09-02", "mean": 3.2},
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
        assert list(resultado["fecha"]) == ["2020-09-01", "2020-09-02"]

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
        assert ("map",) in fake_ee.llamadas

    def test_aplica_factor_de_escala_de_la_variable(self, geometria_provincias_fake):
        """MOD13Q1 codifica NDVI como entero crudo; extract_daily_series debe
        aplicar el factor_escala del spec (0.0001) antes de devolver `valor`,
        para que el resto del pipeline reciba NDVI real en [-1, 1], no el
        entero crudo de Earth Engine."""
        filas_esperadas = [
            {"provincia_id": "PUN-AZA", "fecha": "2020-09-01", "mean": 3607.0},
        ]
        fake_ee = _FakeEeModule(filas_resultado=filas_esperadas)
        cliente = GeeClient(ee_module=fake_ee, project_id="mi-proyecto-gee")

        resultado = cliente.extract_daily_series(
            variable="ndvi",
            geometrias_por_provincia=geometria_provincias_fake,
            fecha_inicio=date(2020, 9, 1),
            fecha_fin=date(2020, 9, 1),
        )

        assert resultado["valor"].iloc[0] == pytest.approx(0.3607)

    def test_aplica_offset_aditivo_de_la_variable(self, geometria_provincias_fake):
        """ERA5-Land reporta temperatura en Kelvin; extract_daily_series debe
        restar 273.15 (offset_aditivo del spec) para devolver Celsius."""
        filas_esperadas = [
            {"provincia_id": "PUN-AZA", "fecha": "2020-09-01", "mean": 284.02},
        ]
        fake_ee = _FakeEeModule(filas_resultado=filas_esperadas)
        cliente = GeeClient(ee_module=fake_ee, project_id="mi-proyecto-gee")

        resultado = cliente.extract_daily_series(
            variable="temperatura_maxima",
            geometrias_por_provincia=geometria_provincias_fake,
            fecha_inicio=date(2020, 9, 1),
            fecha_fin=date(2020, 9, 1),
        )

        assert resultado["valor"].iloc[0] == pytest.approx(10.87)

    def test_usa_un_reducer_real_no_nulo(self, geometria_provincias_fake):
        """Expone el bug original: `reducer=None` fallaba contra la API real
        con 'Parameter reducer is required and may not be null'. El doble de
        prueba ahora reproduce ese rechazo, así que este test falla en rojo
        si el código de producción vuelve a pasar `reducer=None`."""
        filas_esperadas = [
            {"provincia_id": "PUN-AZA", "fecha": "2020-09-01", "mean": 5.0},
        ]
        fake_ee = _FakeEeModule(filas_resultado=filas_esperadas)
        cliente = GeeClient(ee_module=fake_ee, project_id="mi-proyecto-gee")

        cliente.extract_daily_series(
            variable="precipitacion",
            geometrias_por_provincia=geometria_provincias_fake,
            fecha_inicio=date(2020, 9, 1),
            fecha_fin=date(2020, 9, 1),
        )

        llamadas_reduce = [l for l in fake_ee.llamadas if l[0] == "Image.reduceRegions"]
        assert len(llamadas_reduce) == 1
        assert llamadas_reduce[0][1] == "MEAN_REDUCER"

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

    def test_devuelve_dataframe_vacio_si_no_hay_resultados(
        self, geometria_provincias_fake
    ):
        fake_ee = _FakeEeModule(filas_resultado=[])
        cliente = GeeClient(ee_module=fake_ee, project_id="mi-proyecto-gee")

        resultado = cliente.extract_daily_series(
            variable="precipitacion",
            geometrias_por_provincia=geometria_provincias_fake,
            fecha_inicio=date(2020, 9, 1),
            fecha_fin=date(2020, 9, 2),
        )
        assert len(resultado) == 0


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
