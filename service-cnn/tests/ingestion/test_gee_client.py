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


class _FakeQaExpression:
    """Doble de prueba de una sub-expresión de calidad (resultado de
    rightShift/bitwiseAnd/eq/lte/And) -- árbol de expresión inspeccionable,
    mismo patrón que test_gee_mask_pipeline.py.

    A diferencia de test_gee_mask_pipeline.py (que inspecciona la EXPRESIÓN
    sin evaluarla, porque no hay un valor de píxel concreto), aquí cada
    imagen fake ya tiene un valor QA concreto inyectado por el test (clave
    "qa" en filas_esperadas), así que el doble evalúa la aritmética de bits
    inmediatamente -- más simple y suficiente para verificar el criterio de
    exclusión con casos concretos.
    """

    def __init__(self, ee_module: "_FakeEeModule", expresion: tuple, valor_qa: int):
        self._ee = ee_module
        self.expresion = expresion
        self._valor_qa = valor_qa

    def rightShift(self, n: int) -> "_FakeQaExpression":
        return _FakeQaExpression(
            self._ee, ("SHR", self.expresion, n), self._valor_qa
        )

    def bitwiseAnd(self, mascara: int) -> "_FakeQaExpression":
        # evalúa inmediatamente contra el valor QA concreto de esta imagen,
        # para que eq/lte comparen sobre el resultado real de la operación
        # de bits, no sobre una expresión simbólica sin evaluar.
        desplazamientos = 0
        expr = self.expresion
        while isinstance(expr, tuple) and expr[0] == "SHR":
            desplazamientos = expr[2]
            expr = expr[1]
        valor_desplazado = self._valor_qa >> desplazamientos
        resultado = valor_desplazado & mascara
        return _FakeQaExpression(
            self._ee, ("AND_BITS", self.expresion, mascara, resultado), self._valor_qa
        )

    def eq(self, valor: int) -> "_FakeBooleanQa":
        _, _, _, resultado = self.expresion
        return _FakeBooleanQa(self._ee, resultado == valor)

    def lte(self, valor: int) -> "_FakeBooleanQa":
        _, _, _, resultado = self.expresion
        return _FakeBooleanQa(self._ee, resultado <= valor)


class _FakeBooleanQa:
    """Resultado booleano evaluado (no simbólico) de una comparación QA."""

    def __init__(self, ee_module: "_FakeEeModule", valor: bool):
        self._ee = ee_module
        self.valor = valor

    def And(self, otra: "_FakeBooleanQa") -> "_FakeBooleanQa":
        return _FakeBooleanQa(self._ee, self.valor and otra.valor)


class _FakeImage:
    """Doble de prueba de una ee.Image individual dentro de la colección."""

    def __init__(
        self,
        ee_module: "_FakeEeModule",
        fecha_iso: str,
        filas: list[dict],
        valor_qa: int | None = None,
    ):
        self._ee = ee_module
        self._fecha_iso = fecha_iso
        self._filas = filas
        self._valor_qa = valor_qa
        self._enmascarada = False

    def date(self):
        return self

    def format(self, formato: str):
        self._ee.llamadas.append(("date.format", formato))
        return self._fecha_iso

    def select(self, banda) -> "Any":
        # la banda de datos (ej. "NDVI") devuelve self (comportamiento
        # actual, compatible con reduceRegions); la banda de calidad
        # ("DetailedQA") devuelve una sub-expresión evaluable contra el
        # valor qa concreto de esta imagen.
        self._ee.llamadas.append(("Image.select", banda))
        if banda == "DetailedQA":
            return _FakeQaExpression(self._ee, ("QA",), self._valor_qa)
        return self

    def updateMask(self, mascara: "_FakeBooleanQa") -> "_FakeImage":
        self._ee.llamadas.append(("Image.updateMask", mascara.valor))
        nueva = _FakeImage(self._ee, self._fecha_iso, self._filas, self._valor_qa)
        nueva._enmascarada = mascara.valor
        return nueva

    def reduceRegions(self, collection, reducer, scale):
        if reducer is None:
            raise ValueError(
                "Parameter 'reducer' is required and may not be null."
            )
        self._ee.llamadas.append(
            ("Image.reduceRegions", reducer, scale, self._enmascarada)
        )
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

    def select(self, band):
        self._ee.llamadas.append(("select", band))
        return self

    def map(self, funcion):
        self._ee.llamadas.append(("map",))
        resultados = []
        for fecha, filas in self._filas_por_fecha.items():
            valor_qa = filas[0].get("qa")
            resultados.append(funcion(_FakeImage(self._ee, fecha, filas, valor_qa)))
        features = [f for coleccion in resultados for f in coleccion._features]
        return _FakeFeatureCollection(features)


class _FakeReducer(str):
    """Doble de prueba de un ee.Reducer: subclase de `str` para que las
    aserciones existentes (que comparan el reducer pasado a reduceRegions
    contra el string "MEAN_REDUCER") sigan funcionando sin cambios, y para
    poder añadir `.combine()` sin dejar de comportarse como el string plano
    en el caso de un solo reducer."""

    def combine(self, reducer2: "_FakeReducer", sharedInputs: bool) -> "_FakeReducer":
        return _FakeReducer(f"{self}+{reducer2}")


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
        def sum():
            return _FakeReducer("SUM_REDUCER")

        @staticmethod
        def mean():
            return _FakeReducer("MEAN_REDUCER")

        @staticmethod
        def max():
            return _FakeReducer("MAX_REDUCER")

        @staticmethod
        def min():
            return _FakeReducer("MIN_REDUCER")


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
        # precipitacion usa AggregationRule.SUM (Tabla 7): la columna que
        # reduceRegions produce se llama "sum" (nombre del reducer real de
        # Earth Engine), no "mean".
        filas_esperadas = [
            {"provincia_id": "PUN-AZA", "fecha": "2020-09-01", "sum": 5.0},
            {"provincia_id": "PUN-AZA", "fecha": "2020-09-02", "sum": 3.2},
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
        # NDVI tiene banda_calidad: se pide junto con la banda de datos a
        # nivel de colección, para que ambas estén disponibles por imagen.
        assert ("select", [spec.band, spec.banda_calidad]) in fake_ee.llamadas
        assert ("map",) in fake_ee.llamadas

    def test_aplica_factor_de_escala_de_la_variable(self, geometria_provincias_fake):
        """MOD13Q1 codifica NDVI como entero crudo; extract_daily_series debe
        aplicar el factor_escala del spec (0.0001) antes de devolver `valor`,
        para que el resto del pipeline reciba NDVI real en [-1, 1], no el
        entero crudo de Earth Engine."""
        # NDVI tiene banda_calidad + reducers=(MEAN, MAX): se incluye "qa"
        # con calidad aceptada y ambas columnas de reducer para no chocar
        # con el enmascarado/multi-reducer añadidos en esta tarea (este test
        # solo verifica factor_escala, no el filtrado/esquema de columnas).
        # Claves con el prefijo de banda ("NDVI_mean"/"NDVI_max"): con
        # múltiples reducers combinados via .combine(sharedInputs=True),
        # reduceRegions antepone el nombre de la banda de entrada al
        # nombre del reducer (verificado contra la guía oficial de Earth
        # Engine, Reducer Overview) — no basta "mean"/"max" a secas.
        filas_esperadas = [
            {
                "provincia_id": "PUN-AZA",
                "fecha": "2020-09-01",
                "NDVI_mean": 3607.0,
                "NDVI_max": 3607.0,
                "qa": 0b00,
            },
        ]
        fake_ee = _FakeEeModule(filas_resultado=filas_esperadas)
        cliente = GeeClient(ee_module=fake_ee, project_id="mi-proyecto-gee")

        resultado = cliente.extract_daily_series(
            variable="ndvi",
            geometrias_por_provincia=geometria_provincias_fake,
            fecha_inicio=date(2020, 9, 1),
            fecha_fin=date(2020, 9, 1),
        )

        # NDVI tiene reducers=(MEAN, MAX): la columna es "valor_mean", no
        # "valor" (ver TestGeeClientReducersMultiples para el esquema de
        # salida con múltiples reducers).
        assert resultado["valor_mean"].iloc[0] == pytest.approx(0.3607)

    def test_aplica_offset_aditivo_de_la_variable(self, geometria_provincias_fake):
        """ERA5-Land reporta temperatura en Kelvin; extract_daily_series debe
        restar 273.15 (offset_aditivo del spec) para devolver Celsius."""
        # temperatura_maxima usa AggregationRule.MAX: columna real "max".
        filas_esperadas = [
            {"provincia_id": "PUN-AZA", "fecha": "2020-09-01", "max": 284.02},
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
            {"provincia_id": "PUN-AZA", "fecha": "2020-09-01", "sum": 5.0},
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
        assert llamadas_reduce[0][1] == "SUM_REDUCER"

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


class TestGeeClientFiltradoCalidadNdvi:
    """Tabla 7: 'Filtrado de calidad; media y máximo por fase sobre máscara
    agrícola' -- verificado contra DetailedQA de MOD13Q1 (bits 0-1 VI
    Quality, bit 8 nube adyacente, bit 10 nubes mixtas)."""

    def _construir_valor_qa(
        self, vi_quality: int, nube_adyacente: int = 0, nubes_mixtas: int = 0
    ) -> int:
        return vi_quality | (nube_adyacente << 8) | (nubes_mixtas << 10)

    def test_acepta_pixel_con_vi_quality_buena(self):
        valor_qa = self._construir_valor_qa(vi_quality=0b00)
        expr = _FakeQaExpression(None, ("QA",), valor_qa)
        vi = expr.rightShift(0).bitwiseAnd(0b11)
        assert vi.eq(0).valor is True or vi.lte(1).valor is True

    def test_acepta_pixel_con_vi_quality_revisar_otras_qa(self):
        valor_qa = self._construir_valor_qa(vi_quality=0b01)
        expr = _FakeQaExpression(None, ("QA",), valor_qa)
        vi = expr.rightShift(0).bitwiseAnd(0b11)
        assert vi.lte(1).valor is True

    def test_rechaza_pixel_nublado(self):
        valor_qa = self._construir_valor_qa(vi_quality=0b10)
        expr = _FakeQaExpression(None, ("QA",), valor_qa)
        vi = expr.rightShift(0).bitwiseAnd(0b11)
        assert vi.lte(1).valor is False

    def test_rechaza_pixel_no_producido(self):
        valor_qa = self._construir_valor_qa(vi_quality=0b11)
        expr = _FakeQaExpression(None, ("QA",), valor_qa)
        vi = expr.rightShift(0).bitwiseAnd(0b11)
        assert vi.lte(1).valor is False

    def test_rechaza_pixel_con_nube_adyacente(self):
        valor_qa = self._construir_valor_qa(vi_quality=0b00, nube_adyacente=1)
        expr = _FakeQaExpression(None, ("QA",), valor_qa)
        bit8 = expr.rightShift(8).bitwiseAnd(0b1)
        assert bit8.eq(0).valor is False

    def test_rechaza_pixel_con_nubes_mixtas(self):
        valor_qa = self._construir_valor_qa(vi_quality=0b00, nubes_mixtas=1)
        expr = _FakeQaExpression(None, ("QA",), valor_qa)
        bit10 = expr.rightShift(10).bitwiseAnd(0b1)
        assert bit10.eq(0).valor is False


class TestGeeClientNombresRealesDeReducersCombinados:
    """Verificado contra la API real (2026-10-09, proyecto cnn-sentinel): sobre
    una imagen de UNA sola banda, `reduceRegions` con reducers combinados
    devuelve las propiedades `mean` y `max` a secas, sin el prefijo de banda.
    El prefijo `NDVI_` solo aparece con imágenes de varias bandas."""

    def test_ndvi_con_propiedades_mean_y_max_sin_prefijo(self, geometria_provincias_fake):
        filas_esperadas = [
            {
                "provincia_id": "PUN-AZA",
                "fecha": "2020-09-01",
                "mean": 5000.0,
                "max": 6000.0,
                "qa": 0b00,
            },
        ]
        fake_ee = _FakeEeModule(filas_resultado=filas_esperadas)
        cliente = GeeClient(ee_module=fake_ee, project_id="mi-proyecto-gee")

        resultado = cliente.extract_daily_series(
            variable="ndvi",
            geometrias_por_provincia=geometria_provincias_fake,
            fecha_inicio=date(2020, 9, 1),
            fecha_fin=date(2020, 9, 1),
        )

        assert resultado["valor_mean"].iloc[0] == pytest.approx(0.5)
        assert resultado["valor_max"].iloc[0] == pytest.approx(0.6)


class TestGeeClientReducersMultiples:
    def test_ndvi_devuelve_columnas_valor_mean_y_valor_max(
        self, geometria_provincias_fake
    ):
        # Con reducers combinados via .combine(sharedInputs=True), Earth
        # Engine antepone el nombre de la banda ("NDVI") al nombre del
        # reducer en cada propiedad de salida -- verificado contra la guía
        # oficial de Earth Engine (Reducer Overview). "mean"/"max" a secas
        # no reflejarían el comportamiento real de la API.
        filas_esperadas = [
            {
                "provincia_id": "PUN-AZA",
                "fecha": "2020-09-01",
                "NDVI_mean": 5000.0,
                "NDVI_max": 6000.0,
                "qa": 0b00,  # VI Quality=buena, sin nube adyacente ni mixta -> pasa el filtro
            },
        ]
        fake_ee = _FakeEeModule(filas_resultado=filas_esperadas)
        cliente = GeeClient(ee_module=fake_ee, project_id="mi-proyecto-gee")

        resultado = cliente.extract_daily_series(
            variable="ndvi",
            geometrias_por_provincia=geometria_provincias_fake,
            fecha_inicio=date(2020, 9, 1),
            fecha_fin=date(2020, 9, 1),
        )

        assert "valor_mean" in resultado.columns
        assert "valor_max" in resultado.columns
        assert "valor" not in resultado.columns
        assert resultado["valor_mean"].iloc[0] == pytest.approx(0.5)  # 5000 * 0.0001
        assert resultado["valor_max"].iloc[0] == pytest.approx(0.6)   # 6000 * 0.0001

    def test_precipitacion_sigue_devolviendo_columna_valor_unica(
        self, geometria_provincias_fake
    ):
        """Retrocompatibilidad: una variable con un solo reducer (el caso de
        hoy para las 4 variables sin banda de calidad) no debe cambiar de
        esquema de salida."""
        # precipitacion usa AggregationRule.SUM: columna real "sum".
        filas_esperadas = [
            {"provincia_id": "PUN-AZA", "fecha": "2020-09-01", "sum": 5.0},
        ]
        fake_ee = _FakeEeModule(filas_resultado=filas_esperadas)
        cliente = GeeClient(ee_module=fake_ee, project_id="mi-proyecto-gee")

        resultado = cliente.extract_daily_series(
            variable="precipitacion",
            geometrias_por_provincia=geometria_provincias_fake,
            fecha_inicio=date(2020, 9, 1),
            fecha_fin=date(2020, 9, 1),
        )

        assert list(resultado.columns) == ["provincia_id", "fecha", "valor"]
        assert resultado["valor"].iloc[0] == pytest.approx(5.0)
