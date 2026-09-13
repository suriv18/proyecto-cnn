"""Pruebas de la máscara agrícola server-side en Google Earth Engine.

Referencia: service-cnn/docs/superpowers/specs/2026-09-13-mascara-agricola-gee-design.md

No requiere `earthengine-api` instalado ni credenciales: recibe el módulo
`ee` inyectado (doble de prueba), mismo patrón que test_gee_client.py.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import pytest

from src.ingestion.gee_mask_pipeline import build_agricultural_mask_image
from src.preprocessing.mask import MaskThresholds


class _FakeBooleanImage:
    """Doble de prueba de una ee.Image booleana: soporta .And() y guarda
    la expresión como árbol para inspección en los tests."""

    def __init__(self, ee_module: "_FakeEeModule", expresion: tuple):
        self._ee = ee_module
        self.expresion = expresion

    def And(self, otra: "_FakeBooleanImage") -> "_FakeBooleanImage":
        return _FakeBooleanImage(self._ee, ("AND", self.expresion, otra.expresion))


class _FakeSingleBandImage:
    """Doble de prueba de una ee.Image de una banda (elevación, pendiente,
    clase de cobertura, o max/min de NDVI), con comparaciones booleanas."""

    def __init__(self, ee_module: "_FakeEeModule", nombre: str):
        self._ee = ee_module
        self.nombre = nombre

    def eq(self, valor) -> _FakeBooleanImage:
        return _FakeBooleanImage(self._ee, ("EQ", self.nombre, valor))

    def gte(self, valor) -> _FakeBooleanImage:
        return _FakeBooleanImage(self._ee, ("GTE", self.nombre, valor))

    def lte(self, valor) -> _FakeBooleanImage:
        return _FakeBooleanImage(self._ee, ("LTE", self.nombre, valor))

    def subtract(self, otra: "_FakeSingleBandImage") -> "_FakeSingleBandImage":
        return _FakeSingleBandImage(self._ee, f"({self.nombre} - {otra.nombre})")

    def multiply(self, valor) -> "_FakeSingleBandImage":
        return _FakeSingleBandImage(self._ee, f"({self.nombre} * {valor})")

    def rename(self, nombres: list[str]) -> "_FakeSingleBandImage":
        return self


class _FakeImage:
    """Doble de prueba de ee.Image(id): expone .select(banda)."""

    def __init__(self, ee_module: "_FakeEeModule", image_id: str):
        self._ee = ee_module
        self.image_id = image_id

    def select(self, banda: str) -> _FakeSingleBandImage:
        self._ee.llamadas.append(("Image.select", self.image_id, banda))
        return _FakeSingleBandImage(self._ee, f"{self.image_id}:{banda}")


class _FakeImageCollectionParaMascara:
    """Doble de prueba de ee.ImageCollection(id) usado solo para la
    reducción temporal max/min de NDVI dentro de la máscara."""

    def __init__(self, ee_module: "_FakeEeModule", collection_id: str):
        self._ee = ee_module
        self.collection_id = collection_id
        self._banda: str | None = None

    def filterDate(self, inicio, fin) -> "_FakeImageCollectionParaMascara":
        self._ee.llamadas.append(("Mascara.filterDate", self.collection_id, inicio, fin))
        return self

    def select(self, banda: str) -> "_FakeImageCollectionParaMascara":
        self._banda = banda
        self._ee.llamadas.append(("Mascara.select", self.collection_id, banda))
        return self

    def max(self) -> _FakeSingleBandImage:
        return _FakeSingleBandImage(self._ee, f"{self.collection_id}:{self._banda}:max")

    def min(self) -> _FakeSingleBandImage:
        return _FakeSingleBandImage(self._ee, f"{self.collection_id}:{self._banda}:min")


class _FakeEeModule:
    """Doble de prueba del módulo `ee` con álgebra de imágenes suficiente
    para construir y verificar la máscara agrícola, sin tocar la red."""

    def __init__(self):
        self.llamadas: list[tuple] = []

    def Image(self, image_id: str) -> _FakeImage:
        self.llamadas.append(("Image", image_id))
        return _FakeImage(self, image_id)

    def ImageCollection(self, collection_id: str) -> _FakeImageCollectionParaMascara:
        self.llamadas.append(("ImageCollection", collection_id))
        return _FakeImageCollectionParaMascara(self, collection_id)

    class Terrain:
        @staticmethod
        def slope(imagen: _FakeSingleBandImage) -> _FakeSingleBandImage:
            return _FakeSingleBandImage(imagen._ee, f"slope({imagen.nombre})")


class _FakeFeature:
    def __init__(self, propiedades: dict):
        self._propiedades = dict(propiedades)

    def set(self, clave: str, valor):
        nuevas = dict(self._propiedades)
        nuevas[clave] = valor
        return _FakeFeature(nuevas)

    def getInfo(self) -> dict:
        return {"properties": self._propiedades}


class _FakeFeatureCollectionResultado:
    def __init__(self, features: list[_FakeFeature]):
        self._features = features

    def map(self, funcion):
        return _FakeFeatureCollectionResultado([funcion(f) for f in self._features])

    def flatten(self):
        return self

    def getInfo(self) -> dict:
        return {"features": [f.getInfo() for f in self._features]}


class _FakeNdviImage:
    """Imagen NDVI individual dentro de la colección, enmascarable."""

    def __init__(self, ee_module: "_FakeEeModuleConMascara", fecha_iso: str, filas: list[dict]):
        self._ee = ee_module
        self._fecha_iso = fecha_iso
        self._filas = filas
        self._enmascarada = False

    def date(self):
        return self

    def format(self, formato: str):
        return self._fecha_iso

    def updateMask(self, mascara) -> "_FakeNdviImage":
        self._ee.llamadas.append(("Image.updateMask", mascara))
        nueva = _FakeNdviImage(self._ee, self._fecha_iso, self._filas)
        nueva._enmascarada = True
        return nueva

    def reduceRegions(self, collection, reducer, scale):
        self._ee.llamadas.append(("Image.reduceRegions", reducer, scale, self._enmascarada))
        return _FakeFeatureCollectionResultado(
            [_FakeFeature(fila) for fila in self._filas]
        )


class _FakeNdviImageCollection:
    def __init__(self, ee_module: "_FakeEeModuleConMascara", filas_por_fecha: dict[str, list[dict]]):
        self._ee = ee_module
        self._filas_por_fecha = filas_por_fecha

    def filterDate(self, inicio, fin):
        self._ee.llamadas.append(("Ndvi.filterDate", inicio, fin))
        return self

    def select(self, banda: str):
        return self

    def map(self, funcion):
        resultados = [
            funcion(_FakeNdviImage(self._ee, fecha, filas))
            for fecha, filas in self._filas_por_fecha.items()
        ]
        features = [f for coleccion in resultados for f in coleccion._features]
        return _FakeFeatureCollectionResultado(features)

    def max(self):
        return _FakeSingleBandImage(self._ee, "MODIS/061/MOD13Q1:NDVI:max")

    def min(self):
        return _FakeSingleBandImage(self._ee, "MODIS/061/MOD13Q1:NDVI:min")


class _FakeEeModuleConMascara(_FakeEeModule):
    """Extiende el doble de prueba de la máscara para soportar también la
    extracción NDVI real (reduceRegions + updateMask), sobre una colección
    NDVI separada de la usada dentro de build_agricultural_mask_image."""

    def __init__(self, filas_resultado: list[dict]):
        super().__init__()
        self._filas_por_fecha: dict[str, list[dict]] = {}
        for fila in filas_resultado:
            self._filas_por_fecha.setdefault(fila.get("fecha", ""), []).append(fila)
        self._coleccion_ndvi_extraccion = _FakeNdviImageCollection(
            self, self._filas_por_fecha
        )

    def ImageCollection(self, collection_id: str):
        self.llamadas.append(("ImageCollection", collection_id))
        if collection_id == "MODIS/061/MOD13Q1":
            return self._coleccion_ndvi_extraccion
        return super().ImageCollection(collection_id)

    class Reducer:
        @staticmethod
        def mean():
            return "MEAN_REDUCER"


@pytest.fixture
def umbrales() -> MaskThresholds:
    return MaskThresholds(
        altitud_min_m=2500,
        altitud_max_m=4200,
        pendiente_max_grados=25.0,
        amplitud_ndvi_min=0.15,
    )


class TestBuildAgriculturalMaskImage:
    def test_consulta_worldcover_srtm_y_modis(self, umbrales):
        fake_ee = _FakeEeModule()

        build_agricultural_mask_image(
            ee_module=fake_ee,
            umbrales=umbrales,
            fecha_inicio=date(2019, 7, 1),
            fecha_fin=date(2020, 6, 30),
        )

        assert ("Image", "ESA/WorldCover/v200/2021") in fake_ee.llamadas
        assert ("Image", "USGS/SRTMGL1_003") in fake_ee.llamadas
        assert ("ImageCollection", "MODIS/061/MOD13Q1") in fake_ee.llamadas

    def test_selecciona_las_bandas_correctas(self, umbrales):
        fake_ee = _FakeEeModule()

        build_agricultural_mask_image(
            ee_module=fake_ee,
            umbrales=umbrales,
            fecha_inicio=date(2019, 7, 1),
            fecha_fin=date(2020, 6, 30),
        )

        assert ("Image.select", "ESA/WorldCover/v200/2021", "Map") in fake_ee.llamadas
        assert ("Image.select", "USGS/SRTMGL1_003", "elevation") in fake_ee.llamadas
        assert ("Mascara.select", "MODIS/061/MOD13Q1", "NDVI") in fake_ee.llamadas

    def test_filtra_ndvi_a_la_ventana_de_la_campana(self, umbrales):
        fake_ee = _FakeEeModule()

        build_agricultural_mask_image(
            ee_module=fake_ee,
            umbrales=umbrales,
            fecha_inicio=date(2019, 7, 1),
            fecha_fin=date(2020, 6, 30),
        )

        assert (
            "Mascara.filterDate",
            "MODIS/061/MOD13Q1",
            "2019-07-01",
            "2020-06-30",
        ) in fake_ee.llamadas

    def test_combina_los_cuatro_criterios_con_and(self, umbrales):
        fake_ee = _FakeEeModule()

        resultado = build_agricultural_mask_image(
            ee_module=fake_ee,
            umbrales=umbrales,
            fecha_inicio=date(2019, 7, 1),
            fecha_fin=date(2020, 6, 30),
        )

        # el árbol de expresión resultante debe combinar los 4 criterios con AND,
        # donde altitud es un AND interno de min/max (5 hojas totales)
        def contar_hojas(expr) -> int:
            if isinstance(expr, tuple) and expr[0] == "AND":
                return contar_hojas(expr[1]) + contar_hojas(expr[2])
            return 1

        assert contar_hojas(resultado.expresion) == 5

    def test_criterio_de_cobertura_compara_con_clase_40(self, umbrales):
        fake_ee = _FakeEeModule()

        resultado = build_agricultural_mask_image(
            ee_module=fake_ee,
            umbrales=umbrales,
            fecha_inicio=date(2019, 7, 1),
            fecha_fin=date(2020, 6, 30),
        )

        def contiene(expr, buscado) -> bool:
            if isinstance(expr, tuple):
                if expr == buscado:
                    return True
                return any(
                    contiene(sub, buscado) for sub in expr if isinstance(sub, tuple)
                )
            return False

        assert contiene(
            resultado.expresion,
            ("EQ", "ESA/WorldCover/v200/2021:Map", 40),
        )

    def test_criterio_de_pendiente_usa_terrain_slope(self, umbrales):
        fake_ee = _FakeEeModule()

        resultado = build_agricultural_mask_image(
            ee_module=fake_ee,
            umbrales=umbrales,
            fecha_inicio=date(2019, 7, 1),
            fecha_fin=date(2020, 6, 30),
        )

        def contiene(expr, buscado) -> bool:
            if isinstance(expr, tuple):
                if expr == buscado:
                    return True
                return any(
                    contiene(sub, buscado) for sub in expr if isinstance(sub, tuple)
                )
            return False

        assert contiene(
            resultado.expresion,
            (
                "LTE",
                "slope(USGS/SRTMGL1_003:elevation)",
                umbrales.pendiente_max_grados,
            ),
        )

    def test_amplitud_ndvi_se_escala_antes_de_comparar_con_el_umbral(self, umbrales):
        """MOD13Q1 codifica NDVI como entero crudo ([-2000, 10000]); comparar
        la amplitud cruda directamente contra un umbral en unidad real (ej.
        0.15) haría que el criterio nunca excluyera nada. La amplitud debe
        escalarse con factor_escala (0.0001) antes de la comparación."""
        from src.ingestion.gee_config import get_collection_spec

        fake_ee = _FakeEeModule()

        resultado = build_agricultural_mask_image(
            ee_module=fake_ee,
            umbrales=umbrales,
            fecha_inicio=date(2019, 7, 1),
            fecha_fin=date(2020, 6, 30),
        )

        spec_ndvi = get_collection_spec("ndvi")

        def contiene(expr, buscado) -> bool:
            if isinstance(expr, tuple):
                if expr == buscado:
                    return True
                return any(
                    contiene(sub, buscado) for sub in expr if isinstance(sub, tuple)
                )
            return False

        assert contiene(
            resultado.expresion,
            (
                "GTE",
                f"((MODIS/061/MOD13Q1:NDVI:max - MODIS/061/MOD13Q1:NDVI:min)"
                f" * {spec_ndvi.factor_escala})",
                umbrales.amplitud_ndvi_min,
            ),
        )


class TestExtractMaskedNdviSeries:
    @pytest.fixture
    def cliente_fake(self):
        class _ClienteFake:
            def __init__(self, ee_module):
                self.ee_module = ee_module

        return _ClienteFake

    def test_devuelve_mismo_esquema_que_extract_daily_series(self, umbrales, cliente_fake):
        from src.ingestion.gee_mask_pipeline import extract_masked_ndvi_series

        filas_esperadas = [
            {"provincia_id": "PUN-AZA", "fecha": "2019-07-01", "mean": 3607.0},
        ]
        fake_ee = _FakeEeModuleConMascara(filas_resultado=filas_esperadas)
        cliente = cliente_fake(fake_ee)

        resultado = extract_masked_ndvi_series(
            cliente=cliente,
            geometrias_por_provincia=object(),
            umbrales=umbrales,
            fecha_inicio=date(2019, 7, 1),
            fecha_fin=date(2019, 7, 31),
            ventana_mascara=(date(2019, 7, 1), date(2020, 6, 30)),
        )

        assert isinstance(resultado, pd.DataFrame)
        assert list(resultado.columns) == ["provincia_id", "fecha", "valor"]
        assert resultado["provincia_id"].iloc[0] == "PUN-AZA"
        assert resultado["fecha"].iloc[0] == "2019-07-01"

    def test_aplica_factor_de_escala_ndvi(self, umbrales, cliente_fake):
        from src.ingestion.gee_mask_pipeline import extract_masked_ndvi_series

        filas_esperadas = [
            {"provincia_id": "PUN-AZA", "fecha": "2019-07-01", "mean": 3607.0},
        ]
        fake_ee = _FakeEeModuleConMascara(filas_resultado=filas_esperadas)
        cliente = cliente_fake(fake_ee)

        resultado = extract_masked_ndvi_series(
            cliente=cliente,
            geometrias_por_provincia=object(),
            umbrales=umbrales,
            fecha_inicio=date(2019, 7, 1),
            fecha_fin=date(2019, 7, 31),
            ventana_mascara=(date(2019, 7, 1), date(2020, 6, 30)),
        )

        assert resultado["valor"].iloc[0] == pytest.approx(0.3607)

    def test_aplica_updatemask_antes_de_reduceregions(self, umbrales, cliente_fake):
        from src.ingestion.gee_mask_pipeline import extract_masked_ndvi_series

        filas_esperadas = [
            {"provincia_id": "PUN-AZA", "fecha": "2019-07-01", "mean": 5.0},
        ]
        fake_ee = _FakeEeModuleConMascara(filas_resultado=filas_esperadas)
        cliente = cliente_fake(fake_ee)

        extract_masked_ndvi_series(
            cliente=cliente,
            geometrias_por_provincia=object(),
            umbrales=umbrales,
            fecha_inicio=date(2019, 7, 1),
            fecha_fin=date(2019, 7, 31),
            ventana_mascara=(date(2019, 7, 1), date(2020, 6, 30)),
        )

        llamadas_reduce = [l for l in fake_ee.llamadas if l[0] == "Image.reduceRegions"]
        assert len(llamadas_reduce) == 1
        _, reducer, scale, enmascarada = llamadas_reduce[0]
        assert reducer == "MEAN_REDUCER"
        assert enmascarada is True

    def test_usa_reducer_no_nulo(self, umbrales, cliente_fake):
        from src.ingestion.gee_mask_pipeline import extract_masked_ndvi_series

        filas_esperadas = [
            {"provincia_id": "PUN-AZA", "fecha": "2019-07-01", "mean": 5.0},
        ]
        fake_ee = _FakeEeModuleConMascara(filas_resultado=filas_esperadas)
        cliente = cliente_fake(fake_ee)

        extract_masked_ndvi_series(
            cliente=cliente,
            geometrias_por_provincia=object(),
            umbrales=umbrales,
            fecha_inicio=date(2019, 7, 1),
            fecha_fin=date(2019, 7, 31),
            ventana_mascara=(date(2019, 7, 1), date(2020, 6, 30)),
        )

        llamadas_reduce = [l for l in fake_ee.llamadas if l[0] == "Image.reduceRegions"]
        assert llamadas_reduce[0][1] is not None

    def test_la_mascara_usa_la_ventana_de_campana_no_la_ventana_mensual(
        self, umbrales, cliente_fake
    ):
        """La ventana mensual (fecha_inicio/fecha_fin) pagina la extracción
        NDVI real, pero la amplitud de la máscara debe calcularse sobre la
        ventana de CAMPAÑA completa (ventana_mascara) -- si se usara la
        ventana mensual, la máscara variaría mes a mes dentro de una misma
        campaña y desalinearía el umbral calibrado sobre la ventana anual."""
        from src.ingestion.gee_mask_pipeline import extract_masked_ndvi_series

        fake_ee = _FakeEeModuleConMascara(filas_resultado=[])
        cliente = cliente_fake(fake_ee)

        ventana_mensual = (date(2019, 9, 1), date(2019, 9, 30))
        ventana_campana = (date(2019, 7, 1), date(2020, 6, 30))

        extract_masked_ndvi_series(
            cliente=cliente,
            geometrias_por_provincia=object(),
            umbrales=umbrales,
            fecha_inicio=ventana_mensual[0],
            fecha_fin=ventana_mensual[1],
            ventana_mascara=ventana_campana,
        )

        llamadas_filterdate = [
            l for l in fake_ee.llamadas if l[0] == "Ndvi.filterDate"
        ]
        # dos llamadas a filterDate sobre la colección NDVI: una para la
        # máscara (ventana de campaña) y otra para la extracción real
        # (ventana mensual) -- deben ser distintas entre sí.
        assert len(llamadas_filterdate) == 2
        ventanas_usadas = {(l[1], l[2]) for l in llamadas_filterdate}
        assert ventanas_usadas == {
            (ventana_campana[0].isoformat(), ventana_campana[1].isoformat()),
            (ventana_mensual[0].isoformat(), ventana_mensual[1].isoformat()),
        }


class _FakeSampledFeature:
    def __init__(self, propiedades: dict):
        self._propiedades = dict(propiedades)

    def getInfo(self) -> dict:
        return {"properties": self._propiedades}


class _FakeSampledFeatureCollection:
    def __init__(self, features: list[_FakeSampledFeature]):
        self._features = features

    def getInfo(self) -> dict:
        return {"features": [f.getInfo() for f in self._features]}


class _FakeCompositeImage:
    """Doble de prueba de la imagen de 4 bandas usada por sampleRegions.

    Registra en `bandas_agregadas` cada objeto pasado a `.addBands()` (antes
    de descartarlo) para que los tests puedan inspeccionar la expresión
    simbólica de cada banda (ej. `_FakeSingleBandImage.nombre`) y así
    verificar que el álgebra real (incluyendo `.multiply(factor_escala)`)
    efectivamente se ejecutó, en vez de solo confiar en las filas
    pre-cargadas que `sampleRegions` reproduce.
    """

    def __init__(self, ee_module: "_FakeEeModuleConMuestreo", filas: list[dict]):
        self._ee = ee_module
        self._filas = filas
        self.bandas_agregadas: list[Any] = []

    def select(self, banda: str) -> "_FakeCompositeImage":
        return self

    def addBands(self, otra: Any) -> "_FakeCompositeImage":
        self.bandas_agregadas.append(otra)
        return self

    def rename(self, nombres: list[str]) -> "_FakeCompositeImage":
        return self

    def sampleRegions(self, collection, scale, geometries=False):
        self._ee.llamadas.append(("sampleRegions", scale))
        return _FakeSampledFeatureCollection(
            [_FakeSampledFeature(fila) for fila in self._filas]
        )


class _FakeEeModuleConMuestreo(_FakeEeModule):
    """Extiende el doble de la máscara para simular la imagen compuesta que
    sample_cells_for_calibration exporta vía sampleRegions."""

    def __init__(self, filas_muestra: list[dict]):
        super().__init__()
        self._filas_muestra = filas_muestra
        self._imagen_compuesta = _FakeCompositeImage(self, filas_muestra)

    def Image(self, image_id: str):
        self.llamadas.append(("Image", image_id))
        if image_id in (_WORLDCOVER_ID_TEST, _SRTM_ID_TEST):
            return self._imagen_compuesta
        return super().Image(image_id)

    class Terrain:
        @staticmethod
        def slope(imagen: "_FakeCompositeImage") -> "_FakeCompositeImage":
            return imagen


_WORLDCOVER_ID_TEST = "ESA/WorldCover/v200/2021"
_SRTM_ID_TEST = "USGS/SRTMGL1_003"


class TestSampleCellsForCalibration:
    def test_devuelve_columnas_esperadas_por_apply_agricultural_mask(self):
        from src.ingestion.gee_mask_pipeline import sample_cells_for_calibration

        filas_muestra = [
            {
                "cobertura_suelo": 40,
                "altitud_m": 3800.0,
                "pendiente_grados": 10.0,
                "amplitud_ndvi": 0.30,
            },
        ]
        fake_ee = _FakeEeModuleConMuestreo(filas_muestra)

        resultado = sample_cells_for_calibration(
            ee_module=fake_ee,
            geometrias_por_provincia=object(),
            provincia_ids=["PUN-AZA"],
            fecha_inicio=date(2019, 7, 1),
            fecha_fin=date(2020, 6, 30),
            escala_muestreo_m=250,
            n_por_provincia=50,
        )

        assert list(resultado.columns) == [
            "celda_id",
            "cobertura_suelo",
            "altitud_m",
            "pendiente_grados",
            "amplitud_ndvi",
        ]

    def test_reclasifica_clase_40_como_cultivo(self):
        from src.ingestion.gee_mask_pipeline import sample_cells_for_calibration

        filas_muestra = [
            {
                "cobertura_suelo": 40,
                "altitud_m": 3800.0,
                "pendiente_grados": 10.0,
                "amplitud_ndvi": 0.30,
            },
            {
                "cobertura_suelo": 10,  # clase distinta de 40 (ej. bosque)
                "altitud_m": 3800.0,
                "pendiente_grados": 10.0,
                "amplitud_ndvi": 0.30,
            },
        ]
        fake_ee = _FakeEeModuleConMuestreo(filas_muestra)

        resultado = sample_cells_for_calibration(
            ee_module=fake_ee,
            geometrias_por_provincia=object(),
            provincia_ids=["PUN-AZA"],
            fecha_inicio=date(2019, 7, 1),
            fecha_fin=date(2020, 6, 30),
            escala_muestreo_m=250,
            n_por_provincia=50,
        )

        assert list(resultado["cobertura_suelo"]) == ["cultivo", "no_cultivo"]

    def test_asigna_celda_id_unico_por_fila(self):
        from src.ingestion.gee_mask_pipeline import sample_cells_for_calibration

        filas_muestra = [
            {
                "cobertura_suelo": 40,
                "altitud_m": 3800.0,
                "pendiente_grados": 10.0,
                "amplitud_ndvi": 0.30,
            },
            {
                "cobertura_suelo": 40,
                "altitud_m": 3900.0,
                "pendiente_grados": 12.0,
                "amplitud_ndvi": 0.25,
            },
        ]
        fake_ee = _FakeEeModuleConMuestreo(filas_muestra)

        resultado = sample_cells_for_calibration(
            ee_module=fake_ee,
            geometrias_por_provincia=object(),
            provincia_ids=["PUN-AZA"],
            fecha_inicio=date(2019, 7, 1),
            fecha_fin=date(2020, 6, 30),
            escala_muestreo_m=250,
            n_por_provincia=50,
        )

        assert resultado["celda_id"].nunique() == 2

    def test_rechaza_muestra_que_excede_5000_elementos(self):
        from src.ingestion.gee_mask_pipeline import sample_cells_for_calibration

        fake_ee = _FakeEeModuleConMuestreo(filas_muestra=[])
        provincia_ids_80 = [f"PROV-{i}" for i in range(80)]

        with pytest.raises(ValueError, match="5000|5_000"):
            sample_cells_for_calibration(
                ee_module=fake_ee,
                geometrias_por_provincia=object(),
                provincia_ids=provincia_ids_80,
                fecha_inicio=date(2019, 7, 1),
                fecha_fin=date(2020, 6, 30),
                escala_muestreo_m=250,
                n_por_provincia=70,  # 80 x 70 = 5600 > 5000
            )

    def test_acepta_muestra_justo_debajo_del_limite(self):
        from src.ingestion.gee_mask_pipeline import sample_cells_for_calibration

        fake_ee = _FakeEeModuleConMuestreo(filas_muestra=[])
        provincia_ids_80 = [f"PROV-{i}" for i in range(80)]

        resultado = sample_cells_for_calibration(
            ee_module=fake_ee,
            geometrias_por_provincia=object(),
            provincia_ids=provincia_ids_80,
            fecha_inicio=date(2019, 7, 1),
            fecha_fin=date(2020, 6, 30),
            escala_muestreo_m=250,
            n_por_provincia=50,  # 80 x 50 = 4000 < 5000
        )

        assert isinstance(resultado, pd.DataFrame)

    def test_amplitud_ndvi_se_escala_antes_de_agregarse_a_la_imagen_compuesta(self):
        """MOD13Q1 codifica NDVI como entero crudo; si se quitara el
        .multiply(factor_escala) de la banda de amplitud, ningún otro test
        de esta clase lo detectaría porque sampleRegions() reproduce las
        filas pre-cargadas del fixture sin pasar por el álgebra real. Este
        test inspecciona la banda tal como fue pasada a .addBands() (antes
        de descartarse) para verificar que el álgebra real sí se ejecutó."""
        from src.ingestion.gee_config import get_collection_spec
        from src.ingestion.gee_mask_pipeline import sample_cells_for_calibration

        fake_ee = _FakeEeModuleConMuestreo(filas_muestra=[])

        sample_cells_for_calibration(
            ee_module=fake_ee,
            geometrias_por_provincia=object(),
            provincia_ids=["PUN-AZA"],
            fecha_inicio=date(2019, 7, 1),
            fecha_fin=date(2020, 6, 30),
            escala_muestreo_m=250,
            n_por_provincia=50,
        )

        spec_ndvi = get_collection_spec("ndvi")
        imagen_compuesta = fake_ee._imagen_compuesta
        # cobertura/elevación comparten la misma _FakeCompositeImage (sin
        # .nombre); solo la banda de amplitud NDVI es un _FakeSingleBandImage
        # real, con expresión simbólica inspeccionable.
        nombres_bandas = [
            banda.nombre
            for banda in imagen_compuesta.bandas_agregadas
            if hasattr(banda, "nombre")
        ]

        banda_amplitud = next(
            nombre for nombre in nombres_bandas if "max" in nombre and "min" in nombre
        )
        assert f"* {spec_ndvi.factor_escala}" in banda_amplitud


def _evaluar_expresion_mascara(expresion: tuple, valores: dict[str, float]) -> bool:
    """Evalúa el árbol de expresión booleana producido por
    `build_agricultural_mask_image` (nodos `EQ`/`GTE`/`LTE`/`AND`) contra
    valores concretos de celda, para comparar el resultado del camino GEE
    con el de `apply_agricultural_mask` (pandas) sobre la misma celda.

    `valores` debe traer las claves `cobertura`, `altitud`, `pendiente`,
    `ndvi_max`, `ndvi_min` — los mismos 5 insumos que
    `build_agricultural_mask_image` combina.
    """
    operador = expresion[0]
    if operador == "AND":
        return _evaluar_expresion_mascara(
            expresion[1], valores
        ) and _evaluar_expresion_mascara(expresion[2], valores)

    _, nombre_simbolico, umbral = expresion
    valor = _resolver_valor_simbolico(nombre_simbolico, valores)
    if operador == "EQ":
        return valor == umbral
    if operador == "GTE":
        return valor >= umbral
    if operador == "LTE":
        return valor <= umbral
    raise ValueError(f"Operador no reconocido: {operador}")


def _resolver_valor_simbolico(nombre_simbolico: str, valores: dict[str, float]) -> float:
    """Traduce un nombre simbólico del doble de prueba (ej.
    '((MODIS/061/MOD13Q1:NDVI:max - MODIS/061/MOD13Q1:NDVI:min) * 0.0001)')
    a un valor numérico concreto, evaluando la aritmética embebida en el
    propio string (construida por `_FakeSingleBandImage.subtract`/
    `.multiply`) sobre los valores base de `ndvi_max`/`ndvi_min`."""
    if nombre_simbolico == "ESA/WorldCover/v200/2021:Map":
        return valores["cobertura"]
    if nombre_simbolico == "USGS/SRTMGL1_003:elevation":
        return valores["altitud"]
    if nombre_simbolico == "slope(USGS/SRTMGL1_003:elevation)":
        return valores["pendiente"]

    expresion_python = (
        nombre_simbolico
        .replace("MODIS/061/MOD13Q1:NDVI:max", str(valores["ndvi_max"]))
        .replace("MODIS/061/MOD13Q1:NDVI:min", str(valores["ndvi_min"]))
    )
    return eval(expresion_python)  # noqa: S307 — expresión aritmética generada internamente por los propios dobles de prueba, no input externo


class TestConsistenciaMascaraGeeYPandas:
    """Sección 'Fuera de alcance' del diseño (spec) exige verificar que
    build_agricultural_mask_image (GEE) y apply_agricultural_mask (pandas)
    son dos implementaciones consistentes de la misma regla de 4 criterios
    -- son caminos independientes (calibración/auditoría vs. producción) y
    nada impide que diverjan silenciosamente con el tiempo."""

    @pytest.fixture
    def umbrales_consistencia(self) -> MaskThresholds:
        return MaskThresholds(
            altitud_min_m=2500,
            altitud_max_m=4200,
            pendiente_max_grados=25.0,
            amplitud_ndvi_min=0.15,
        )

    @pytest.mark.parametrize(
        "celda,es_agricola_esperado",
        [
            pytest.param(
                {
                    "cobertura": 40,
                    "altitud": 3800,
                    "pendiente": 10.0,
                    "ndvi_max": 6000,
                    "ndvi_min": 2000,  # amplitud real = 0.4 >= 0.15
                },
                True,
                id="celda_agricola_cumple_los_4_criterios",
            ),
            pytest.param(
                {
                    "cobertura": 10,  # bosque, no cultivo
                    "altitud": 3800,
                    "pendiente": 10.0,
                    "ndvi_max": 6000,
                    "ndvi_min": 2000,
                },
                False,
                id="excluida_por_cobertura_no_agricola",
            ),
            pytest.param(
                {
                    "cobertura": 40,
                    "altitud": 1500,  # fuera de rango (demasiado baja)
                    "pendiente": 10.0,
                    "ndvi_max": 6000,
                    "ndvi_min": 2000,
                },
                False,
                id="excluida_por_altitud_fuera_de_rango",
            ),
            pytest.param(
                {
                    "cobertura": 40,
                    "altitud": 3800,
                    "pendiente": 40.0,  # excesiva
                    "ndvi_max": 6000,
                    "ndvi_min": 2000,
                },
                False,
                id="excluida_por_pendiente_excesiva",
            ),
            pytest.param(
                {
                    "cobertura": 40,
                    "altitud": 3800,
                    "pendiente": 10.0,
                    "ndvi_max": 2500,
                    "ndvi_min": 2000,  # amplitud real = 0.05 < 0.15
                },
                False,
                id="excluida_por_amplitud_ndvi_insuficiente",
            ),
        ],
    )
    def test_gee_y_pandas_coinciden_en_la_misma_celda(
        self, umbrales_consistencia, celda, es_agricola_esperado
    ):
        from src.preprocessing.mask import apply_agricultural_mask

        fake_ee = _FakeEeModule()
        mascara = build_agricultural_mask_image(
            ee_module=fake_ee,
            umbrales=umbrales_consistencia,
            fecha_inicio=date(2019, 7, 1),
            fecha_fin=date(2020, 6, 30),
        )
        resultado_gee = _evaluar_expresion_mascara(mascara.expresion, celda)

        celdas_df = pd.DataFrame(
            [
                {
                    "celda_id": "c1",
                    "cobertura_suelo": "cultivo" if celda["cobertura"] == 40 else "no_cultivo",
                    "altitud_m": celda["altitud"],
                    "pendiente_grados": celda["pendiente"],
                    "amplitud_ndvi": (celda["ndvi_max"] - celda["ndvi_min"]) * 0.0001,
                }
            ]
        )
        resultado_pandas = apply_agricultural_mask(celdas_df, umbrales_consistencia)[
            "es_agricola"
        ].iloc[0]

        assert resultado_gee == es_agricola_esperado
        assert resultado_pandas == es_agricola_esperado
        assert resultado_gee == resultado_pandas
