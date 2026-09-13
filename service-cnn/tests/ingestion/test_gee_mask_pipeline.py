"""Pruebas de la máscara agrícola server-side en Google Earth Engine.

Referencia: service-cnn/docs/superpowers/specs/2026-09-13-mascara-agricola-gee-design.md

No requiere `earthengine-api` instalado ni credenciales: recibe el módulo
`ee` inyectado (doble de prueba), mismo patrón que test_gee_client.py.
"""
from __future__ import annotations

from datetime import date

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
            date(2019, 7, 1),
            date(2020, 6, 30),
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
