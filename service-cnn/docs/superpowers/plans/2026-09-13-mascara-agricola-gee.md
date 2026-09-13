# Máscara Agrícola por Celda en GEE — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reimplementar los 4 criterios de `apply_agricultural_mask` como una máscara server-side en Google Earth Engine (`ee.Image` booleana), para poder calcular NDVI enmascarado agregado por provincia-campaña sin traer celdas individuales a pandas — y dar un camino de calibración/auditoría acotado con una muestra de celdas.

**Architecture:** Módulo nuevo `src/ingestion/gee_mask_pipeline.py`, mismo patrón de inyección de `ee` que `gee_client.py` (no se importa `ee` a nivel de módulo; se recibe inyectado, real o doble de prueba). Tres funciones públicas: `build_agricultural_mask_image` (construye la `ee.Image` booleana), `extract_masked_ndvi_series` (aplica la máscara antes de `reduceRegions`, mismo contrato de salida que `GeeClient.extract_daily_series`), y `sample_cells_for_calibration` (exporta una muestra acotada de celdas para calibración/auditoría con `mask.py`, ya existente y sin cambios). `gee_extraction_pipeline.extract_all_provinces_all_variables` gana un parámetro opcional para enrutar `"ndvi"` a `extract_masked_ndvi_series` cuando se le pasan umbrales.

**Tech Stack:** Python 3.14, pandas, pytest, `earthengine-api` (inyectado/doblado en tests, nunca importado a nivel de módulo).

## Global Constraints

- `ee` nunca se importa a nivel de módulo — se recibe inyectado (real o doble de prueba), igual que `gee_client.py` y `province_geometries.py`.
- Todo test corre sin credenciales de GEE ni red: dobles de prueba en memoria, mismo estilo que `tests/ingestion/test_gee_client.py`.
- `src/preprocessing/mask.py` (`apply_agricultural_mask`, `calibrate_ndvi_amplitude_threshold`, `MaskThresholds`) no se modifica — se reutiliza tal cual.
- El esquema de salida de `extract_masked_ndvi_series` debe ser exactamente `provincia_id, fecha, valor` (columnas y tipos), igual que `GeeClient.extract_daily_series`, para no romper `gee_series_builder.aggregate_by_phase` ni `tensor_builder.py`.
- `sample_cells_for_calibration` debe validar `n_por_provincia * len(provincia_ids) < 5000` y lanzar `ValueError` explícito si se excede — límite real de `ee.FeatureCollection.getInfo()` ya documentado en `gee_extraction_pipeline.py`.
- Cobertura de suelo: `ESA/WorldCover/v200/2021`, banda `Map`, clase `40` = agrícola/cropland.
- MDE/pendiente: `USGS/SRTMGL1_003`, banda `elevation`; pendiente vía `ee.Terrain.slope()`.
- NDVI: `MODIS/061/MOD13Q1`, banda `NDVI`, `factor_escala=0.0001` (ya en `gee_config.get_collection_spec("ndvi")` — reutilizar, no hardcodear de nuevo).
- Amplitud NDVI (máx-mín) se calcula sobre la ventana de extracción de cada campaña, no un año fijo.
- Referencia de diseño completa: `service-cnn/docs/superpowers/specs/2026-09-13-mascara-agricola-gee-design.md`.

---

## Task 1: `build_agricultural_mask_image` — máscara booleana combinada

**Files:**
- Create: `service-cnn/src/ingestion/gee_mask_pipeline.py`
- Test: `service-cnn/tests/ingestion/test_gee_mask_pipeline.py`

**Interfaces:**
- Consumes: `src.preprocessing.mask.MaskThresholds` (ya existe: campos `altitud_min_m`, `altitud_max_m`, `pendiente_max_grados`, `amplitud_ndvi_min`). `src.ingestion.gee_config.get_collection_spec("ndvi")` (ya existe: expone `.collection_id`, `.band`, `.factor_escala`, `.offset_aditivo`).
- Produces: `build_agricultural_mask_image(ee_module, umbrales: MaskThresholds, fecha_inicio: date, fecha_fin: date) -> Any` (el tipo de retorno es opaco — una `ee.Image` real o el doble de prueba `_FakeMaskImage` que las siguientes tareas consumirán vía su método `.getInfo()`-like registrado en `llamadas`).

Para que esta tarea sea testeable sin la API real, el doble de prueba de `ee` debe soportar el álgebra mínima usada por `build_agricultural_mask_image`: `Image(id).select(banda)`, comparaciones (`.eq()`, `.gte()`, `.lte()`), `.And()` encadenado, `ee.Terrain.slope(imagen)`, e `ImageCollection(id).filterDate(...).select(...).max()`/`.min()`. Se define aquí, en el archivo de test, y se reutiliza en las Tasks 2 y 3 (mismo archivo de test, no se duplica).

- [ ] **Step 1: Escribir el doble de prueba de `ee` con álgebra de imágenes**

Crear `service-cnn/tests/ingestion/test_gee_mask_pipeline.py` con el siguiente contenido inicial (doble de prueba + primer test, en rojo):

```python
"""Pruebas de la máscara agrícola server-side en Google Earth Engine.

Referencia: service-cnn/docs/superpowers/specs/2026-09-13-mascara-agricola-gee-design.md

No requiere `earthengine-api` instalado ni credenciales: recibe el módulo
`ee` inyectado (doble de prueba), mismo patrón que test_gee_client.py.
"""
from __future__ import annotations

from datetime import date

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
        # donde altitud es en sí un AND interno de gte/lte (5 hojas totales:
        # cobertura==40, altitud>=min, altitud<=max, pendiente<=max,
        # amplitud>=min — 5 comparaciones booleanas, no 4, porque "altitud en
        # rango" no es una comparación atómica sino la conjunción de dos).
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
```

- [ ] **Step 2: Ejecutar el test para verificar que falla**

Run: `cd service-cnn && python -m pytest tests/ingestion/test_gee_mask_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ingestion.gee_mask_pipeline'` (o `ImportError`).

- [ ] **Step 3: Implementar `build_agricultural_mask_image`**

Crear `service-cnn/src/ingestion/gee_mask_pipeline.py`:

```python
"""Máscara de superficie agrícola server-side en Google Earth Engine.

Referencia: service-cnn/docs/superpowers/specs/2026-09-13-mascara-agricola-gee-design.md

Reimplementa los 4 criterios de `src.preprocessing.mask.apply_agricultural_mask`
como una `ee.Image` booleana (cobertura de suelo + altitud + pendiente +
amplitud NDVI), evaluada server-side antes de `reduceRegions` — necesario
porque a resolución de celda (NDVI 250 m, SRTM ~30 m) sobre 80 provincias el
volumen de píxeles excede ampliamente el límite de 5000 elementos de
`ee.FeatureCollection.getInfo()` (ya documentado en gee_extraction_pipeline.py),
lo que hace inviable traer celdas individuales a pandas para el flujo de
producción. `apply_agricultural_mask` (pandas) se conserva sin cambios para
calibración y auditoría con una muestra acotada de celdas (ver
`sample_cells_for_calibration`).

No importa `ee` a nivel de módulo: recibe el módulo `ee` (real o doble de
prueba) inyectado, mismo patrón que `gee_client.py`.
"""
from __future__ import annotations

from datetime import date
from typing import Any

from src.ingestion.gee_config import get_collection_spec
from src.preprocessing.mask import MaskThresholds

_WORLDCOVER_ID = "ESA/WorldCover/v200/2021"
_SRTM_ID = "USGS/SRTMGL1_003"
_CLASE_WORLDCOVER_CROPLAND = 40


def build_agricultural_mask_image(
    ee_module: Any,
    umbrales: MaskThresholds,
    fecha_inicio: date,
    fecha_fin: date,
) -> Any:
    """Construye la máscara agrícola booleana combinando los 4 criterios.

    Args:
        ee_module: el módulo `ee` real, ya inicializado, o un doble de
            prueba con la misma interfaz mínima usada aquí.
        umbrales: `MaskThresholds` ya calibrados (mismo dataclass que usa
            `apply_agricultural_mask` en pandas).
        fecha_inicio, fecha_fin: ventana de extracción de la campaña
            (ver `gee_extraction_pipeline.campana_a_ventana_extraccion`) —
            la amplitud NDVI se calcula sobre esta ventana específica, no
            sobre un año fijo, porque el ciclo agrícola cambia campaña a
            campaña.

    Returns:
        Una `ee.Image` de una banda booleana: `True` solo en las celdas que
        cumplen los 4 criterios (cobertura agrícola, altitud en rango,
        pendiente aceptable, amplitud NDVI suficiente).
    """
    cobertura = ee_module.Image(_WORLDCOVER_ID).select("Map")
    es_cultivo = cobertura.eq(_CLASE_WORLDCOVER_CROPLAND)

    elevacion = ee_module.Image(_SRTM_ID).select("elevation")
    altitud_en_rango = elevacion.gte(umbrales.altitud_min_m).And(
        elevacion.lte(umbrales.altitud_max_m)
    )

    pendiente = ee_module.Terrain.slope(elevacion)
    pendiente_aceptable = pendiente.lte(umbrales.pendiente_max_grados)

    spec_ndvi = get_collection_spec("ndvi")
    coleccion_ndvi = (
        ee_module.ImageCollection(spec_ndvi.collection_id)
        .filterDate(fecha_inicio, fecha_fin)
        .select(spec_ndvi.band)
    )
    amplitud = coleccion_ndvi.max().subtract(coleccion_ndvi.min())
    amplitud_suficiente = amplitud.gte(umbrales.amplitud_ndvi_min)

    return (
        es_cultivo.And(altitud_en_rango)
        .And(pendiente_aceptable)
        .And(amplitud_suficiente)
    )
```

- [ ] **Step 4: Ejecutar los tests para verificar que pasan**

Run: `cd service-cnn && python -m pytest tests/ingestion/test_gee_mask_pipeline.py -v`
Expected: PASS — 6 tests en `TestBuildAgriculturalMaskImage`.

- [ ] **Step 5: Commit**

```bash
cd service-cnn
git add src/ingestion/gee_mask_pipeline.py tests/ingestion/test_gee_mask_pipeline.py
git commit -m "$(cat <<'EOF'
Agregar build_agricultural_mask_image: mascara agricola booleana en GEE

Combina WorldCover (cobertura), SRTM/slope (altitud y pendiente) y
amplitud NDVI de la campana en una sola ee.Image booleana, server-side,
evitando traer celdas individuales a pandas.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016mkokyMQFzBE7QpPxKea4U
EOF
)"
```

---

## Task 2: `extract_masked_ndvi_series` — NDVI enmascarado agregado por provincia

**Files:**
- Modify: `service-cnn/src/ingestion/gee_mask_pipeline.py`
- Test: `service-cnn/tests/ingestion/test_gee_mask_pipeline.py`

**Interfaces:**
- Consumes: `build_agricultural_mask_image(ee_module, umbrales, fecha_inicio, fecha_fin) -> Any` (Task 1). `src.ingestion.gee_config.get_collection_spec("ndvi")` (ya existe).
- Produces: `extract_masked_ndvi_series(cliente: Any, geometrias_por_provincia: Any, umbrales: MaskThresholds, fecha_inicio: date, fecha_fin: date) -> pd.DataFrame` con columnas `provincia_id, fecha, valor` (mismo contrato que `GeeClient.extract_daily_series`). `cliente` es un `GeeClient` ya inyectado — esta función usa `cliente.ee_module` para acceder al módulo `ee` (ver `GeeClient.ee_module` en `gee_client.py:42-50`), no reimplementa su propia inicialización.

El doble de prueba de `ee` de esta tarea necesita, además de lo de Task 1, soportar `.updateMask()` sobre la imagen NDVI y el flujo `ImageCollection.map().flatten()` con `reduceRegions` (igual patrón que `test_gee_client.py`). Se extiende `_FakeEeModule` en el mismo archivo de test.

- [ ] **Step 1: Extender el doble de prueba con `updateMask` + `reduceRegions` sobre NDVI, y escribir el test en rojo**

Añadir a `service-cnn/tests/ingestion/test_gee_mask_pipeline.py`, antes de `TestBuildAgriculturalMaskImage`, las clases auxiliares de reducción (idénticas en espíritu a `test_gee_client.py`, adaptadas a que cada imagen NDVI se pueda enmascarar):

```python
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
```

Nota: `_FakeEeModuleConMascara.ImageCollection` devuelve `self._coleccion_ndvi_extraccion` para **toda** llamada a `MODIS/061/MOD13Q1` — incluida la que hace `build_agricultural_mask_image` para calcular la amplitud. Esto es aceptable porque `_FakeNdviImageCollection.select()`/`.filterDate()` son no-op que devuelven `self`, y sus `.max()`/`.min()` no están definidos: si `extract_masked_ndvi_series` invoca `build_agricultural_mask_image` primero y luego reconstruye una colección NDVI aparte para la extracción, ambas rutas deben pasar por el mismo objeto sin chocar. Si al implementar aparece un `AttributeError` por esto, añadir `.max()`/`.min()` a `_FakeNdviImageCollection` que reutilicen `_FakeSingleBandImage` de Task 1 (import cruzado dentro del mismo archivo de test, ambas clases ya están definidas ahí).

Ahora el test:

```python
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
        )

        llamadas_reduce = [l for l in fake_ee.llamadas if l[0] == "Image.reduceRegions"]
        assert llamadas_reduce[0][1] is not None
```

- [ ] **Step 2: Ejecutar los tests para verificar que fallan**

Run: `cd service-cnn && python -m pytest tests/ingestion/test_gee_mask_pipeline.py::TestExtractMaskedNdviSeries -v`
Expected: FAIL — `ImportError: cannot import name 'extract_masked_ndvi_series'`.

- [ ] **Step 3: Implementar `extract_masked_ndvi_series`**

Añadir a `service-cnn/src/ingestion/gee_mask_pipeline.py`:

```python
import pandas as pd


def extract_masked_ndvi_series(
    cliente: Any,
    geometrias_por_provincia: Any,
    umbrales: MaskThresholds,
    fecha_inicio: date,
    fecha_fin: date,
) -> "pd.DataFrame":
    """Extrae NDVI agregado por provincia, calculado solo sobre celdas agrícolas.

    Args:
        cliente: `GeeClient` ya inicializado (real o doble de prueba) — se
            usa `cliente.ee_module` para acceder a la sesión de Earth Engine
            ya inicializada, mismo patrón que `province_geometries.
            build_province_feature_collection`.
        geometrias_por_provincia: `ee.FeatureCollection` con un feature por
            provincia (ver `province_geometries.py`).
        umbrales: `MaskThresholds` ya calibrados.
        fecha_inicio, fecha_fin: ventana de extracción de la campaña.

    Returns:
        DataFrame con columnas `provincia_id`, `fecha`, `valor` — mismo
        contrato que `GeeClient.extract_daily_series`, con `valor` ya
        convertido a NDVI real ([-1, 1]) vía el `factor_escala` del spec.
    """
    ee = cliente.ee_module
    mascara = build_agricultural_mask_image(ee, umbrales, fecha_inicio, fecha_fin)

    spec = get_collection_spec("ndvi")
    coleccion = (
        ee.ImageCollection(spec.collection_id)
        .filterDate(fecha_inicio.isoformat(), fecha_fin.isoformat())
        .select(spec.band)
    )

    def _reducir_imagen_enmascarada(imagen):
        fecha = imagen.date().format("YYYY-MM-dd")
        imagen_enmascarada = imagen.updateMask(mascara)
        reducido = imagen_enmascarada.reduceRegions(
            collection=geometrias_por_provincia,
            reducer=ee.Reducer.mean(),
            scale=spec.spatial_resolution_m,
        )
        return reducido.map(lambda f: f.set("fecha", fecha))

    resultado_aplanado = coleccion.map(_reducir_imagen_enmascarada).flatten()

    filas = [f["properties"] for f in resultado_aplanado.getInfo()["features"]]
    tabla = pd.DataFrame(filas)
    if tabla.empty:
        return pd.DataFrame(columns=["provincia_id", "fecha", "valor"])
    tabla = tabla.rename(columns={"mean": "valor"})[["provincia_id", "fecha", "valor"]]
    tabla["valor"] = tabla["valor"] * spec.factor_escala + spec.offset_aditivo
    return tabla
```

Mover el `import pandas as pd` al encabezado del archivo (junto a los demás imports), no dejarlo inline dentro de la función.

- [ ] **Step 4: Ejecutar los tests para verificar que pasan**

Run: `cd service-cnn && python -m pytest tests/ingestion/test_gee_mask_pipeline.py -v`
Expected: PASS — los 7 tests de Task 1 + los 4 de `TestExtractMaskedNdviSeries`.

Si aparece `AttributeError` por `.max()`/`.min()` faltantes en `_FakeNdviImageCollection` (ver nota del Step 1), añadir a esa clase:

```python
    def max(self):
        return _FakeSingleBandImage(self._ee, "MODIS/061/MOD13Q1:NDVI:max")

    def min(self):
        return _FakeSingleBandImage(self._ee, "MODIS/061/MOD13Q1:NDVI:min")
```

- [ ] **Step 5: Commit**

```bash
cd service-cnn
git add src/ingestion/gee_mask_pipeline.py tests/ingestion/test_gee_mask_pipeline.py
git commit -m "$(cat <<'EOF'
Agregar extract_masked_ndvi_series: NDVI agregado sobre celdas agricolas

Aplica la mascara de build_agricultural_mask_image via updateMask antes
de reduceRegions, devolviendo el mismo esquema provincia_id/fecha/valor
que GeeClient.extract_daily_series -- no rompe gee_series_builder ni
tensor_builder aguas abajo.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016mkokyMQFzBE7QpPxKea4U
EOF
)"
```

---

## Task 3: `sample_cells_for_calibration` — muestra acotada para `mask.py`

**Files:**
- Modify: `service-cnn/src/ingestion/gee_mask_pipeline.py`
- Test: `service-cnn/tests/ingestion/test_gee_mask_pipeline.py`

**Interfaces:**
- Consumes: nada de Tasks 1-2 directamente (construye su propia imagen compuesta de 4 bandas), pero reutiliza las mismas fuentes (`ESA/WorldCover/v200/2021`, `USGS/SRTMGL1_003`, `MODIS/061/MOD13Q1`).
- Produces: `sample_cells_for_calibration(ee_module: Any, geometrias_por_provincia: Any, provincia_ids: list[str], fecha_inicio: date, fecha_fin: date, escala_muestreo_m: int, n_por_provincia: int) -> pd.DataFrame` con columnas `celda_id, cobertura_suelo, altitud_m, pendiente_grados, amplitud_ndvi` — exactamente las que espera `apply_agricultural_mask` (`src/preprocessing/mask.py:33-51`, fixture `celdas_candidatas` en `tests/preprocessing/test_mask.py:34-73`). `cobertura_suelo` debe ser el string `"cultivo"` cuando la clase WorldCover es 40, y `"no_cultivo"` en cualquier otro caso (`_COBERTURAS_AGRICOLAS = {"cultivo"}` en `mask.py:20`).

- [ ] **Step 1: Escribir los tests en rojo**

Añadir a `service-cnn/tests/ingestion/test_gee_mask_pipeline.py`:

```python
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
    """Doble de prueba de la imagen de 4 bandas usada por sampleRegions."""

    def __init__(self, ee_module: "_FakeEeModuleConMuestreo", filas: list[dict]):
        self._ee = ee_module
        self._filas = filas

    def addBands(self, otra: "_FakeCompositeImage") -> "_FakeCompositeImage":
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
```

- [ ] **Step 2: Ejecutar los tests para verificar que fallan**

Run: `cd service-cnn && python -m pytest tests/ingestion/test_gee_mask_pipeline.py::TestSampleCellsForCalibration -v`
Expected: FAIL — `ImportError: cannot import name 'sample_cells_for_calibration'`.

- [ ] **Step 3: Implementar `sample_cells_for_calibration`**

Añadir a `service-cnn/src/ingestion/gee_mask_pipeline.py`:

```python
def sample_cells_for_calibration(
    ee_module: Any,
    geometrias_por_provincia: Any,
    provincia_ids: list[str],
    fecha_inicio: date,
    fecha_fin: date,
    escala_muestreo_m: int,
    n_por_provincia: int,
) -> "pd.DataFrame":
    """Exporta una muestra acotada de celdas para calibrar/auditar la máscara.

    Uso: alimentar `src.preprocessing.mask.apply_agricultural_mask` (para
    auditar cuántas celdas caen en cada motivo de exclusión) y
    `calibrate_ndvi_amplitude_threshold` (para fijar `amplitud_ndvi_min`
    sobre campañas de entrenamiento) — nunca para el flujo de producción,
    que usa `extract_masked_ndvi_series` (server-side, sin traer celdas).

    Args:
        ee_module: el módulo `ee` real, ya inicializado, o un doble de
            prueba.
        geometrias_por_provincia: `ee.FeatureCollection` con un feature por
            provincia.
        provincia_ids: provincias incluidas en `geometrias_por_provincia` —
            usado solo para validar el tamaño total de la muestra contra el
            límite de `.getInfo()`.
        fecha_inicio, fecha_fin: ventana de extracción de la campaña (debe
            restringirse a campañas de ENTRENAMIENTO cuando el resultado se
            use para calibrar `amplitud_ndvi_min` — responsabilidad del
            llamador, misma disciplina anti-fuga que
            `calibrate_ndvi_amplitude_threshold`).
        escala_muestreo_m: resolución de muestreo en metros, pasada a
            `sampleRegions`.
        n_por_provincia: máximo de celdas a muestrear por provincia.

    Returns:
        DataFrame con columnas `celda_id`, `cobertura_suelo` (`"cultivo"` o
        `"no_cultivo"`), `altitud_m`, `pendiente_grados`, `amplitud_ndvi` —
        exactamente el esquema que espera `apply_agricultural_mask`.

    Raises:
        ValueError: si `n_por_provincia * len(provincia_ids)` alcanza o
            excede 5000 — límite real de `ee.FeatureCollection.getInfo()`
            (`Collection query aborted after accumulating over 5000
            elements`, ya documentado en `gee_extraction_pipeline.
            dividir_en_meses`).
    """
    tamano_muestra_total = n_por_provincia * len(provincia_ids)
    if tamano_muestra_total >= 5000:
        raise ValueError(
            f"n_por_provincia ({n_por_provincia}) x provincias "
            f"({len(provincia_ids)}) = {tamano_muestra_total} alcanza o "
            "excede el límite de 5000 elementos de "
            "ee.FeatureCollection.getInfo(). Reduzca n_por_provincia."
        )

    cobertura = ee_module.Image(_WORLDCOVER_ID).select("Map").rename(["cobertura_suelo"])
    elevacion = ee_module.Image(_SRTM_ID).select("elevation").rename(["altitud_m"])
    pendiente = ee_module.Terrain.slope(elevacion).rename(["pendiente_grados"])

    spec_ndvi = get_collection_spec("ndvi")
    coleccion_ndvi = (
        ee_module.ImageCollection(spec_ndvi.collection_id)
        .filterDate(fecha_inicio, fecha_fin)
        .select(spec_ndvi.band)
    )
    amplitud = (
        coleccion_ndvi.max()
        .subtract(coleccion_ndvi.min())
        .rename(["amplitud_ndvi"])
    )

    imagen_compuesta = (
        cobertura.addBands(elevacion).addBands(pendiente).addBands(amplitud)
    )

    muestra = imagen_compuesta.sampleRegions(
        collection=geometrias_por_provincia,
        scale=escala_muestreo_m,
        geometries=False,
    )

    filas = [f["properties"] for f in muestra.getInfo()["features"]]
    tabla = pd.DataFrame(filas)
    if tabla.empty:
        return pd.DataFrame(
            columns=[
                "celda_id",
                "cobertura_suelo",
                "altitud_m",
                "pendiente_grados",
                "amplitud_ndvi",
            ]
        )

    tabla["celda_id"] = [f"celda_{i}" for i in range(len(tabla))]
    tabla["cobertura_suelo"] = tabla["cobertura_suelo"].apply(
        lambda clase: "cultivo" if clase == _CLASE_WORLDCOVER_CROPLAND else "no_cultivo"
    )
    return tabla[
        ["celda_id", "cobertura_suelo", "altitud_m", "pendiente_grados", "amplitud_ndvi"]
    ]
```

Nota: `rename(["banda"])` y `addBands` en la implementación real de `ee.Image` reciben/devuelven imágenes multi-banda; el doble de prueba (`_FakeCompositeImage`) los trata como no-op porque ya construye las filas planas directamente — es una simplificación deliberada del doble, análoga a cómo `_FakeFeatureCollection.flatten()` en `test_gee_client.py` también es no-op.

- [ ] **Step 4: Ejecutar los tests para verificar que pasan**

Run: `cd service-cnn && python -m pytest tests/ingestion/test_gee_mask_pipeline.py -v`
Expected: PASS — todos los tests del archivo (Task 1 + 2 + 3).

- [ ] **Step 5: Commit**

```bash
cd service-cnn
git add src/ingestion/gee_mask_pipeline.py tests/ingestion/test_gee_mask_pipeline.py
git commit -m "$(cat <<'EOF'
Agregar sample_cells_for_calibration: muestra acotada para mask.py

Exporta una muestra de celdas (cobertura, altitud, pendiente, amplitud
NDVI) via sampleRegions, respetando el limite de 5000 elementos de
getInfo(). Alimenta apply_agricultural_mask y
calibrate_ndvi_amplitude_threshold (mask.py, sin cambios) para
calibracion/auditoria -- nunca el flujo de produccion.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016mkokyMQFzBE7QpPxKea4U
EOF
)"
```

---

## Task 4: Enrutar NDVI enmascarado desde `extract_all_provinces_all_variables`

**Files:**
- Modify: `service-cnn/src/ingestion/gee_extraction_pipeline.py`
- Modify: `service-cnn/tests/ingestion/test_gee_extraction_pipeline.py`

**Interfaces:**
- Consumes: `extract_masked_ndvi_series(cliente, geometrias_por_provincia, umbrales, fecha_inicio, fecha_fin) -> pd.DataFrame` (Task 2). `src.preprocessing.mask.MaskThresholds`.
- Produces: `extract_all_provinces_all_variables(..., umbrales_mascara: MaskThresholds | None = None)` — mismo contrato de retorno que hoy (no cambia). Cuando `umbrales_mascara` no es `None` y la tarea es de la variable `"ndvi"`, se llama `extract_masked_ndvi_series(cliente=cliente, ...)` en vez de `cliente.extract_daily_series(...)`; el resto de variables no se ve afectado.

- [ ] **Step 1: Escribir el test en rojo**

Añadir a `service-cnn/tests/ingestion/test_gee_extraction_pipeline.py`, dentro de `TestExtractAllProvincesAllVariables`:

```python
    def test_usa_extract_masked_ndvi_series_para_ndvi_si_se_pasan_umbrales(
        self, geometrias_fake, monkeypatch
    ):
        """Cuando se pasan umbrales_mascara, la variable 'ndvi' debe pasar
        por extract_masked_ndvi_series (NDVI agregado sobre celdas
        agrícolas) en vez de cliente.extract_daily_series -- las demás
        variables no cambian de camino."""
        from src.preprocessing.mask import MaskThresholds

        umbrales = MaskThresholds(
            altitud_min_m=2500,
            altitud_max_m=4200,
            pendiente_max_grados=25.0,
            amplitud_ndvi_min=0.15,
        )

        llamadas_masked = []

        def _fake_extract_masked_ndvi_series(
            cliente, geometrias_por_provincia, umbrales, fecha_inicio, fecha_fin
        ):
            llamadas_masked.append((fecha_inicio, fecha_fin))
            return pd.DataFrame(
                {
                    "provincia_id": ["ABANCAY"],
                    "fecha": ["2019-07-01"],
                    "valor": [0.5],
                }
            )

        monkeypatch.setattr(
            "src.ingestion.gee_extraction_pipeline.extract_masked_ndvi_series",
            _fake_extract_masked_ndvi_series,
        )

        cliente = MagicMock()
        cliente.extract_daily_series.return_value = pd.DataFrame(
            {
                "provincia_id": ["ABANCAY"],
                "fecha": ["2019-07-01"],
                "valor": [1.0],
            }
        )

        resultado = extract_all_provinces_all_variables(
            cliente=cliente,
            geometrias_por_provincia=geometrias_fake,
            variables=["precipitacion", "ndvi"],
            campanas=[2020],
            umbrales_mascara=umbrales,
        )

        # 12 meses de ndvi via extract_masked_ndvi_series
        assert len(llamadas_masked) == 12
        # precipitacion (12 meses) sigue via extract_daily_series
        assert cliente.extract_daily_series.call_count == 12
        assert (resultado["variable"] == "ndvi").sum() == 12
        assert (resultado["variable"] == "precipitacion").sum() == 12

    def test_sin_umbrales_mascara_ndvi_usa_extract_daily_series_como_antes(
        self, cliente_fake, geometrias_fake
    ):
        """Compatibilidad hacia atrás: sin umbrales_mascara (default None),
        el comportamiento no cambia respecto al código ya existente."""
        resultado = extract_all_provinces_all_variables(
            cliente=cliente_fake,
            geometrias_por_provincia=geometrias_fake,
            variables=["ndvi"],
            campanas=[2020],
        )

        assert cliente_fake.extract_daily_series.call_count == 12
        assert len(resultado) == 24
```

- [ ] **Step 2: Ejecutar los tests para verificar que fallan**

Run: `cd service-cnn && python -m pytest tests/ingestion/test_gee_extraction_pipeline.py::TestExtractAllProvincesAllVariables::test_usa_extract_masked_ndvi_series_para_ndvi_si_se_pasan_umbrales -v`
Expected: FAIL — `TypeError: extract_all_provinces_all_variables() got an unexpected keyword argument 'umbrales_mascara'`.

- [ ] **Step 3: Modificar `extract_all_provinces_all_variables`**

En `service-cnn/src/ingestion/gee_extraction_pipeline.py`, añadir el import y modificar la firma y el cuerpo de la función (líneas 74-175 actuales):

```python
from src.ingestion.gee_mask_pipeline import extract_masked_ndvi_series
from src.preprocessing.mask import MaskThresholds
```

(añadir junto a los imports existentes al inicio del archivo, después de `import pandas as pd`)

Modificar la firma de `extract_all_provinces_all_variables`:

```python
def extract_all_provinces_all_variables(
    cliente: Any,
    geometrias_por_provincia: Any,
    variables: list[str],
    campanas: list[int],
    devolver_fallos: bool = False,
    max_workers: int = 1,
    umbrales_mascara: Optional[MaskThresholds] = None,
) -> pd.DataFrame | tuple[pd.DataFrame, list[dict]]:
```

Y añadir `umbrales_mascara` al docstring de `Args` (después de `max_workers`):

```
        umbrales_mascara: si se proporciona, la variable "ndvi" se extrae
            vía `extract_masked_ndvi_series` (NDVI agregado solo sobre
            celdas agrícolas, ver `gee_mask_pipeline.py`) en vez de
            `cliente.extract_daily_series`. Las demás variables no se ven
            afectadas. `None` (por defecto) preserva el comportamiento
            actual para todas las variables, incluida "ndvi".
```

Modificar `_ejecutar_tarea` para enrutar según la variable:

```python
    def _ejecutar_tarea(tarea: tuple[str, int, date, date]) -> tuple[tuple, Any]:
        variable, campana_id, fecha_inicio, fecha_fin = tarea
        resultado_o_excepcion: Any
        try:
            if variable == "ndvi" and umbrales_mascara is not None:
                resultado_o_excepcion = extract_masked_ndvi_series(
                    cliente=cliente,
                    geometrias_por_provincia=geometrias_por_provincia,
                    umbrales=umbrales_mascara,
                    fecha_inicio=fecha_inicio,
                    fecha_fin=fecha_fin,
                )
            else:
                resultado_o_excepcion = cliente.extract_daily_series(
                    variable=variable,
                    geometrias_por_provincia=geometrias_por_provincia,
                    fecha_inicio=fecha_inicio,
                    fecha_fin=fecha_fin,
                )
        except Exception as exc:  # noqa: BLE001 — se reporta, no se enmascara
            resultado_o_excepcion = exc
        return tarea, resultado_o_excepcion
```

- [ ] **Step 4: Ejecutar los tests para verificar que pasan**

Run: `cd service-cnn && python -m pytest tests/ingestion/test_gee_extraction_pipeline.py -v`
Expected: PASS — todos los tests existentes siguen pasando + los 2 nuevos.

- [ ] **Step 5: Ejecutar toda la suite de tests de `service-cnn` para verificar que nada más se rompió**

Run: `cd service-cnn && python -m pytest -v`
Expected: PASS — 256+ tests (256 previos + los nuevos de esta plan).

- [ ] **Step 6: Commit**

```bash
cd service-cnn
git add src/ingestion/gee_extraction_pipeline.py tests/ingestion/test_gee_extraction_pipeline.py
git commit -m "$(cat <<'EOF'
Enrutar NDVI a extract_masked_ndvi_series cuando se pasan umbrales

extract_all_provinces_all_variables gana el parametro opcional
umbrales_mascara: MaskThresholds | None. Cuando se provee, la variable
"ndvi" usa extract_masked_ndvi_series (NDVI sobre celdas agricolas) en
vez de cliente.extract_daily_series; las demas variables no cambian.
None preserva el comportamiento actual.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016mkokyMQFzBE7QpPxKea4U
EOF
)"
```

---

## Task 5: Actualizar `PENDIENTE.md`

**Files:**
- Modify: `service-cnn/docs/PENDIENTE.md`

**Interfaces:** ninguna — solo documentación.

- [ ] **Step 1: Marcar la sub-parte de extracción GEE por celda como resuelta y anotar lo que queda**

En `service-cnn/docs/PENDIENTE.md`, reemplazar la sección `### 2.1 Máscara de superficie agrícola (bloquea mask.py en producción)` (líneas 20-26 actuales) por:

```markdown
### 2.1 Máscara de superficie agrícola — extracción GEE ya implementada, falta calibrar y conectar a producción

`src/ingestion/gee_mask_pipeline.py` ya implementa la extracción server-side
por celda (diseño: `docs/superpowers/specs/2026-09-13-mascara-agricola-gee-design.md`;
plan: `docs/superpowers/plans/2026-09-13-mascara-agricola-gee.md`):
- `build_agricultural_mask_image`: máscara booleana (WorldCover v200 2021 +
  SRTM/slope + amplitud NDVI por campaña).
- `extract_masked_ndvi_series`: NDVI agregado por provincia-campaña, ya
  enmascarado — mismo contrato que `GeeClient.extract_daily_series`.
- `sample_cells_for_calibration`: muestra acotada de celdas para calibrar
  con `mask.py` (sin cambios).
- `gee_extraction_pipeline.extract_all_provinces_all_variables` ya acepta
  `umbrales_mascara: MaskThresholds | None` para enrutar `"ndvi"`.

Falta antes de usar esto en producción:
- [ ] Fijar los umbrales reales de `MaskThresholds` (`altitud_min_m`,
  `altitud_max_m`, `pendiente_max_grados` por criterio agronómico;
  `amplitud_ndvi_min` calibrado con `sample_cells_for_calibration` +
  `calibrate_ndvi_amplitude_threshold` sobre campañas de entrenamiento) —
  no hay valores reales todavía, solo los de ejemplo en los tests.
- [ ] Verificar manualmente contra la API real de GEE (no cubierto por los
  dobles de prueba): que la clase 40 de WorldCover es "Cropland" en la
  sierra peruana, que `ee.Terrain.slope()` devuelve grados, y el volumen/
  tiempo real de `sampleRegions`.
- [ ] Re-ejecutar la extracción real de NDVI (Actividad 3) con
  `umbrales_mascara` ya fijados, para regenerar
  `data/raw/gee_series_crudo.parquet` con NDVI enmascarado.
```

- [ ] **Step 2: Commit**

```bash
git add service-cnn/docs/PENDIENTE.md
git commit -m "$(cat <<'EOF'
Actualizar PENDIENTE.md: mascara agricola GEE implementada, falta calibrar

Registra que la extraccion server-side de la mascara agricola (2.1) ya
esta implementada y probada, y deja explicito lo que falta antes de
producción: fijar umbrales reales, verificacion manual contra la API
real de GEE, y re-ejecutar la extraccion de NDVI con la mascara aplicada.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016mkokyMQFzBE7QpPxKea4U
EOF
)"
```
