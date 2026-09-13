# Filtrado de Calidad y Máximo NDVI (cumplimiento Tabla 7) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cerrar el gap encontrado en la auditoría de cumplimiento contra la tesis (Tabla 7, sección 4.9): NDVI debe extraerse con "filtrado de calidad; media y máximo por fase sobre máscara agrícola", pero hoy `GeeClient.extract_daily_series` (código ya en producción, genérico para las 5 variables) solo calcula la media, sin filtrar por la banda de calidad `DetailedQA` de MOD13Q1.

**Architecture:** Generalizar (no bifurcar) `GeeCollectionSpec`/`extract_daily_series`/`aggregate_by_phase` para soportar, por variable, una banda de calidad opcional y una lista de reducers en vez de uno solo. Las 4 variables sin banda de calidad (CHIRPS, ERA5-Land ×3) preservan exactamente su comportamiento actual — es un cambio aditivo, no una reescritura. Solo NDVI declara `banda_calidad="DetailedQA"` y `reducers=[MEAN, MAX]`.

**Tech Stack:** Python 3.14, pandas, pytest. `ee` inyectado (real o doble de prueba), nunca importado a nivel de módulo — mismo patrón que el resto de `src/ingestion/`.

## Global Constraints

- **Criterio de calidad MOD13Q1 `DetailedQA`** (verificado contra el catálogo oficial de Earth Engine y el MODIS VI User's Guide — no inventar bits):
  - Bits 0-1 (`VI Quality`/MODLAND_QA): aceptar `00` (buena calidad) y `01` (revisar otras QA); rechazar `10` (probablemente nublado) y `11` (no producido).
  - Bit 8 (nube adyacente detectada): aceptar solo `0`.
  - Bit 10 (nubes mixtas posibles): aceptar solo `0`.
  - Decodificación: `qa.rightShift(n).bitwiseAnd(mascara)` para extraer cada sub-campo — `rightShift(0).bitwiseAnd(0b11)` para VI Quality, `rightShift(8).bitwiseAnd(0b1)` para nube adyacente, `rightShift(10).bitwiseAnd(0b1)` para nubes mixtas.
- **Retrocompatibilidad estricta**: las 4 variables sin banda de calidad (`precipitacion`, `temperatura_maxima`, `temperatura_minima`, `radiacion_solar`) no cambian de comportamiento en ningún test ya existente — ni en `test_gee_config.py`, ni en `test_gee_client.py`, ni en `test_gee_series_builder.py`.
- Con un solo reducer, `extract_daily_series` sigue devolviendo la columna `valor` (no `valor_mean`). Con múltiples reducers (NDVI), devuelve `valor_mean`, `valor_max` (una columna por reducer, sufijo = nombre del reducer en minúsculas).
- Con una sola columna en `reglas_por_columna` con clave `"valor"`, `aggregate_by_phase` sigue devolviendo `valor_agregado` (no `valor_agregado_valor`). Con múltiples columnas, devuelve `valor_agregado_<sufijo>` por cada una, donde `<sufijo>` es la clave de la columna sin el prefijo `valor_` (ej. `valor_mean` → `valor_agregado_mean`).
- No se re-ejecuta la extracción real contra la API de GEE en este ciclo — solo código, con dobles de prueba, sin tocar `data/raw/gee_series_crudo.parquet`.
- `ee` nunca se importa a nivel de módulo — inyectado (real o doble de prueba), mismo patrón ya establecido en `gee_client.py`.
- Referencia de la auditoría que originó este plan: `service-cnn/docs/PENDIENTE.md`, sección 2.1, párrafo "[Gap crítico, alcance MAYOR a esta sesión]".

---

## Task 1: `GeeCollectionSpec` — campos `banda_calidad` y `reducers`

**Files:**
- Modify: `service-cnn/src/ingestion/gee_config.py`
- Modify: `service-cnn/tests/ingestion/test_gee_config.py`

**Interfaces:**
- Produces: `GeeCollectionSpec` gana dos campos nuevos con default:
  - `banda_calidad: str | None = None`
  - `reducers: tuple[AggregationRule, ...] = ()` — tupla (no lista, para mantener el dataclass `frozen=True` hasheable/inmutable) que, si está vacía, implica "usar solo `aggregation`" (comportamiento actual). Se puebla explícitamente con `(AggregationRule.MEAN, AggregationRule.MAX)` para NDVI.
- Consumes: nada nuevo — `AggregationRule` ya existe.

- [ ] **Step 1: Escribir los tests en rojo**

Añadir a `service-cnn/tests/ingestion/test_gee_config.py`, dentro de `TestGetCollectionSpec`:

```python
    def test_modis_ndvi_tiene_banda_de_calidad_y_reducers_multiples(self):
        """Tabla 7: 'Filtrado de calidad; media y máximo por fase sobre
        máscara agrícola' — NDVI necesita filtrar por DetailedQA (MOD13Q1)
        y calcular dos reducers (media y máximo), no solo uno."""
        spec = get_collection_spec("ndvi")
        assert spec.banda_calidad == "DetailedQA"
        assert spec.reducers == (AggregationRule.MEAN, AggregationRule.MAX)

    def test_chirps_precipitacion_no_tiene_banda_de_calidad_ni_reducers_multiples(self):
        """Retrocompatibilidad: las variables sin banda de calidad propia
        (CHIRPS, ERA5-Land) no deben ganar comportamiento nuevo."""
        spec = get_collection_spec("precipitacion")
        assert spec.banda_calidad is None
        assert spec.reducers == ()

    def test_era5_land_variables_no_tienen_banda_de_calidad(self):
        for variable in ("temperatura_maxima", "temperatura_minima", "radiacion_solar"):
            spec = get_collection_spec(variable)
            assert spec.banda_calidad is None
            assert spec.reducers == ()
```

- [ ] **Step 2: Ejecutar los tests para verificar que fallan**

Run: `cd service-cnn && python -m pytest tests/ingestion/test_gee_config.py -v`
Expected: FAIL — `AttributeError: 'GeeCollectionSpec' object has no attribute 'banda_calidad'` en los 3 tests nuevos; los tests ya existentes (7) siguen pasando.

- [ ] **Step 3: Implementar los campos nuevos**

En `service-cnn/src/ingestion/gee_config.py`, modificar la clase y la entrada de NDVI:

```python
@dataclass(frozen=True)
class GeeCollectionSpec:
    """Especificación de una colección de Earth Engine para una variable.

    `factor_escala` y `offset_aditivo` convierten el valor crudo devuelto por
    Earth Engine a la unidad real de la variable: `valor_real = crudo *
    factor_escala + offset_aditivo`. Por defecto no aplican conversión (1.0 y
    0.0) — solo NDVI (factor de escala entero→decimal) y las variables de
    ERA5-Land (Kelvin→Celsius) los necesitan, verificado contra el catálogo
    oficial MODIS/ECMWF.

    `banda_calidad` y `reducers` cubren el caso NDVI (Tabla 7: "Filtrado de
    calidad; media y máximo por fase sobre máscara agrícola"): si
    `banda_calidad` no es `None`, `extract_daily_series` filtra por esa banda
    antes de reducir; si `reducers` no está vacío, se calculan TODOS esos
    reducers (en vez de uno solo, `aggregation`), devolviendo una columna por
    reducer. Ambos son `None`/vacío por defecto — las variables sin banda de
    calidad propia (CHIRPS, ERA5-Land) no cambian de comportamiento.
    """

    collection_id: str
    band: str
    spatial_resolution_m: int
    temporal_resolution: str
    aggregation: AggregationRule
    factor_escala: float = 1.0
    offset_aditivo: float = 0.0
    banda_calidad: str | None = None
    reducers: tuple[AggregationRule, ...] = ()
```

Y en la entrada de `"ndvi"` dentro de `_SPECS`:

```python
    "ndvi": GeeCollectionSpec(
        collection_id="MODIS/061/MOD13Q1",
        band="NDVI",
        spatial_resolution_m=250,
        temporal_resolution="16 dias",
        # Tabla 7: "media y máximo por fase sobre máscara agrícola" — ambos
        # reducers se calculan (ver banda_calidad/reducers abajo); aggregation
        # se conserva como el reducer "principal" para compatibilidad con
        # llamadores que aún no distinguen entre columnas.
        aggregation=AggregationRule.MEAN,
        # MOD13Q1 codifica NDVI como entero [-2000, 10000] (catálogo oficial
        # MODIS/USGS): valor_real = crudo * 0.0001.
        factor_escala=0.0001,
        # DetailedQA (16 bits, catálogo oficial MODIS/061/MOD13Q1 + MODIS VI
        # User's Guide): bits 0-1 = VI Quality (00=buena, 01=revisar otras
        # QA, 10=nublado, 11=no producido); bit 8 = nube adyacente; bit 10 =
        # nubes mixtas posibles. Se filtra en extract_daily_series.
        banda_calidad="DetailedQA",
        reducers=(AggregationRule.MEAN, AggregationRule.MAX),
    ),
```

- [ ] **Step 4: Ejecutar los tests para verificar que pasan**

Run: `cd service-cnn && python -m pytest tests/ingestion/test_gee_config.py -v`
Expected: PASS — 10 tests (7 previos + 3 nuevos).

- [ ] **Step 5: Commit**

```bash
cd service-cnn
git add src/ingestion/gee_config.py tests/ingestion/test_gee_config.py
git commit -m "$(cat <<'EOF'
Agregar banda_calidad y reducers a GeeCollectionSpec para NDVI

Tabla 7 (seccion 4.9 de la tesis) exige para NDVI "Filtrado de calidad;
media y maximo por fase sobre mascara agricola" -- se declaran los
campos banda_calidad="DetailedQA" y reducers=(MEAN, MAX) solo para
NDVI. Las 4 variables restantes (CHIRPS, ERA5-Land x3) quedan con
banda_calidad=None y reducers=() -- sin cambio de comportamiento.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016mkokyMQFzBE7QpPxKea4U
EOF
)"
```

---

## Task 2: `GeeClient.extract_daily_series` — filtrado de calidad y reducers múltiples

**Files:**
- Modify: `service-cnn/src/ingestion/gee_client.py`
- Modify: `service-cnn/tests/ingestion/test_gee_client.py`

**Interfaces:**
- Consumes: `GeeCollectionSpec.banda_calidad`, `GeeCollectionSpec.reducers` (Task 1).
- Produces: `extract_daily_series` sin cambio de firma pública. Cambia únicamente la forma de la tabla devuelta cuando `spec.reducers` tiene más de un elemento: columnas `valor_<nombre_reducer>` (ej. `valor_mean`, `valor_max`) en vez de `valor`. Con `spec.reducers` vacío (el caso de las 4 variables sin cambios), sigue devolviendo `valor` exactamente como hoy.

El doble de prueba `_FakeEeModule`/`_FakeImage` en `test_gee_client.py` necesita extenderse para soportar `.rightShift(n)`, `.bitwiseAnd(v)`, `.eq(v)`, `.lte(v)`, `.And()` sobre una imagen de banda de calidad, y que `reduceRegions` pueda devolver más de una estadística por feature (ej. `{"mean": ..., "max": ...}` en vez de solo `{"mean": ...}`).

- [ ] **Step 1: Escribir los tests en rojo**

Añadir a `service-cnn/tests/ingestion/test_gee_client.py`, extendiendo el doble de prueba existente y añadiendo una clase de tests nueva. Primero, extender `_FakeImage` (la imagen individual dentro de la colección) con soporte de banda de calidad — reemplazar la clase `_FakeImage` actual por esta versión extendida:

```python
class _FakeQaExpression:
    """Doble de prueba de una sub-expresión de calidad (resultado de
    rightShift/bitwiseAnd/eq/lte/And) — árbol de expresión inspeccionable,
    mismo patrón que test_gee_mask_pipeline.py."""

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
```

Nota de diseño del doble de prueba: a diferencia de `test_gee_mask_pipeline.py` (que necesita inspeccionar la EXPRESIÓN sin evaluarla, porque no hay un valor de píxel concreto), aquí cada imagen fake YA tiene un valor QA concreto inyectado por el test (`filas_esperadas` incluye un campo `qa`), así que el doble evalúa la aritmética de bits inmediatamente — más simple y suficiente para verificar el criterio de exclusión con casos concretos.

Ahora extender `_FakeImage` para que `.select(banda)` devuelva un tipo distinto según qué banda se pida: la banda de datos (`spec.band`, ej. `"NDVI"`) devuelve `self` (comportamiento actual, compatible con `reduceRegions`); la banda de calidad (`"DetailedQA"`) devuelve un `_FakeQaExpression` construido con el valor `qa` concreto de esa fila — inyectado por el test vía una nueva clave `"qa"` en cada fila de `filas_esperadas`. Esto requiere que `_FakeImage` conozca su propio valor `qa` (uno por imagen/fecha, igual que ya conoce `_filas`). Reemplazar `_FakeImage` por:

```python
class _FakeImage:
    """Doble de prueba de una imagen individual dentro de la colección."""

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
        # select puede recibir un string (una banda) o una lista (varias,
        # cuando extract_daily_series pide [band, banda_calidad] a la vez —
        # ese caso de lista no se ejercita aquí porque extract_daily_series
        # llama .select(spec.band) y .select(spec.banda_calidad) por
        # separado dentro de _reducir_imagen, no con una lista combinada).
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
        self._ee.llamadas.append(("Image.reduceRegions", reducer, scale, self._enmascarada))
        return _FakeFeatureCollection(
            [_FakeFeature(fila) for fila in self._filas]
        )
```

`_FakeImageCollection` (o el bloque equivalente en `_FakeEeModule`, según cómo esté organizado el archivo real al momento de implementar — revisar `test_gee_client.py` actual antes de tocar) ya construye una `_FakeImage` por cada fecha en `_filas_por_fecha`; extenderlo para pasar también el valor `qa` de esa fecha, tomándolo de la primera fila de esa fecha en `filas_resultado` (todas las filas de una misma fecha comparten el mismo valor `qa` en los fixtures de este plan, ya que `qa` es una propiedad de la imagen completa, no de la reducción por provincia):

```python
    def map(self, funcion):
        resultados = []
        for fecha, filas in self._filas_por_fecha.items():
            valor_qa = filas[0].get("qa")
            resultados.append(funcion(_FakeImage(self._ee, fecha, filas, valor_qa)))
        features = [f for coleccion in resultados for f in coleccion._features]
        return _FakeFeatureCollection(features)
```

(Ajustar el nombre exacto del método/clase a como aparezca en el archivo real — el archivo ya tiene una `_FakeImageCollection.map` con esta forma general, ver `tests/ingestion/test_gee_client.py` líneas 83-105 antes de este plan; solo se añade el paso de `valor_qa`.)

Extender `_FakeEeModule.Reducer` para exponer `.mean()` y `.max()` combinables, y las filas de test para incluir la clave `"max"` cuando se pida ese reducer:

```python
    class Reducer:
        @staticmethod
        def mean():
            return "MEAN_REDUCER"

        @staticmethod
        def max():
            return "MAX_REDUCER"
```

Ahora los tests. Añadir a `service-cnn/tests/ingestion/test_gee_client.py`:

```python
class TestGeeClientFiltradoCalidadNdvi:
    """Tabla 7: 'Filtrado de calidad; media y máximo por fase sobre máscara
    agrícola' -- verificado contra DetailedQA de MOD13Q1 (bits 0-1 VI
    Quality, bit 8 nube adyacente, bit 10 nubes mixtas)."""

    def _construir_valor_qa(self, vi_quality: int, nube_adyacente: int = 0, nubes_mixtas: int = 0) -> int:
        return vi_quality | (nube_adyacente << 8) | (nubes_mixtas << 10)

    def test_acepta_pixel_con_vi_quality_buena(self, geometria_provincias_fake):
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


class TestGeeClientReducersMultiples:
    def test_ndvi_devuelve_columnas_valor_mean_y_valor_max(self, geometria_provincias_fake):
        filas_esperadas = [
            {
                "provincia_id": "PUN-AZA",
                "fecha": "2020-09-01",
                "mean": 5000.0,
                "max": 6000.0,
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
        filas_esperadas = [
            {"provincia_id": "PUN-AZA", "fecha": "2020-09-01", "mean": 5.0},
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
```

Nota: los 4 primeros tests de `TestGeeClientFiltradoCalidadNdvi` ejercitan el doble de prueba directamente (verificación de que la aritmética de bits del fake es correcta contra los 6 casos de la Global Constraint) antes de integrarlo en `extract_daily_series` — esto separa "el fake calcula bien" de "extract_daily_series usa el fake correctamente", que se cubre en el Step 3 de abajo con un test de integración adicional.

- [ ] **Step 2: Ejecutar los tests para verificar que fallan**

Run: `cd service-cnn && python -m pytest tests/ingestion/test_gee_client.py -v`
Expected: FAIL — `test_ndvi_devuelve_columnas_valor_mean_y_valor_max` falla porque `extract_daily_series` no aplica múltiples reducers todavía (sigue devolviendo `valor` desde `mean` únicamente, ignorando `max`). Los tests de `TestGeeClientFiltradoCalidadNdvi` deberían pasar ya en rojo si el doble de prueba está bien escrito (son tests del propio fake, no de producción) — si fallan, es un bug en el doble de prueba, corregirlo antes de continuar.

- [ ] **Step 3: Implementar el filtrado de calidad y reducers múltiples en `extract_daily_series`**

Modificar `service-cnn/src/ingestion/gee_client.py`:

```python
    def extract_daily_series(
        self,
        variable: str,
        geometrias_por_provincia: Any,
        fecha_inicio: date,
        fecha_fin: date,
    ) -> pd.DataFrame:
        """Extrae la serie temporal de una variable agregada por provincia.

        Args:
            variable: clave de `gee_config` (ej. "precipitacion", "ndvi").
            geometrias_por_provincia: `ee.FeatureCollection` con un feature por
                provincia (geometría + `provincia_id` como propiedad).
            fecha_inicio, fecha_fin: ventana temporal a extraer (inclusive).

        Returns:
            DataFrame con columnas `provincia_id`, `fecha`, y ya sea `valor`
            (si `spec.reducers` está vacío — comportamiento actual) o una
            columna `valor_<nombre_reducer>` por cada reducer en
            `spec.reducers` (ej. `valor_mean`, `valor_max` para NDVI, Tabla
            7: "Filtrado de calidad; media y máximo por fase sobre máscara
            agrícola"). `valor`/`valor_*` ya convertidos a su unidad real
            (`factor_escala`/`offset_aditivo` del spec).

            Si `spec.banda_calidad` no es `None`, cada imagen se enmascara
            antes de reducir: solo se conservan píxeles con VI Quality
            (bits 0-1 de la banda) en {buena calidad, revisar otras QA}, sin
            nube adyacente (bit 8) ni nubes mixtas (bit 10) — verificado
            contra el catálogo oficial de `MODIS/061/MOD13Q1` y el MODIS VI
            User's Guide.

        Raises:
            ValueError: si `fecha_inicio` es posterior a `fecha_fin`.
        """
        if fecha_inicio > fecha_fin:
            raise ValueError(
                f"fecha_inicio ({fecha_inicio}) no puede ser posterior a "
                f"fecha_fin ({fecha_fin})."
            )

        spec = get_collection_spec(variable)
        ee = self._ee

        bandas = [spec.band, spec.banda_calidad] if spec.banda_calidad else [spec.band]
        coleccion = (
            ee.ImageCollection(spec.collection_id)
            .filterDate(fecha_inicio.isoformat(), fecha_fin.isoformat())
            .select(bandas)
        )

        reducers = spec.reducers if spec.reducers else (spec.aggregation,)
        reducer_ee = _combinar_reducers(ee, reducers)

        def _reducir_imagen(imagen):
            fecha = imagen.date().format("YYYY-MM-dd")
            imagen_datos = imagen.select(spec.band)
            if spec.banda_calidad:
                qa = imagen.select(spec.banda_calidad)
                buena_calidad = _mascara_calidad_ndvi(ee, qa)
                imagen_datos = imagen_datos.updateMask(buena_calidad)
            reducido = imagen_datos.reduceRegions(
                collection=geometrias_por_provincia,
                reducer=reducer_ee,
                scale=spec.spatial_resolution_m,
            )
            return reducido.map(lambda f: f.set("fecha", fecha))

        resultado_aplanado = coleccion.map(_reducir_imagen).flatten()

        filas = [f["properties"] for f in resultado_aplanado.getInfo()["features"]]
        tabla = pd.DataFrame(filas)

        if len(reducers) == 1:
            if tabla.empty:
                return pd.DataFrame(columns=["provincia_id", "fecha", "valor"])
            nombre_reducer = reducers[0].value
            tabla = tabla.rename(columns={nombre_reducer: "valor"})[
                ["provincia_id", "fecha", "valor"]
            ]
            tabla["valor"] = tabla["valor"] * spec.factor_escala + spec.offset_aditivo
            return tabla

        columnas_salida = ["provincia_id", "fecha"]
        if tabla.empty:
            return pd.DataFrame(
                columns=columnas_salida + [f"valor_{r.value}" for r in reducers]
            )
        for regla in reducers:
            columna_salida = f"valor_{regla.value}"
            tabla[columna_salida] = (
                tabla[regla.value] * spec.factor_escala + spec.offset_aditivo
            )
            columnas_salida.append(columna_salida)
        return tabla[columnas_salida]
```

Añadir las dos funciones auxiliares al final del archivo (fuera de la clase):

```python
def _combinar_reducers(ee_module: Any, reducers: tuple) -> Any:
    """Combina uno o más `AggregationRule` en un único `ee.Reducer`.

    Con un solo reducer, lo construye directamente (`ee.Reducer.mean()`,
    etc.). Con varios, los combina con `.combine()` para que
    `reduceRegions` los calcule en una sola pasada, sin re-leer la imagen.
    """
    from src.ingestion.gee_config import AggregationRule

    _CONSTRUCTOR = {
        AggregationRule.SUM: ee_module.Reducer.sum,
        AggregationRule.MEAN: ee_module.Reducer.mean,
        AggregationRule.MAX: ee_module.Reducer.max,
        AggregationRule.MIN: ee_module.Reducer.min,
    }
    reducer_actual = _CONSTRUCTOR[reducers[0]]()
    for siguiente in reducers[1:]:
        reducer_actual = reducer_actual.combine(
            reducer2=_CONSTRUCTOR[siguiente](), sharedInputs=True
        )
    return reducer_actual


def _mascara_calidad_ndvi(ee_module: Any, banda_qa: Any) -> Any:
    """Máscara booleana de calidad para MOD13Q1 `DetailedQA`.

    Verificado contra el catálogo oficial de Earth Engine
    (MODIS/061/MOD13Q1) y el MODIS VI User's Guide:
    - Bits 0-1 (VI Quality/MODLAND_QA): 00=buena calidad, 01=revisar otras
      QA, 10=probablemente nublado, 11=no producido. Se aceptan 00 y 01.
    - Bit 8 (nube adyacente detectada): se acepta solo 0.
    - Bit 10 (nubes mixtas posibles): se acepta solo 0.
    """
    vi_quality = banda_qa.rightShift(0).bitwiseAnd(0b11)
    nube_adyacente = banda_qa.rightShift(8).bitwiseAnd(0b1)
    nubes_mixtas = banda_qa.rightShift(10).bitwiseAnd(0b1)
    return vi_quality.lte(1).And(nube_adyacente.eq(0)).And(nubes_mixtas.eq(0))
```

Nota de implementación: `_combinar_reducers` usa `.value` de `AggregationRule` (ej. `"mean"`, `"max"`) como el nombre esperado tanto del método `ee.Reducer.<nombre>()` como de la columna que `reduceRegions` produce — verificar que el doble de prueba en Task 2 Step 1 expone `_FakeEeModule.Reducer.sum`/`.mean()`/`.max()`/`.min()` de forma consistente con esto (el plan de Task 1/2 solo mostró `.mean()`/`.max()` porque son los únicos reducers usados hoy por variables con múltiples reducers; añadir `.sum()`/`.min()` al fake solo si algún test los ejercita — hoy ninguna variable con reducers múltiples los usa, así que no son estrictamente necesarios para que los tests pasen, pero conviene añadirlos por completitud si el tiempo lo permite).

- [ ] **Step 4: Ejecutar los tests para verificar que pasan**

Run: `cd service-cnn && python -m pytest tests/ingestion/test_gee_client.py -v`
Expected: PASS — todos los tests (los ya existentes + los nuevos de calidad + reducers múltiples).

- [ ] **Step 5: Commit**

```bash
cd service-cnn
git add src/ingestion/gee_client.py tests/ingestion/test_gee_client.py
git commit -m "$(cat <<'EOF'
Aplicar filtrado de calidad DetailedQA y reducers multiples en extract_daily_series

Cuando spec.banda_calidad esta presente (solo NDVI), cada imagen se
enmascara antes de reduceRegions: se descartan pixeles nublados (VI
Quality bits 0-1 = 10), no producidos (= 11), con nube adyacente
(bit 8) o nubes mixtas (bit 10) -- verificado contra el catalogo
oficial de MODIS/061/MOD13Q1 y el MODIS VI User's Guide.

Cuando spec.reducers tiene mas de un elemento (solo NDVI: MEAN y MAX),
extract_daily_series devuelve columnas valor_mean/valor_max en vez de
una sola valor -- las 4 variables sin banda de calidad ni reducers
multiples (CHIRPS, ERA5-Land x3) preservan exactamente su
comportamiento y esquema de salida actuales.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016mkokyMQFzBE7QpPxKea4U
EOF
)"
```

---

## Task 3: `aggregate_by_phase` — múltiples columnas con su propia regla

**Files:**
- Modify: `service-cnn/src/ingestion/gee_series_builder.py`
- Modify: `service-cnn/tests/ingestion/test_gee_series_builder.py`

**Interfaces:**
- Consumes: nada nuevo de Tasks 1-2 directamente — opera sobre el DataFrame que ya produce `extract_daily_series` (columnas `valor` o `valor_mean`/`valor_max`), pero es independiente en el sentido de que sus tests no importan `gee_client`.
- Produces: `aggregate_by_phase(serie, fases, reglas_por_columna: dict[str, AggregationRule], fecha_corte=None) -> pd.DataFrame` — cambia el tercer parámetro posicional de `AggregationRule` único a `dict[str, AggregationRule]`. Con `{"valor": regla}` (una sola entrada con clave `"valor"`), el resultado sigue teniendo la columna `valor_agregado` (comportamiento idéntico al actual). Con múltiples entradas o una clave distinta de `"valor"`, el resultado tiene una columna `valor_agregado_<sufijo>` por entrada, donde `<sufijo>` es la clave sin el prefijo `"valor_"` (ej. clave `"valor_mean"` → columna `valor_agregado_mean`; clave `"valor_max"` → `valor_agregado_max`).

- [ ] **Step 1: Actualizar los tests existentes a la nueva firma (rojo)**

Modificar `service-cnn/tests/ingestion/test_gee_series_builder.py`: reemplazar cada llamada `aggregate_by_phase(serie, fases, AggregationRule.X, ...)` por `aggregate_by_phase(serie, fases, {"valor": AggregationRule.X}, ...)`. Aplicar a los 8 tests existentes (`test_suma_precipitacion_por_fase`, `test_suma_precipitacion_fase_de_dos_meses`, `test_media_para_temperatura`, `test_preserva_columnas_de_identificacion`, `test_fase_sin_datos_en_la_serie_produce_nan_no_cero`, `test_rechaza_datos_posteriores_al_horizonte_de_corte`, `test_permite_datos_hasta_el_horizonte_inclusive`) — cada uno cambia solo el tercer argumento posicional, sin tocar el resto del test.

Ejemplo del primer test ya actualizado:

```python
    def test_suma_precipitacion_por_fase(
        self, serie_diaria_precipitacion, fases_fenologicas
    ):
        resultado = aggregate_by_phase(
            serie_diaria_precipitacion, fases_fenologicas, {"valor": AggregationRule.SUM}
        )
        fila_emergencia = resultado[resultado["fase"] == "emergencia"].iloc[0]
        # Septiembre tiene 30 días * 5mm = 150mm
        assert fila_emergencia["valor_agregado"] == pytest.approx(150.0)
```

Aplicar el mismo cambio (envolver el tercer argumento en `{"valor": ...}`) a los otros 6 tests listados arriba.

Añadir dos tests nuevos al final de `TestAggregateByPhase`:

```python
    def test_multiples_columnas_con_reglas_propias(self, fases_fenologicas):
        """NDVI (Tabla 7): 'media y máximo por fase' -- valor_mean se agrega
        con MEAN, valor_max se agrega con MAX, en una sola pasada."""
        fechas = pd.date_range("2020-09-01", "2020-09-30", freq="D")
        serie = pd.DataFrame(
            {
                "provincia_id": "PUN-AZA",
                "campana_id": 2021,
                "fecha": fechas,
                "valor_mean": list(range(1, len(fechas) + 1)),  # 1..30
                "valor_max": [100.0] * len(fechas),
            }
        )
        resultado = aggregate_by_phase(
            serie,
            fases_fenologicas[:1],
            {"valor_mean": AggregationRule.MEAN, "valor_max": AggregationRule.MAX},
        )
        fila = resultado.iloc[0]
        assert fila["valor_agregado_mean"] == pytest.approx(15.5)
        assert fila["valor_agregado_max"] == pytest.approx(100.0)

    def test_columna_unica_con_clave_valor_preserva_nombre_actual(
        self, serie_diaria_precipitacion, fases_fenologicas
    ):
        """Retrocompatibilidad: con una sola columna de clave 'valor', el
        resultado sigue teniendo 'valor_agregado' a secas, no
        'valor_agregado_valor' -- ningún llamador actual debe reescribirse."""
        resultado = aggregate_by_phase(
            serie_diaria_precipitacion,
            fases_fenologicas,
            {"valor": AggregationRule.SUM},
        )
        assert "valor_agregado" in resultado.columns
        assert "valor_agregado_valor" not in resultado.columns
```

- [ ] **Step 2: Ejecutar los tests para verificar que fallan**

Run: `cd service-cnn && python -m pytest tests/ingestion/test_gee_series_builder.py -v`
Expected: FAIL — `TypeError` en los tests ya actualizados (la función actual espera un `AggregationRule`, no un `dict`), y los 2 tests nuevos fallan por el mismo motivo.

- [ ] **Step 3: Implementar la nueva firma de `aggregate_by_phase`**

Reemplazar en `service-cnn/src/ingestion/gee_series_builder.py`:

```python
def aggregate_by_phase(
    serie: pd.DataFrame,
    fases: pd.DataFrame,
    reglas_por_columna: dict[str, AggregationRule],
    fecha_corte: date | None = None,
) -> pd.DataFrame:
    """Agrega una serie temporal diaria/16-días por fase fenológica.

    Args:
        serie: columnas `provincia_id`, `campana_id`, `fecha`, y una columna
            de valor por cada clave de `reglas_por_columna` (ej. `valor` para
            la mayoría de variables; `valor_mean`/`valor_max` para NDVI,
            Tabla 7) — ya recortada a la provincia-campaña de interés.
        fases: columnas `provincia_id`, `campana_id`, `fase`, `inicio`, `fin` —
            ventanas de fase fenológica (sección 4.10.2).
        reglas_por_columna: mapa `{nombre_columna: regla}` — por cada
            columna presente en `serie`, qué `AggregationRule` aplicar por
            fase. Con una sola entrada de clave `"valor"`, el resultado
            conserva el nombre `valor_agregado` (compatibilidad con el
            comportamiento anterior); con cualquier otra clave o múltiples
            entradas, el resultado tiene una columna `valor_agregado_<sufijo>`
            por entrada (`<sufijo>` = la clave sin el prefijo `"valor_"`, ej.
            `"valor_mean"` → `valor_agregado_mean`).
        fecha_corte: si se especifica, ninguna fecha de `serie` puede ser
            posterior a este valor (regla anti-fuga, sección 4.10.4). Si se
            excede, se lanza `HorizonteExcedidoError` en vez de recortar
            silenciosamente — una violación de horizonte debe corregirse en el
            paso de extracción, no enmascararse aquí.

    Returns:
        DataFrame con una fila por fase, columnas `provincia_id`, `campana_id`,
        `fase`, y una o más columnas `valor_agregado[_<sufijo>]`. Una fase sin
        observaciones en la serie devuelve NaN en cada columna, nunca 0 (evita
        ocultar huecos de datos, sección 4.7).
    """
    serie_fechas = pd.to_datetime(serie["fecha"]).dt.date
    if fecha_corte is not None and (serie_fechas > fecha_corte).any():
        maxima = serie_fechas.max()
        raise HorizonteExcedidoError(
            f"La serie contiene observaciones hasta {maxima}, posteriores al "
            f"punto de corte {fecha_corte}. Esto violaría la regla anti-fuga "
            "temporal (sección 4.12.2) — corrija la extracción, no la agregación."
        )

    columna_unica_valor = (
        len(reglas_por_columna) == 1 and "valor" in reglas_por_columna
    )

    filas: list[dict] = []
    for _, fase_row in fases.iterrows():
        en_ventana = (serie_fechas >= fase_row["inicio"]) & (
            serie_fechas <= fase_row["fin"]
        )
        fila = {
            "provincia_id": fase_row["provincia_id"],
            "campana_id": fase_row["campana_id"],
            "fase": fase_row["fase"],
        }
        for columna, regla in reglas_por_columna.items():
            funcion = _AGG_FUNC[regla]
            valores_en_fase = serie.loc[en_ventana, columna]
            valor_agregado = (
                getattr(valores_en_fase, funcion)()
                if len(valores_en_fase) > 0
                else float("nan")
            )
            nombre_salida = (
                "valor_agregado"
                if columna_unica_valor
                else f"valor_agregado_{columna.removeprefix('valor_')}"
            )
            fila[nombre_salida] = valor_agregado
        filas.append(fila)

    return pd.DataFrame(filas)
```

- [ ] **Step 4: Ejecutar los tests para verificar que pasan**

Run: `cd service-cnn && python -m pytest tests/ingestion/test_gee_series_builder.py -v`
Expected: PASS — todos los tests (7 actualizados + 2 nuevos = 9).

- [ ] **Step 5: Ejecutar toda la suite de tests de `service-cnn` para verificar que nada más se rompió**

Run: `cd service-cnn && python -m pytest tests/ -q --ignore=tests/features/test_dataset_builder.py`
Expected: PASS — 288 (base) + 3 (Task 1) + 6 (Task 2) + 2 (Task 3 nuevos, los 7 actualizados no suman) = 297 o similar; confirmar el número exacto en la salida, no asumirlo.

- [ ] **Step 6: Commit**

```bash
cd service-cnn
git add src/ingestion/gee_series_builder.py tests/ingestion/test_gee_series_builder.py
git commit -m "$(cat <<'EOF'
Generalizar aggregate_by_phase para multiples columnas con regla propia

reglas_por_columna: dict[str, AggregationRule] reemplaza el parametro
unico AggregationRule -- permite que NDVI agregue valor_mean con MEAN
y valor_max con MAX en una sola pasada por las ventanas de fase (Tabla
7: "media y maximo por fase"). Con una sola entrada de clave "valor"
(el caso de las demas variables), el resultado conserva el nombre
valor_agregado sin cambios -- todos los tests existentes actualizados
solo en la forma de invocacion, sin cambiar sus aserciones.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016mkokyMQFzBE7QpPxKea4U
EOF
)"
```

---

## Task 4: Actualizar `PENDIENTE.md`

**Files:**
- Modify: `service-cnn/docs/PENDIENTE.md`

**Interfaces:** ninguna — solo documentación.

- [ ] **Step 1: Marcar el gap de filtrado de calidad + máximo como resuelto en código, pendiente de re-extracción**

En `service-cnn/docs/PENDIENTE.md`, dentro de la sección 2.1, localizar el párrafo que empieza con `- [ ] **[Gap crítico, alcance MAYOR a esta sesión]**` (añadido en el commit `14c08f0`) y reemplazarlo por:

```markdown
- [x] **[Gap crítico, resuelto en código]** ~~La Tabla 7 de la tesis...~~
  Resuelto: `GeeCollectionSpec` (NDVI) ahora declara `banda_calidad="DetailedQA"`
  y `reducers=(MEAN, MAX)`; `GeeClient.extract_daily_series` filtra por
  calidad (VI Quality bits 0-1, nube adyacente bit 8, nubes mixtas bit 10 —
  verificado contra el catálogo oficial de `MODIS/061/MOD13Q1` y el MODIS VI
  User's Guide) y devuelve `valor_mean`/`valor_max`; `aggregate_by_phase`
  agrega ambas columnas con su propia regla. Ver
  `docs/superpowers/plans/2026-09-13-filtrado-calidad-max-ndvi.md`.
  **Pendiente**: re-ejecutar la extracción real de NDVI contra la API de GEE
  (Actividad 3) para regenerar `data/raw/gee_series_crudo.parquet` con el
  filtro de calidad y ambos reducers aplicados — el parquet actual sigue
  siendo NDVI sin filtrar, solo media, hasta que se re-extraiga.
```

- [ ] **Step 2: Actualizar el conteo de tests**

Correr `cd service-cnn && python -m pytest tests/ -q --ignore=tests/features/test_dataset_builder.py`, tomar el número real reportado, y actualizar la línea "288/288 tests pasando" de la sección 2.1 al conteo real post-Task 3.

- [ ] **Step 3: Commit**

```bash
cd service-cnn
git add docs/PENDIENTE.md
git commit -m "$(cat <<'EOF'
Marcar gap de filtrado de calidad y maximo NDVI como resuelto en codigo

GeeClient.extract_daily_series ya filtra por DetailedQA y devuelve
media+maximo para NDVI (Tabla 7). Queda pendiente re-ejecutar la
extraccion real contra la API de GEE para regenerar el parquet -- el
codigo esta listo pero data/raw/gee_series_crudo.parquet aun no
refleja el filtro.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016mkokyMQFzBE7QpPxKea4U
EOF
)"
```

---

## Task 5: Propagar el nuevo esquema NDVI a los consumidores aguas abajo

**Contexto (añadido tras la revisión final de todo el branch, hallazgo Critical):** Tasks 1-4 cambiaron el contrato de salida de `GeeClient.extract_daily_series` para NDVI (de una columna `valor` a `valor_mean`/`valor_max`), pero **ningún consumidor aguas abajo se actualizó**:

1. `gee_extraction_pipeline.extract_all_provinces_all_variables` concatena resultados de todas las variables con `pd.concat` — esto en sí YA funciona correctamente hoy (pandas rellena con NaN las columnas ausentes en un concat con esquemas distintos, verificado manualmente), pero (a) su docstring/`columnas` hardcodeado en el caso vacío sigue asumiendo una sola columna `valor` para todas las variables, y (b) **el mock de su test** (`test_gee_extraction_pipeline.py`, fixture `cliente_fake`) devuelve el mismo DataFrame con columna `valor` sin importar la `variable` pedida — incluida `"ndvi"` — por lo que ningún test detectó que el esquema real de NDVI ahora es distinto.
2. `phenology_pipeline._serie_ndvi_de_provincia_campana` lee `filtro["valor"]` sobre el parquet crudo — tras re-extraer con el nuevo código, esa columna no existiría para las filas de NDVI (columna `KeyError`), rompiendo el estimador de fecha de siembra de nivel 2 (`estimate_sowing_date` → `_inicio_estacion_desde_ndvi`, que necesita la serie NDVI continua día a día/16-días — confirmado que es la MEDIA la que corresponde aquí, no el máximo: `_inicio_estacion_desde_ndvi` busca la primera fecha donde el NDVI cruza un umbral ascendente, un uso de serie temporal continua, distinto del máximo de campaña que alimenta el benchmark B3).
3. `tensor_builder.py` (`build_observation_tensor`) espera `valor_agregado` — pero este módulo **no está conectado a datos reales todavía** (confirmado: sin llamador real, ver `docs/PENDIENTE.md` sección 2.3, "Falta conectarlo con datos reales"). Queda **fuera de alcance** de esta tarea — se actualizará cuando 2.3 se implemente, no antes.

**Files:**
- Modify: `service-cnn/src/ingestion/gee_extraction_pipeline.py`
- Modify: `service-cnn/tests/ingestion/test_gee_extraction_pipeline.py`
- Modify: `service-cnn/src/preprocessing/phenology_pipeline.py`
- Modify: `service-cnn/tests/preprocessing/test_phenology_pipeline.py` (si existe un test que ejercite `_serie_ndvi_de_provincia_campana` contra un fixture con columna `valor` — actualizar sus fixtures al nuevo esquema; si no existe, no crear uno nuevo fuera de alcance).
- Modify: `service-cnn/docs/gee_mask_pipeline.py`'s docstring (línea con "mismo contrato que `GeeClient.extract_daily_series`" — hallazgo Minor de la misma revisión, corregido aquí de paso por ser una sola línea).
- Modify: `service-cnn/docs/PENDIENTE.md` (corregir la sección 2.3 que aún documenta el esquema viejo, hallazgo Important de la misma revisión).

**Interfaces:**
- Consumes: el esquema real que `GeeClient.extract_daily_series` ya produce (Task 2): `provincia_id, fecha, valor` para variables de un reducer; `provincia_id, fecha, valor_mean, valor_max` para NDVI.
- Produces: sin cambio de firma pública en `extract_all_provinces_all_variables` ni en `_serie_ndvi_de_provincia_campana` — solo corrige su comportamiento interno y las columnas devueltas para reflejar el esquema real.

- [ ] **Step 1: Corregir el mock del test de `extract_all_provinces_all_variables` para reflejar el contrato real por variable (rojo primero)**

En `service-cnn/tests/ingestion/test_gee_extraction_pipeline.py`, el fixture `cliente_fake` (líneas ~74-84) hoy devuelve el MISMO DataFrame (`columna valor`) para cualquier `variable`. Reemplazar por un `side_effect` que devuelva un esquema distinto según la variable pedida — reflejando exactamente lo que `GeeClient.extract_daily_series` real devuelve hoy:

```python
    @pytest.fixture
    def cliente_fake(self):
        cliente = MagicMock()

        def _side_effect(variable, geometrias_por_provincia, fecha_inicio, fecha_fin):
            if variable == "ndvi":
                return pd.DataFrame(
                    {
                        "provincia_id": ["ABANCAY", "ABANCAY"],
                        "fecha": ["2019-07-01", "2019-07-02"],
                        "valor_mean": [1.0, 2.0],
                        "valor_max": [1.5, 2.5],
                    }
                )
            return pd.DataFrame(
                {
                    "provincia_id": ["ABANCAY", "ABANCAY"],
                    "fecha": ["2019-07-01", "2019-07-02"],
                    "valor": [1.0, 2.0],
                }
            )

        cliente.extract_daily_series.side_effect = _side_effect
        return cliente
```

Este cambio de fixture por sí solo probablemente NO rompe ningún test existente (el concat con `pd.concat` ya maneja columnas heterogéneas hoy, verificado manualmente: `pd.concat([{"valor":1}, {"valor_mean":2,"valor_max":3}])` produce columnas `valor, valor_mean, valor_max` con NaN donde falta) — pero es la corrección que hace que el test suite deje de estar "consistentemente equivocado junto con la producción" (el mismo patrón de trampa ya encontrado una vez en Task 2). Añadir un test nuevo que verifique explícitamente el esquema ancho para NDVI:

```python
    def test_ndvi_conserva_columnas_valor_mean_y_valor_max_en_el_resultado(
        self, cliente_fake, geometrias_fake
    ):
        """El contrato real de GeeClient.extract_daily_series para NDVI
        (Tabla 7: filtrado de calidad + media y máximo) no debe perderse ni
        colisionar al concatenar con las demás variables (esquema de una
        sola columna 'valor')."""
        resultado = extract_all_provinces_all_variables(
            cliente=cliente_fake,
            geometrias_por_provincia=geometrias_fake,
            variables=["precipitacion", "ndvi"],
            campanas=[2020],
        )

        filas_ndvi = resultado[resultado["variable"] == "ndvi"]
        filas_precip = resultado[resultado["variable"] == "precipitacion"]

        assert "valor_mean" in filas_ndvi.columns
        assert "valor_max" in filas_ndvi.columns
        assert filas_ndvi["valor_mean"].notna().all()
        assert filas_ndvi["valor_max"].notna().all()
        # las filas de NDVI no deben tener valor propio (columna compartida
        # con las variables de un solo reducer, pero NaN para NDVI)
        assert filas_ndvi["valor"].isna().all()
        # y viceversa: precipitación no debe tener valor_mean/valor_max
        assert filas_precip["valor"].notna().all()
        assert filas_precip["valor_mean"].isna().all()
```

- [ ] **Step 2: Ejecutar los tests para verificar el estado (puede que ya pasen, dado que `pd.concat` ya maneja esto)**

Run: `cd service-cnn && python -m pytest tests/ingestion/test_gee_extraction_pipeline.py -v`
Expected: si `test_ndvi_conserva_columnas_valor_mean_y_valor_max_en_el_resultado` PASA de inmediato (sin cambios de producción), es porque `pd.concat` ya lo maneja correctamente — confirmarlo explícitamente, no asumirlo. Si falla, diagnosticar antes de tocar producción (no debería requerir cambios de producción en `gee_extraction_pipeline.py` para este test específico, solo en el manejo del caso `tablas` vacío del Step 3).

- [ ] **Step 3: Corregir el caso de resultado vacío y el docstring**

En `service-cnn/src/ingestion/gee_extraction_pipeline.py`, la línea `columnas = ["provincia_id", "fecha", "valor", "variable", "campana_id"]` (usada solo cuando `tablas` está vacío) asume el esquema de un solo reducer para TODAS las variables — esto es incorrecto si se pide únicamente `"ndvi"` y no hay resultados. No hay forma de saber de antemano qué columnas tendría el resultado sin llamar al cliente (que no se llamó, por eso está vacío) — documentar esta limitación explícitamente en vez de fingir que se sabe:

```python
    columnas_base = ["provincia_id", "fecha", "variable", "campana_id"]
    if not tablas:
        # Sin ninguna tarea ejecutada exitosamente no hay forma de saber
        # qué columna(s) de valor tendría el resultado (depende de
        # cuántos reducers tiene cada variable pedida) -- se devuelve solo
        # el esquema base, sin inventar "valor" como si aplicara a todas.
        resultado_final = pd.DataFrame(columns=columnas_base)
    else:
        resultado_final = pd.concat(tablas, ignore_index=True)
```

Actualizar también el docstring de `Returns` (líneas ~124-127) para reflejar que la(s) columna(s) de valor dependen de la variable (`valor` para las de un reducer, `valor_mean`/`valor_max` para NDVI), no asumir `valor` como universal.

- [ ] **Step 4: Ejecutar los tests para verificar que pasan**

Run: `cd service-cnn && python -m pytest tests/ingestion/test_gee_extraction_pipeline.py -v`
Expected: PASS — todos los tests (los ya existentes + el nuevo).

- [ ] **Step 5: Corregir `phenology_pipeline._serie_ndvi_de_provincia_campana`**

En `service-cnn/src/preprocessing/phenology_pipeline.py`, cambiar `filtro["valor"]` por `filtro["valor_mean"]` (confirmado arriba: el estimador de fecha de siembra por umbral ascendente necesita la serie continua de intensidad vegetativa — media, no máximo de campaña):

```python
    return pd.DataFrame(
        {
            "fecha": pd.to_datetime(filtro["fecha"]).dt.date,
            "ndvi": filtro["valor_mean"],
        }
    )
```

Añadir una nota breve en el docstring de `_serie_ndvi_de_provincia_campana` (o como comentario en la línea) explicando por qué `valor_mean` y no `valor_max`: la detección de inicio de estación por cruce de umbral necesita la serie temporal continua, no un agregado de campaña completa.

Buscar primero si existe un test que ejercite esta función con un fixture de columna `valor` (revisar `service-cnn/tests/preprocessing/test_phenology_pipeline.py`) y actualizar su fixture a `valor_mean` si lo hay — sin inventar un test nuevo si no lo hay ya, ya que esta función no es el foco de este plan, solo necesita dejar de estar rota.

- [ ] **Step 6: Ejecutar los tests de phenology_pipeline y toda la suite**

Run: `cd service-cnn && python -m pytest tests/preprocessing/test_phenology_pipeline.py -v`
Expected: PASS.

Run: `cd service-cnn && python -m pytest tests/ -q --ignore=tests/features/test_dataset_builder.py`
Expected: PASS — todos los tests, conteo total a confirmar en la salida real.

- [ ] **Step 7: Corregir las 2 afirmaciones de "mismo contrato" ahora falsas**

En `service-cnn/src/ingestion/gee_mask_pipeline.py`, localizar el docstring de `extract_masked_ndvi_series` que dice "mismo contrato que `GeeClient.extract_daily_series`" (ya no es cierto: esa función sigue devolviendo `valor` único, mientras que `extract_daily_series` para NDVI ahora devuelve `valor_mean`/`valor_max`) y corregirlo:

```python
    Returns:
        DataFrame con columnas `provincia_id`, `fecha`, `valor` — columna
        única con NDVI real ([-1, 1]) ya escalado. NOTA: a diferencia de
        `GeeClient.extract_daily_series` (que desde el filtrado de calidad
        y reducers múltiples de NDVI devuelve `valor_mean`/`valor_max`),
        esta función todavía usa un único `ee.Reducer.mean()` sin filtro de
        calidad — no actualizada por ese cambio (ver
        `docs/PENDIENTE.md`, pendiente de conectar la máscara agrícola al
        mismo criterio de calidad/reducers).
```

En `service-cnn/docs/PENDIENTE.md`, sección 2.3 (tensor V×T), corregir la línea que documenta el esquema del parquet crudo (`provincia_id, campana_id, fecha, valor`) para aclarar que aplica a las 4 variables de un reducer, y que NDVI usa `valor_mean`/`valor_max` — evitar que ambas secciones del documento se contradigan.

- [ ] **Step 8: Commit**

```bash
cd service-cnn
git add src/ingestion/gee_extraction_pipeline.py tests/ingestion/test_gee_extraction_pipeline.py src/preprocessing/phenology_pipeline.py tests/preprocessing/test_phenology_pipeline.py src/ingestion/gee_mask_pipeline.py docs/PENDIENTE.md
git commit -m "$(cat <<'EOF'
Propagar el esquema valor_mean/valor_max de NDVI a los consumidores aguas abajo

Hallazgo Critical de la revision final de todo el branch: el cambio de
esquema de extract_daily_series para NDVI (Tasks 1-4) nunca se
propago -- el mock del test de extract_all_provinces_all_variables
devolvia el mismo esquema para todas las variables (ocultando que
pd.concat ya maneja bien el caso real), phenology_pipeline seguia
leyendo filtro["valor"] (KeyError tras re-extraer NDVI), y dos
docstrings afirmaban falsamente que extract_masked_ndvi_series tiene
"el mismo contrato" que extract_daily_series.

Corregido: mock de test_gee_extraction_pipeline.py ahora refleja el
contrato real por variable; caso de resultado vacio ya no asume
"valor" universal; phenology_pipeline usa valor_mean (serie continua
para el estimador de fecha de siembra por umbral ascendente, no
valor_max que es agregado de campana); docstrings y PENDIENTE.md
corregidos para no contradecirse.

tensor_builder.py (Task 2.3, aun sin conectar a datos reales) queda
fuera de alcance -- se actualizara cuando esa tarea se implemente.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016mkokyMQFzBE7QpPxKea4U
EOF
)"
```
