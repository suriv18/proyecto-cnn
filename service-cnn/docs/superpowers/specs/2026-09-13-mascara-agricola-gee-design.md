# Diseño: máscara de superficie agrícola por celda (GEE)

Retoma el pendiente 2.1 de `service-cnn/docs/PENDIENTE.md`: `apply_agricultural_mask`
(`src/preprocessing/mask.py`) ya está implementado y probado con datos sintéticos, pero
necesita datos reales **por celda** (cobertura de suelo, altitud/pendiente, amplitud NDVI),
a diferencia de la extracción actual (`gee_extraction_pipeline.py`), que agrega a nivel de
polígono provincial completo con `reduceRegions`.

## Contexto y evidencia relevante

- `data/manifest/gee_series_climaticas_espectrales.yaml` ya documenta el gap: la extracción
  actual es agregada a nivel de polígono, no de celda, y deja pendiente esta extracción.
- `gee_extraction_pipeline.py` (`dividir_en_meses`) ya chocó una vez con el límite de 5000
  elementos de `ee.FeatureCollection.getInfo()`: 80 provincias × 365 días diarios = 29 200
  elementos excedía el límite; tuvieron que paginar por mes. A resolución de celda (NDVI
  250 m, SRTM ~30 m) sobre 80 provincias enteras, el volumen de píxeles sería órdenes de
  magnitud mayor — traer celdas individuales a pandas vía `.getInfo()` es inviable con el
  cliente actual.
- `tensor_builder.py` y `gee_series_builder.aggregate_by_phase` ya esperan la serie NDVI en
  formato agregado `provincia_id, campana_id, fecha, valor` — el mismo que produce hoy
  `GeeClient.extract_daily_series`. Ningún consumidor aguas abajo espera celdas
  individuales.
- `tests/preprocessing/test_mask.py` fija el comportamiento exacto de
  `apply_agricultural_mask` (los 4 criterios, columna `motivo_exclusion` con todos los
  motivos concatenados) y de `calibrate_ndvi_amplitude_threshold` (percentil sobre amplitud
  NDVI, solo con datos de entrenamiento). No debe romperse.
- Patrón arquitectónico ya establecido en `src/ingestion/`: `ee` nunca se importa a nivel de
  módulo (se inyecta vía constructor/parámetro), todo cliente GEE tiene un doble de prueba
  en memoria para testear sin credenciales ni red.

## Decisión de arquitectura

La máscara se reimplementa **server-side en GEE** (`ee.Image` booleana) para el flujo de
producción — es la única opción viable dado el límite de `.getInfo()` y el volumen de
celdas a resolución de píxel. `apply_agricultural_mask` y
`calibrate_ndvi_amplitude_threshold` (pandas) se conservan sin cambios en su lógica,
reservados para calibración de umbrales y auditoría con una **muestra acotada** de celdas.

## Fuentes de datos

- **Cobertura de suelo**: `ESA/WorldCover/v200/2021` (10 m), banda `Map`, clase 40
  ("Cropland") reclasificada a agrícola. Estática para las 10 campañas 2016-2025 (cobertura
  de suelo cambia lento en zonas de sierra; una sola fecha es una aproximación aceptada).
- **Altitud / pendiente**: `USGS/SRTMGL1_003` (30 m), banda `elevation`; pendiente derivada
  con `ee.Terrain.slope()`.
- **Amplitud NDVI**: `MODIS/061/MOD13Q1`, banda `NDVI`, reducida con `.max()`/`.min()` sobre
  la ventana de extracción de **cada campaña** (`campana_a_ventana_extraccion`, ya existente
  en `gee_extraction_pipeline.py`) — consistente con que el resto del pipeline ya trata la
  extracción por campaña, y captura cambios de uso de suelo/humedad campaña a campaña.

## Componentes nuevos (`src/ingestion/gee_mask_pipeline.py`)

Mismo patrón de inyección de dependencias que `gee_client.py`: no importa `ee` a nivel de
módulo.

### `build_agricultural_mask_image(ee_module, umbrales: MaskThresholds, fecha_inicio, fecha_fin) -> ee.Image`

Construye la máscara booleana combinando los 4 criterios:

1. Cobertura: `WorldCover.select("Map")` reclasificada — `True` solo si es clase 40.
2. Altitud: `altitud_min_m <= elevation <= altitud_max_m`.
3. Pendiente: `slope <= pendiente_max_grados`.
4. Amplitud NDVI: `(max - min) >= amplitud_ndvi_min` (con `factor_escala` de `gee_config`
   aplicado antes de comparar).

Combina los 4 booleanos con `.And()` encadenado → una `ee.Image` de 1 banda.

### `extract_masked_ndvi_series(cliente: GeeClient, geometrias_por_provincia, umbrales, fecha_inicio, fecha_fin) -> pd.DataFrame`

Análogo a `GeeClient.extract_daily_series`, pero aplica `.updateMask(mascara)` a cada
imagen NDVI antes de `reduceRegions` (mismo patrón `ee.Reducer.mean()` + `.flatten()`).
Devuelve el mismo formato `provincia_id, fecha, valor` — no cambia el contrato aguas abajo.

### `sample_cells_for_calibration(ee_module, geometrias_por_provincia, fecha_inicio, fecha_fin, escala_muestreo_m, n_por_provincia) -> pd.DataFrame`

Usa `ee.Image.sampleRegions()` (no `reduceRegions`) sobre una imagen compuesta de 4 bandas
(`cobertura_suelo` reclasificada a texto `"cultivo"`/`"no_cultivo"`, `altitud_m`,
`pendiente_grados`, `amplitud_ndvi`). Devuelve exactamente las columnas que ya espera
`apply_agricultural_mask`: `celda_id, cobertura_suelo, altitud_m, pendiente_grados,
amplitud_ndvi` (`celda_id` sintético `f"{provincia_id}_{i}"`).

Valida `n_por_provincia * n_provincias < 5000` y lanza `ValueError` explícito si se excede
(mismo estilo defensivo que `dividir_en_meses`), en vez de dejar que `.getInfo()` falle con
un mensaje críptico de la API.

## Cambio en `gee_extraction_pipeline.py`

`extract_all_provinces_all_variables` gana un parámetro opcional `umbrales_mascara:
MaskThresholds | None`. Si se pasa y la variable es `"ndvi"`, usa
`extract_masked_ndvi_series` en vez de `cliente.extract_daily_series`. Las demás 4
variables (CHIRPS, ERA5-Land ×3) no se ven afectadas. No rompe la firma existente
(parámetro opcional, default `None` preserva el comportamiento actual).

## Umbrales (`MaskThresholds`)

El dataclass ya existente en `mask.py` es la única fuente de verdad, reutilizado por ambos
caminos (GEE y pandas):

1. `amplitud_ndvi_min`: calibrado con `calibrate_ndvi_amplitude_threshold` sobre una muestra
   de `sample_cells_for_calibration` restringida a campañas de **entrenamiento** (mismo
   principio anti-fuga temporal que ya aplica esa función).
2. `altitud_min_m`, `altitud_max_m`, `pendiente_max_grados`: fijados por criterio agronómico
   documentado (no calibración estadística — el pendiente actual no indica que estos se
   calibren). Se dejan como parámetros explícitos del script de calibración, a definir con
   el asesor si corresponde.

Flujo de calibración (script nuevo, ej. `scripts/calibrate_agricultural_mask.py`):
1. `sample_cells_for_calibration` sobre campañas de entrenamiento.
2. `calibrate_ndvi_amplitude_threshold` sobre esa muestra → `amplitud_ndvi_min`.
3. Fijar los otros 3 umbrales por criterio agronómico.
4. Instanciar `MaskThresholds` final; usarlo para auditar con `apply_agricultural_mask`
   (reporta `motivo_exclusion` sobre la muestra) y para producción vía
   `extract_masked_ndvi_series`.

## Testing

Mismo patrón que `tests/ingestion/test_gee_client.py`: doble de prueba de `ee` en memoria,
sin credenciales ni red.

- `tests/ingestion/test_gee_mask_pipeline.py`:
  - `build_agricultural_mask_image`: verifica que se consultan las 3 fuentes correctas
    (WorldCover, SRTM, MOD13Q1) y que el resultado combina los 4 criterios con `.And()`.
  - `extract_masked_ndvi_series`: verifica que aplica `updateMask` antes de `reduceRegions`
    y que el esquema de salida coincide con `extract_daily_series`.
  - `sample_cells_for_calibration`: verifica columnas devueltas, reclasificación de
    cobertura, y que lanza `ValueError` si el tamaño de muestra excede el límite de 5000.
  - Test de consistencia: dado el mismo `MaskThresholds`, una celda sintética con cobertura
    no agrícola queda excluida tanto por el camino GEE (`build_agricultural_mask_image`)
    como por el camino pandas (`apply_agricultural_mask`).
- `tests/ingestion/test_gee_extraction_pipeline.py`: extendido para verificar que, con
  `umbrales_mascara` presente, `"ndvi"` usa `extract_masked_ndvi_series` y las demás
  variables siguen usando `cliente.extract_daily_series` sin cambios.

**Fuera de test automatizado** (verificación manual contra la API real, documentada como
nota igual que ya hace `gee_client.py`):
- Que la clase 40 de WorldCover corresponde a "Cropland" en la sierra peruana.
- Que `ee.Terrain.slope()` sobre SRTM devuelve grados (no radianes).
- Volumen y tiempo real de `sampleRegions` contra la API real (el doble de prueba no puede
  simular cuotas/timeouts reales).

## Fuera de alcance de este diseño

- Definir los valores numéricos finales de `altitud_min_m`/`altitud_max_m`/
  `pendiente_max_grados` (decisión agronómica, no técnica).
- Integrar la salida de `extract_masked_ndvi_series` en `run_preprocessing.py` (pendiente
  2.2, posterior a este).
- Recalibración de `amplitud_ndvi_min` restringida por pliegue de validación (aplica igual
  que la limitación ya documentada para el respaldo departamental exploratorio en
  `PENDIENTE.md` sección 1).
