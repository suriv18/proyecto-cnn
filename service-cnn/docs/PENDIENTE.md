# Pendiente — próxima sesión

Estado al cierre: **256/256 tests pasando**, rama `developer` (commit `5abb974`, pusheado a origin). `main` sigue sin estos cambios — no fusionado todavía.

Datos reales ya generados y verificados (regenerables vía los scripts, no versionados en git — ver `.gitignore`):
- `data/raw/gee_series_crudo.parquet` — 1 148 160 filas, 80 provincias × 10 campañas (2016-2025) × 5 variables (CHIRPS, ERA5-Land ×3, MODIS NDVI), 0 nulos, valores en unidad real.
- `data/interim/fenologia_alineada.parquet` — 4000 filas (800 provincia-campañas × 5 fases BBCH), 0 fallos: 688 nivel primario (MIDAGRI), 81 secundario (NDVI), 31 respaldo departamental exploratorio.

## 1. Decisiones que requieren al asesor (bloquean, no unilaterales)

- [ ] **Confirmar la ventana de extracción amplia** usada en `gee_extraction_pipeline.campana_a_ventana_extraccion` (año calendario jul(t-1)-jun(t) del ciclo agrícola) como interpretación correcta antes de proceder a la Actividad 4. Ver `data/manifest/gee_series_climaticas_espectrales.yaml`, sección `pendiente`.
- [ ] **Aprobación formal de la enmienda de periodo** (2006-2024 → 2016-2025) — ya formalizada en `data/manifest/midagri_sisagri.yaml` (`enmienda_metodologica.estado: "PROPUESTA"`) y en la tesis (.docx editado), pero falta el visto bueno del asesor antes de avanzar a la Actividad 4 (integración del conjunto de datos).
- [ ] **Decidir si ampliar N3** a las 76 provincias con producción real observada de quinua (14 más que las 62 preespecificadas en la Tabla 3 de la tesis) — hallazgo de la auditoría H1, no resuelto unilateralmente.
- [ ] **Validar el nivel de respaldo departamental exploratorio** (`phenology_pipeline.estimate_departmental_modal_dates_exploratory`) — usa TODO el histórico disponible sin distinguir entrenamiento/prueba; es una aproximación aceptada solo para esta primera pasada. Para H2/evaluación definitiva debe recalcularse restringido a las campañas de ENTRENAMIENTO de cada pliegue de validación (violaría la regla anti-fuga temporal, sección 4.12.2, si se usa tal cual).
- [ ] Confirmar en H1 si las filas a nivel DISTRITO de SISAGRI deben sumarse a nivel PROVINCIA — ya se está haciendo así en `midagri_siembra_mensual.py` y `compute_n4_n5.py`, pero sigue marcado como "pendiente confirmar" en `data/manifest/midagri_sisagri.yaml`.
- [ ] Verificar con SENAMHI/INEI si existe otra fuente que cubra 2006-2015 (brecha de cobertura de MIDAGRI ya documentada, probablemente no hay fuente tabular oficial).

## 2. Trabajo técnico pendiente (siguiente pieza natural del pipeline)

### 2.1 Máscara de superficie agrícola — extracción GEE ya implementada, falta calibrar y conectar a producción

`src/ingestion/gee_mask_pipeline.py` ya implementa la extracción server-side
por celda (diseño: `docs/superpowers/specs/2026-09-13-mascara-agricola-gee-design.md`;
plan: `docs/superpowers/plans/2026-09-13-mascara-agricola-gee.md`):
- `build_agricultural_mask_image`: máscara booleana (WorldCover v200 2021 +
  SRTM/slope + amplitud NDVI por campaña, con `factor_escala` aplicado).
- `extract_masked_ndvi_series`: NDVI agregado por provincia-campaña, ya
  enmascarado — mismo contrato que `GeeClient.extract_daily_series`.
- `sample_cells_for_calibration`: muestra acotada de celdas para calibrar
  con `mask.py` (sin cambios), respetando el límite de 5000 elementos de
  `getInfo()`.
- `gee_extraction_pipeline.extract_all_provinces_all_variables` ya acepta
  `umbrales_mascara: MaskThresholds | None` para enrutar `"ndvi"` — cambio
  aditivo, `None` preserva el comportamiento anterior para todas las
  variables.

288/288 tests pasando (5 tareas del plan, cada una revisada por separado; se
corrigió en el camino un bug real de correctitud — el factor de escala de
NDVI no se aplicaba a la amplitud antes de compararla con el umbral, lo que
habría hecho el criterio un no-op en producción — más una revisión final de
todo el branch que encontró y corrigió 2 bugs Critical adicionales: la
máscara se recalculaba con la ventana MENSUAL de paginación en vez de la
ventana de CAMPAÑA completa, y `filterDate` recibía objetos `date` crudos en
vez de `.isoformat()`).

**Auditoría de cumplimiento contra la tesis (sección 4.10.1 y Tabla 7,
`docs/Proyecto_Tesis_Maestria_UNMSM_WMSR.docx`)**: la tesis NO fija nombres
de producto ni umbrales numéricos para la máscara — el texto dice
literalmente que "el nombre, versión, resolución y fecha de cada capa, así
como los umbrales de altitud, pendiente y amplitud, se fijarán y
registrarán antes de la evaluación definitiva". ESA WorldCover v200 2021 y
SRTM 30m (elegidos en esta sesión) son decisiones válidas dentro de ese
margen, pero **falta formalizarlas en un manifiesto reproducible** (mismo
patrón que `data/manifest/gee_series_climaticas_espectrales.yaml` ya usa
para las otras variables) — hoy solo existen en el código.

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
- [ ] **Formalizar ESA WorldCover v200 2021 y SRTM 30m en un manifiesto**
  reproducible (nombre, versión, resolución, fecha de descarga), como exige
  la tesis (sección 4.10.1) — hoy solo están en el código, no registrados
  como manifiesto.
- [x] **[Gap crítico, resuelto en código]** ~~La Tabla 7 de la tesis (sección
  4.9 / Anexo 01) exige para NDVI: "Filtrado de calidad; media y máximo por
  fase sobre máscara agrícola." Ni GeeClient.extract_daily_series ni
  extract_masked_ndvi_series aplicaban filtrado de calidad ni calculaban el
  máximo — ambos solo usaban ee.Reducer.mean().~~
  Resuelto: `GeeCollectionSpec` (NDVI) declara `banda_calidad="DetailedQA"`
  y `reducers=(MEAN, MAX)`; `GeeClient.extract_daily_series` filtra por
  calidad (VI Quality bits 0-1 en {00,01}, sin nube adyacente bit 8, sin
  nubes mixtas bit 10 — verificado contra el catálogo oficial de
  `MODIS/061/MOD13Q1` y el MODIS VI User's Guide) y devuelve
  `valor_mean`/`valor_max`; `aggregate_by_phase` (`gee_series_builder.py`)
  agrega ambas columnas con su propia regla (`reglas_por_columna: dict[str,
  AggregationRule]`). Ver
  `docs/superpowers/plans/2026-09-13-filtrado-calidad-max-ndvi.md`.
  301/301 tests pasando (incluye un bug Critical encontrado y corregido en
  el camino: `ee.Reducer.combine(sharedInputs=True)` antepone el nombre de
  la banda al nombre del reducer en la salida real de `reduceRegions` —ej.
  `NDVI_mean`, no `mean`— y el código inicial no lo tenía en cuenta).
  **Pendiente**: re-ejecutar la extracción real de NDVI contra la API de
  GEE (Actividad 3) para regenerar `data/raw/gee_series_crudo.parquet` con
  el filtro de calidad y ambos reducers aplicados — el parquet actual sigue
  siendo NDVI sin filtrar, solo media, hasta que se re-extraiga. `extract_
  masked_ndvi_series` (máscara agrícola) también debe actualizarse para
  usar el mismo filtrado/reducers cuando se conecte a producción (hoy sigue
  usando `ee.Reducer.mean()` sin filtro — no tocado por este ciclo, que se
  centró en `GeeClient.extract_daily_series`).

Deuda menor conocida (hallazgos Minor/Important de baja severidad de la
revisión final del branch, no bloqueantes):
- [ ] `sample_cells_for_calibration` (solo calibración/auditoría, no
  producción): 3 de sus 4 bandas exportadas (`cobertura_suelo`,
  `altitud_m`, `pendiente_grados`) no tienen verificación algebraica real
  en el doble de prueba de `test_gee_mask_pipeline.py` — comparten una
  sola `_FakeCompositeImage` sin identidad simbólica propia (a diferencia
  de `amplitud_ndvi`, que sí la tiene). Si se reordenara accidentalmente
  qué banda recibe qué `.rename()`, ningún test lo detectaría hoy.
- [ ] `celda_id` en `sample_cells_for_calibration` es `f"celda_{i}"`
  (secuencial), no `f"{provincia_id}_{i}"` como decía el diseño original —
  `sampleRegions(geometries=False)` no propaga `provincia_id` por celda, así
  que las filas de auditoría quedan sin atribución de provincia. Evaluar si
  vale la pena resolverlo antes de usar la función para auditoría real.
- [ ] `n_por_provincia` en `sample_cells_for_calibration` valida el tamaño
  total contra el límite de 5000, pero nunca se pasa como `numPixels` (o
  equivalente) a `sampleRegions` — el nombre del parámetro promete un tope
  por provincia que la llamada real no impone.

### 2.2 Integrar `run_preprocessing.py` (hoy es un placeholder con `NotImplementedError`)
Una vez resuelta 2.1, este script debe orquestar en orden:
1. `apply_agricultural_mask` (con umbrales calibrados vía `calibrate_ndvi_amplitude_threshold` sobre campañas de entrenamiento).
2. La alineación fenológica YA EJECUTABLE (`scripts/run_phenology_alignment.py`) — falta conectarla aquí en vez de dejarla como script standalone.
3. Actualizar `dvc.yaml` (stage `preprocess`) con las dependencias reales una vez este script deje de ser un placeholder.

### 2.3 Construcción del tensor V×T (features/tensor_builder.py)
Ya implementado y probado con datos sintéticos (`build_observation_tensor`, `build_variable_dictionary`). Falta conectarlo con datos reales:
- Input: `data/raw/gee_series_crudo.parquet` (series) + `data/interim/fenologia_alineada.parquet` (ventanas de fase) → agregación por fase con `src/ingestion/gee_series_builder.aggregate_by_phase` (ya implementado, no verificado aún con datos reales de punta a punta).
- Nota: `aggregate_by_phase` espera columnas `provincia_id, campana_id, fecha, valor` por serie — el parquet crudo tiene ese formato para las 4 variables de un solo reducer (precipitación, ERA5-Land), pero NDVI tiene `valor_mean`/`valor_max` en vez de `valor` (filtrado de calidad + media y máximo, Tabla 7 — ver sección 2.1); hay que filtrar por `variable` antes de pasar cada serie, y para NDVI decidir/adaptar qué columna(s) alimentan `aggregate_by_phase` (probablemente ambas, con una regla de agregación propia por columna, como ya hace `gee_series_builder.aggregate_by_phase` vía `reglas_por_columna`).
- Definir explícitamente qué covariables estáticas (altitud) y dinámicas (% superficie sembrada con quinua) se añaden al tensor, y de dónde salen (altitud: pendiente del MDE de 2.1; % superficie: ya disponible en `midagri_siembra_mensual`/`midagri_aggregation`).

### 2.4 Tratamiento de la variable objetivo
`features/target_transform.py` (`fit_provincial_baseline`, `to_anomaly`) ya implementado y probado con sintéticos. Falta conectar con el rendimiento real reconstruido por `midagri_loader.load_midagri_production` / `midagri_aggregation.aggregate_to_provincia_campana` (ya ejecutado sobre datos reales en la auditoría H1 — reutilizar esa salida, no releer SISAGRI de cero).

### 2.5 `run_experiment.py` y `run_shap.py` (ambos placeholders con `NotImplementedError`)
Solo tienen sentido después de 2.3 y 2.4 (necesitan el tensor final construido). El código de modelado (`models/`), evaluación (`evaluation/`) y explicabilidad (`explainability/`) ya está completo y probado con sintéticos — falta únicamente la orquestación real.

## 3. Deuda menor / limpieza

- [ ] `provincia_a_departamento` en `phenology_pipeline.build_all_phase_windows` se pasa hoy desde `configs/provincias.csv` en cada script — considerar si merece una función utilitaria compartida si se repite en un tercer lugar.
- [ ] Revisar si `gee_extraction_pipeline.extract_all_provinces_all_variables` necesita reintentos automáticos ante fallos transitorios de red (hoy solo registra el fallo y continúa; en la ejecución real de 600 llamadas no hubo ningún fallo, pero no hay garantía en corridas futuras).
- [ ] El tiempo real de extracción GEE completa fue ~2h34min con `max_workers=8` (más lento de lo estimado por contención de GIL en deserialización JSON) — si se necesita re-ejecutar con más frecuencia, evaluar `multiprocessing` en vez de `threading`, o exportar a Google Drive/Cloud Storage vía `ee.batch.Export` en lugar de `getInfo()` síncrono.

## 4. Dónde retomar mañana (orden sugerido)

1. Empezar por **2.1 (máscara agrícola)** si se decide seguir el orden natural del pipeline de la tesis, **o**
2. Saltar directo a **2.3 (tensor V×T)** omitiendo la máscara por ahora (usando NDVI sin enmascarar como aproximación), para tener antes un extremo a extremo modelable — decisión de producto, no técnica, a definir con el usuario al retomar.
3. En paralelo, resolver las decisiones de la sección 1 no requiere código y puede avanzar independientemente (son preguntas para el asesor).
