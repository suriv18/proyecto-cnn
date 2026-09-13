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

### 2.1 Máscara de superficie agrícola (bloquea `mask.py` en producción)
`src/preprocessing/mask.py` (`apply_agricultural_mask`) ya está implementado y probado con datos sintéticos, pero necesita, **por celda** (no agregado a nivel de polígono provincial como la extracción actual):
- Capa de cobertura de suelo (candidato: ESA WorldCover, en GEE).
- Modelo digital de elevación / pendiente (candidato: SRTM, en GEE).
- Amplitud NDVI (ya se puede derivar de `data/raw/gee_series_crudo.parquet`, variable `ndvi`, calculando max-min por celda/campaña).

Esto requiere una extracción GEE nueva y distinta a `gee_extraction_pipeline.py` (que opera a nivel de polígono provincial completo, agregando con `reduceRegions`); la máscara necesita resolución de celda dentro de cada provincia — diseño aún no explorado.

### 2.2 Integrar `run_preprocessing.py` (hoy es un placeholder con `NotImplementedError`)
Una vez resuelta 2.1, este script debe orquestar en orden:
1. `apply_agricultural_mask` (con umbrales calibrados vía `calibrate_ndvi_amplitude_threshold` sobre campañas de entrenamiento).
2. La alineación fenológica YA EJECUTABLE (`scripts/run_phenology_alignment.py`) — falta conectarla aquí en vez de dejarla como script standalone.
3. Actualizar `dvc.yaml` (stage `preprocess`) con las dependencias reales una vez este script deje de ser un placeholder.

### 2.3 Construcción del tensor V×T (features/tensor_builder.py)
Ya implementado y probado con datos sintéticos (`build_observation_tensor`, `build_variable_dictionary`). Falta conectarlo con datos reales:
- Input: `data/raw/gee_series_crudo.parquet` (series) + `data/interim/fenologia_alineada.parquet` (ventanas de fase) → agregación por fase con `src/ingestion/gee_series_builder.aggregate_by_phase` (ya implementado, no verificado aún con datos reales de punta a punta).
- Nota: `aggregate_by_phase` espera columnas `provincia_id, campana_id, fecha, valor` por serie — el parquet crudo ya tiene ese formato (con columna extra `variable` a filtrar antes de pasar).
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
