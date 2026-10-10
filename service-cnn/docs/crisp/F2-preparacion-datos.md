# F2 · Preparación de datos

- **DSRM**: A3 (diseño y desarrollo).
- **Cronograma**: Act. 3 máscara y descargas (ago→sep 2026); Act. 4 integración y alineación fenológica (sep→oct 2026); Act. 5 preprocesamiento y EDA (oct 2026).
- **Entregables**: máscara validada y manifiesto de descargas (producto intermedio), E2 (dataset y diccionario de variables), E3 (informe de depuración y EDA).
- **Secciones de la tesis**: 4.9–4.11, 4.12.2, Tabla 7.
- **Estado de la fase**: ~50 %, solo código. **Gate: abierto** (depende de H1).

## Requisitos y métricas de calidad

| ID | Requisito / métrica | Estado | Evidencia |
|---|---|---|---|
| Q-F2-1 | Manifiesto con versión, fecha y hash por fuente (4.9, 4.18) | Parcial | `data/manifest/{gee_series_climaticas_espectrales,mascara_agricola,midagri_sisagri}.yaml`; verificar que incluyan hash |
| Q-F2-2 | Umbrales de la máscara fijados **antes** de la evaluación; `amplitud_ndvi_min` calibrado solo con campañas de entrenamiento (4.10.1) | **Pendiente** | `amplitud_ndvi_min` sin valor → `load_mask_thresholds` lanza `AmplitudNdviNoCalibradaError`; altitud 2500–4000 m y pendiente 14,04° **PROVISIONALES** |
| Q-F2-3 | NDVI con filtro de calidad (DetailedQA) y media + máximo por fase sobre máscara (Tabla 7) | Parcial | Implementado en código (`gee_client`, `gee_series_builder`); `extract_masked_ndvi_series` aún con `Reducer.mean()` sin filtro |
| Q-F2-4 | `data/raw/gee_series_crudo.parquet` regenerado con el esquema vigente (`valor_mean`/`valor_max`) | **Pendiente** | El parquet actual es esquema viejo (`valor` único, NDVI sin filtrar ni enmascarar) |
| Q-F2-5 | Alineación fenológica jerárquica (primario/secundario/respaldo) sin fuga de información (4.10.2, 4.12.2) | Parcial | `data/interim/fenologia_alineada.parquet`: 800 provincia-campañas × 5 fases, 688 primario / 81 secundario / 31 respaldo. El respaldo usa **todo el histórico** (exploratorio) |
| Q-F2-6 | Control: comparar alineación fenológica vs ventanas de calendario fijas (4.10.2) | Pendiente | Sin evidencia |
| Q-F2-7 | Tensor V×T con covariables estáticas (altitud) y dinámicas (% quinua) y diccionario de variables (4.10.3) | Parcial | `src/features/tensor_builder.py` y `tests/features/test_dataset_builder.py` (sin versionar); no conectado a datos reales |
| Q-F2-8 | Variable objetivo: nivel y anomalía con línea base solo con entrenamiento (4.11) | Parcial | `src/features/target_transform.py` probado con sintéticos; sin datos reales |
| Q-F2-9 | Cobertura de predictores ≥ 80 % por predictor/fase; exclusiones registradas con motivo (4.7) | Pendiente | N5 no calculado |
| Q-F2-10 | `run_preprocessing.py` orquesta máscara → alineación → tensor y `dvc.yaml` refleja dependencias reales | **Pendiente** | `run_preprocessing.py` es un placeholder (`NotImplementedError`) |
| Q-F2-11 | Horizonte principal: ventanas truncadas 30 días antes de la cosecha estimada, sin información posterior (4.10.4) | Parcial | `estimate_horizon_cutoff` + `build_all_phase_windows(dias_antes_de_cosecha=)` + flag `--dias-antes-de-cosecha` en `run_phenology_alignment.py`; tests verdes. **Falta**: regenerar `fenologia_alineada.parquet` con el flag, calcular la fecha de cosecha con duraciones calibradas solo en entrenamiento (4.10.4) y el horizonte secundario (cierre de floración) |

## Riesgos

| ID | Riesgo | Estado |
|---|---|---|
| R-F2-1 | Fuga temporal por respaldo departamental calculado con todo el histórico | **Mitigado en código, sin conectar** (ver C-F2-4); activo mientras se use el parquet exploratorio en la evaluación |
| R-F2-2 | Código nuevo contra datos viejos: `phenology_pipeline` falla con `KeyError: valor_mean` sobre el parquet actual | **Activo** |
| R-F2-3 | Señal NDVI poco representativa de quinua (máscara no distingue cultivos) | Mitigación diseñada (proporción de quinua, estratos); sin ejecutar |
| R-F2-4 | Supuestos de GEE no verificados con la API real: clase 40 de WorldCover = Cropland en sierra peruana, `ee.Terrain.slope()` en grados, volumen/tiempo de `sampleRegions` | **Abierto** |
| R-F2-5 | Reducción del conjunto analítico por cobertura insuficiente (Tabla 6) | Por medir tras N5 |
| R-F2-6 | Re-extracción GEE costosa (~2 h 34 min con 8 hilos) | Aceptado; evaluar `ee.batch.Export` |

## Tareas de control

| Control | Mitiga | Acción |
|---|---|---|
| C-F2-1 | Q-F2-2, R-F2-4 | Verificar manualmente contra GEE real (WorldCover clase 40, slope en grados, `sampleRegions`); luego ejecutar `sample_cells_for_calibration` + `calibrate_ndvi_amplitude_threshold` sobre campañas de entrenamiento y registrar `amplitud_ndvi_min` en el manifiesto |
| C-F2-2 | Q-F2-2 | Obtener aprobación del asesor de altitud/pendiente (quitar `PROVISIONAL`) |
| C-F2-3 | Q-F2-3, Q-F2-4, R-F2-2 | Alinear `extract_masked_ndvi_series` con filtro y reducers; re-extraer NDVI enmascarado y regenerar el parquet; ejecutar la suite completa después |
| C-F2-4 | R-F2-1 | **Hecho en código (2026-10-09)**: `phenology_pipeline.estimate_departmental_modal_dates(siembra, prov→depto, campanas_entrenamiento)` filtra a entrenamiento; 3 tests, incluido uno que verifica que el resultado no cambia al alterar una campaña de prueba. La versión `_exploratory` ahora delega en ella usando todas las campañas. **Falta conectarla por pliegue** en `run_preprocessing.py` (C-F2-7) y en el experimento; mientras tanto `fenologia_alineada.parquet` sigue siendo exploratorio |
| C-F2-5 | Q-F2-6 | Implementar la comparación con ventanas fijas como análisis de sensibilidad |
| C-F2-6 | Q-F2-9 | Calcular N5 con desagregación por campaña y departamento; log de exclusiones |
| C-F2-7 | Q-F2-10 | Implementar `run_preprocessing.py` y actualizar el stage `preprocess` de `dvc.yaml` |
| C-F2-8 | Q-F2-7 | Versionar `test_dataset_builder.py` y producir diccionario de variables (E2) |

## Entregables y criterio de gate

- **E2**: conjunto de datos (`data/processed`) + diccionario de variables (forma, unidad, fuente, transformación, rol).
- **E3**: informe de depuración y EDA (conteos de entrada, exclusión, imputación y cobertura por campaña).
- **Gate F2**: Q-F2-1…Q-F2-10 cumplidos, suite de tests verde sobre datos regenerados, `dvc repro preprocess` reproduce `data/processed`, E2 y E3 entregados, umbrales aprobados y fijados antes de cualquier evaluación.
