# Arquitectura Técnica y Estructura de Repositorio

**Proyecto:** Modelo interpretable con CNN-1D y SHAP para predecir el rendimiento de quinua en la sierra altoandina del Perú
**Depende de:** `docs/01-analisis-sistema.md`
**Alcance de este documento:** decisiones de stack, estructura de código, pipeline de datos, plan de modelado/evaluación/explicabilidad y reproducibilidad — sin implementar aún el código de modelado (Actividad 6 del cronograma).

## 1. Decisiones de stack

| Componente | Decisión | Justificación |
|---|---|---|
| Lenguaje | Python 3.11+ | Estándar del dominio; todas las librerías objetivo (PyTorch, scikit-learn, XGBoost, shap, geemap) tienen soporte maduro |
| Framework CNN-1D | **PyTorch** (revisado — ver nota) | TensorFlow no tiene build disponible para la versión de Python del entorno de desarrollo (3.14); PyTorch sí instala sin fricción. Es igualmente válido para CNN-1D + SHAP: Joshi et al. (2025) lo usó con `shap.DeepExplainer`. Sección 4.15 de la tesis exige declarar el explainer antes del piloto — la elección de framework no cambia esa obligación |
| Modelos comparadores | scikit-learn (Elastic-Net, Random Forest, SVR), XGBoost (librería oficial) | Exigidos explícitamente por la Tabla 8 de la tesis como comparadores confirmatorio/secundarios |
| Optimización de hiperparámetros | **Optuna** (TPE / optimización bayesiana con *pruning*) | Ver §1.1 — más eficiente que grid search bajo presupuesto de cómputo acotado (S/1500), y aplicable uniformemente a los 5 modelos ajustables para cumplir el presupuesto de tuning equiparado (sección 4.12.3) |
| Explicabilidad | Librería `shap` | Exigida por el diseño de la tesis (Cap. 2.3.2, 4.15) |
| Adquisición de datos satelitales/climáticos | Google Earth Engine (Python API) + `geemap` + `xarray`/`rioxarray` | Ver §2 — evita descargar rasters completos; permite exportar series agregadas por polígono provincial directamente a tabla |
| Adquisición de datos oficiales peruanos | `requests` sobre APIs/portales de datosabiertos.gob.pe (MIDAGRI) y SENAMHI | No requieren SDK dedicado; se documentan como descargas versionadas en el manifiesto de datos |
| Tracking de experimentos | MLflow (self-hosted) | Gratuito, suficiente para un tesista solo; registra métricas, hiperparámetros y artefactos por ejecución (pliegue × modelo × semilla) |
| Versionado de datos | DVC | Complementa a Git para versionar datasets grandes sin inflar el repositorio; se integra con el manifiesto de datos exigido en 4.18 |
| Pruebas | `pytest` | Sobre las transformaciones deterministas del pipeline (máscara, alineación, construcción de tensores, splits) — no sobre el entrenamiento estocástico en sí |
| Gestión de entorno | `pyproject.toml` + lockfile (uv o poetry) | Exigido por el plan de reproducibilidad (4.18, punto 3) |

### 1.1 Arquitectura CNN-1D de referencia: por qué Sabo et al. (2023) y no Li et al. (2025) o Cheema et al. (2026)

El dataset esperado de esta tesis, tras depuración, se ubica en el orden de **~300-700 celdas provincia-campaña** (cota superior N3 = 1178 en la Tabla 5; N4/N5 reales, menores, se determinan en H1). Se comparó cómo implementaron su CNN-1D otros investigadores en función del tamaño de su dataset, para adoptar una arquitectura defendible y no sobreajustada:

| Estudio | Tamaño dataset | Arquitectura CNN-1D | Explainer SHAP | Validación | Relevancia |
|---|---|---|---|---|---|
| **Sabo et al. (2023)** | 340-408 obs. (17 años × 20+ provincias) | 2 capas conv., kernel 2-3, 5-15 filtros, *avg pooling* + *global average pooling*, dropout 0.001-0.01, *batch normalization*, inicialización He Normal, L2 (1e-6), 0-1 capas densas — **256 a 1800 parámetros totales**. Tuning con Optuna (TPE, 100 trials, *pruning*) | No aplicó SHAP (foco en comparación de exactitud vs. modelos clásicos) | Leave-one-year-out | **Referencia arquitectónica principal** — dataset del mismo orden de magnitud que esta tesis |
| Srivastava et al. (2022) | 271 condados × 21 años | CNN-1D + capas FC concatenadas con suelo/fenología (kernels no publicados en el paper) | `KernelExplainer`, valores agregados por semana del ciclo | Walk-forward (3 años de prueba: 2017, 2018, 2019) | Antecedente metodológico más citado en la tesis; referencia para el protocolo de validación temporal walk-forward |
| Joshi et al. (2025) | 606 condados | CNN-1D: 3 capas conv. + 2 *max-pooling* + 2 capas FC, dropout tras conv. | `DeepExplainer` para redes, `TreeExplainer` para Random Forest | Entrena 2008-2018, prueba 2019-2021 | Referencia para declarar explainer distinto por familia de modelo |
| Li et al. (2025) | 4220 obs. (201 condados × 21 años) | CNNAtBiGRU con mecanismo de atención — 10x más grande que esta tesis | No especificado | Split aleatorio 80:20 + año fijo 2021 (más débil que el esquema de esta tesis) | **Descartado** como referencia arquitectónica: dataset demasiado grande, arquitectura sobredimensionada para datos escasos |
| Cheema et al. (2026) | 36 590 obs. | Dual-rama CNN+LSTM+Multi-Head Attention (128→64→32 filtros, LSTM 64 unidades, atención 4 heads) — 100x más grande | No especificado (solo *summary plots*) | No detallado | **Descartado** — mismo motivo que Li et al. |

**Decisión:** la arquitectura de partida replica la parsimonia de Sabo et al. (2023): 2 capas convolucionales (kernel 2-3, 5-15 filtros), *batch normalization*, *global average pooling*, dropout bajo, regularización L2, y como máximo una capa densa antes de la salida, con las covariables estáticas/dinámicas concatenadas después del bloque convolucional (sección 4.10.3 de la tesis). El espacio de búsqueda de Optuna se ajustará en el ensayo piloto (H2) según el N5 real determinado en H1 — si N5 resulta considerablemente mayor a lo esperado, se documentará como enmienda la ampliación de capacidad del modelo (ver sección 6.1 de la tesis, "Plan de contingencia").

**Riesgo documentado y plan de contingencia SHAP:** la librería `shap` tiene fricción conocida (2025-2026) entre versiones de TensorFlow/PyTorch y el explainer `DeepExplainer`, con reportes de violación de la propiedad de "local accuracy" en series temporales multivariadas. Plan: usar `DeepExplainer` como opción principal (más rápido, precedente de Joshi et al. 2025); si falla la validación de local accuracy durante el ensayo piloto (H2), migrar a `KernelExplainer` (model-agnostic, precedente de Srivastava et al. 2022, más robusto ante predictores correlacionados pero computacionalmente más costoso). Esta decisión se declara explícitamente en el registro de preespecificación (H2), como exige la sección 4.15 de la tesis.

## 2. Acceso a fuentes de datos

| Fuente | Vía de acceso | Formato de salida | Notas de reproducibilidad |
|---|---|---|---|
| MIDAGRI/SIEA | Dashboard `siea.midagri.gob.pe/herramientas/estadistica-agropecuarias` (la URL de datosabiertos.gob.pe citada originalmente ya no existe) | XLSX (`SISAGRI.xlsx`, 5 hojas fragmentadas por límite de filas de Excel) | Ver `data/manifest/midagri_sisagri.yaml` para fuente, hash, esquema confirmado y **enmienda metodológica de periodo** (cobertura real 2015-2026, no 2006-2024) |
| INEI/SIRTOD | Consulta web (`systems.inei.gob.pe/SIRTOD`) | Tabla exportable | Uso exclusivo de contraste/verificación (4.5.2), no como fuente primaria |
| CHIRPS v2.0 | Google Earth Engine (`UCSB-CHG/CHIRPS/DAILY`) | Serie agregada por polígono provincial (exportada a tabla, sin descargar rasters completos) | Resolución 0.05°, diaria |
| ERA5-Land | Google Earth Engine (`ECMWF/ERA5_LAND/HOURLY`) o Copernicus Climate Data Store (`cdsapi`) | NetCDF (CDS) o tabla agregada (GEE) | Resolución 0.1°, horaria/diaria; GEE evita cuota de descarga del CDS |
| SENAMHI | Portal/datos abiertos institucionales | CSV | Solo para validación de sesgo, no fuente principal |
| MODIS MOD13Q1 V061 | Google Earth Engine (`MODIS/061/MOD13Q1`) | Serie agregada por polígono, filtrada por calidad | 250m, 16 días; requiere máscara agrícola aplicada antes de agregar (RF3) |

Google Earth Engine requiere una cuenta vinculada a un proyecto de Google Cloud (registro no comercial/de investigación, gratuito) y se accede vía `earthengine-api` en Python, con `geemap` como capa de conveniencia para exportar series por geometría sin manipular rasters localmente. Este patrón evita el cuello de botella de almacenamiento local que enfrentarían descargas completas de CHIRPS/ERA5-Land/MODIS para 8 departamentos × 19 campañas.

**Implementado** (`src/ingestion/`): la integración con GEE se dividió en 3 módulos para que la lógica de negocio sea testeable sin credenciales de Google:
- `gee_config.py` — catálogo de colecciones/bandas/resolución/regla de agregación por variable (Tabla 7), sin dependencia de `ee`.
- `gee_series_builder.py` — agregación de series por fase fenológica con la regla anti-fuga de horizonte (rechaza explícitamente observaciones posteriores al punto de corte, sección 4.10.4); NaN explícito ante fases sin datos, nunca 0 falso.
- `gee_client.py` — capa delgada que sí ejecuta consultas reales; `ee` se inyecta en el constructor (patrón de inyección de dependencia) en vez de importarse a nivel de módulo, permitiendo probar el armado de consultas con un doble de prueba. `GeeClient.from_default()` importa `earthengine-api` de verdad y falla con mensaje claro si no está instalado.

**Implementado y verificado contra el archivo real** (`src/ingestion/midagri_headerless.py`, `src/preprocessing/campaign_calendar.py`): el esquema real de MIDAGRI (`AÑO, MES, DEPARTAMENTO, PROVINCIA, DISTRITO, PRODUCTO, SIEMBRA, COSECHA, PRODUCCION, ...`) reporta mes calendario, no campaña agrícola directa. `derive_campana_from_month()` deriva el año de cosecha (julio-diciembre del año t → campaña t+1; enero-junio del año t → campaña t, coherente con el ciclo set-nov/abr-jun de la quinua). `load_sisagri_headerless()` orquesta selección de columnas + derivación de campaña + `load_midagri_production`. **Hallazgo crítico**: la fuente real solo cubre campañas completas 2016-2025 (10, no las 19 originales de la tesis) — ver `data/manifest/midagri_sisagri.yaml`, sección `enmienda_metodologica`, para el detalle completo y sus implicaciones en el diseño experimental (ventana inicial de origen móvil, potencia estadística de los contrastes).

## 3. Estructura de repositorio

**Patrón arquitectónico confirmado mediante investigación (ver §3.1): pipeline DAG por etapas, estilo Cookiecutter Data Science v2 (CCDS v2)** — no arquitectura hexagonal/clean, que resultaría sobre-ingeniería para un ejecutor único sin necesidad de intercambiar infraestructura. El código de la tesis vive en `service-cnn/`, hermano de `docs/` en la raíz del repositorio.

```
proyecto-cnn-tesis/
├── docs/                          # análisis, arquitectura, tesis fuente (este documento)
└── service-cnn/                   # todo el código, datos y artefactos del artefacto computacional
    ├── data/
    │   ├── raw/                   # descargas sin modificar (versionadas con DVC)
    │   ├── interim/                # productos intermedios (máscara agrícola, series alineadas)
    │   ├── processed/              # tensores V×T finales por provincia-campaña
    │   └── manifest/                # manifiesto de datos: fuente/versión/hash/licencia/fecha
    ├── references/                 # diccionario de variables, documentación de fuentes (CCDS v2)
    ├── src/
    │   ├── ingestion/              # descarga/consulta de cada fuente (RF2)
    │   ├── preprocessing/          # máscara agrícola (RF3), alineación fenológica (RF4)
    │   ├── features/               # construcción del tensor V×T + covariables (RF5), target_transform.py (RF13)
    │   ├── models/                 # CNN-1D, Elastic-Net, RF, XGBoost, SVR, benchmarks B1-B3 (RF6)
    │   ├── evaluation/             # splits anti-fuga, métricas, contrastes estadísticos (RF7, RF12)
    │   ├── explainability/         # SHAP, estabilidad, oclusión, concordancia agronómica (RF8, RF9)
    │   └── reporting/              # generación de entregables E1-E8
    ├── notebooks/                  # exploración únicamente, nunca producción
    ├── configs/                    # hiperparámetros, semillas, espacios de búsqueda preespecificados (RF10)
    ├── reports/                    # entregables E1-E8 y figuras
    ├── tests/                      # pruebas de transformaciones deterministas del pipeline
    └── dvc.yaml                    # definición del DAG de orquestación (ver §3.2)
```

Reglas de esta estructura:
- Ningún notebook contiene lógica de producción; toda transformación reutilizable vive en `src/` e importa desde ahí.
- `configs/` es la única fuente de verdad para hiperparámetros y semillas — nunca hardcodeados en `src/`.
- `data/raw/` es inmutable una vez descargado (append-only); las correcciones se documentan como nuevas versiones en el manifiesto.
- `references/` documenta el diccionario de variables (unidad, fuente, transformación, rol — exigido en 4.10.3) y las decisiones de clasificación territorial (Tabla 4).

### 3.1 Por qué este patrón y no arquitectura hexagonal/clean

Se investigó explícitamente (2024-2026) si un patrón más "empresarial" (hexagonal, puertos y adaptadores, clean architecture) aportaría valor. Conclusión de la literatura consultada:

- **Cookiecutter Data Science v2** es el estándar de facto actual para proyectos de investigación en ciencia de datos/ML: filosofía de "datos crudos inmutables", "el análisis es un DAG dirigido", herramientas intercambiables encadenadas (filosofía Unix) en vez de un framework monolítico. La estructura ya propuesta coincide casi 1:1 con CCDS v2.
- **Arquitectura hexagonal/clean es sobre-ingeniería aquí**: la literatura específica de proyectos de investigación (a diferencia de software de producción) señala que estos patrones resuelven un problema — intercambiar adaptadores de infraestructura (múltiples bases de datos, múltiples consumidores) — que no existe en este proyecto: hay 5 fuentes de datos fijas, un solo consumidor (el propio pipeline de evaluación) y un único ejecutor (el tesista). Aplicar puertos y adaptadores completos añadiría capas de indirección sin beneficio, dificultando en lugar de facilitar la trazabilidad exigida por el diseño experimental.
- **Feature Store no aplica**: ese patrón resuelve el *training-serving skew* (features calculados distinto en entrenamiento vs. producción en vivo), problema inexistente aquí porque no hay *serving* — solo evaluación offline sobre campañas históricas.

### 3.2 Orquestación: DVC pipelines, no Kedro

Se comparó Kedro (framework de pipelines con nodos modulares) contra DVC pipelines (`dvc.yaml` con `stages`) para orquestar el DAG ingesta→máscara→alineación→tensor→split→entrenamiento→evaluación→SHAP:

- Kedro no versiona datos por defecto (requiere configuración adicional por dataset) y añadiría una segunda capa de abstracción/aprendizaje sobre un framework completo.
- DVC ya está en el stack (§1) para versionado de datos; sus `stages` en `dvc.yaml` dan el mismo DAG reproducible con dependencias explícitas (`deps`/`outs`) y ejecución incremental vía `dvc repro`, sin añadir un segundo framework de orquestación.
- **Decisión**: `service-cnn/dvc.yaml` define cada etapa del pipeline (ver §4) como un `stage` de DVC, con sus dependencias (código + datos de entrada) y salidas (datos procesados, métricas, artefactos) declaradas explícitamente — esto también satisface el requisito de trazabilidad y reproducibilidad (RNF2, sección 4.18) sin coste adicional de herramienta.

## 4. Pipeline de datos (detalle por etapa)

_Nota: todas las rutas de esta sección y las siguientes son relativas a `service-cnn/` (ej. `src/ingestion/` = `service-cnn/src/ingestion/`)._

| Etapa | Módulo | Regla anti-fuga (4.12.2) |
|---|---|---|
| Ingesta | `src/ingestion/` | Ninguna (datos crudos, sin transformación aprendida) |
| Máscara agrícola | `src/preprocessing/mask.py` (implementado) | Umbrales de altitud/pendiente/amplitud calibrados solo con campañas de entrenamiento o fuentes externas independientes del resultado de prueba (4.10.1). `apply_agricultural_mask` reporta TODOS los motivos de exclusión por celda (no solo el primero) para habilitar el análisis de sensibilidad por umbral; `calibrate_ndvi_amplitude_threshold` exige explícitamente datos ya restringidos a entrenamiento |
| Alineación fenológica | `src/preprocessing/phenology.py` (implementado) | Parámetros de fenometría NDVI estimados solo en entrenamiento; fecha modal departamental de respaldo estimada solo con entrenamiento (4.10.2). Jerarquía de 3 niveles con trazabilidad explícita de la fuente (`FuenteAlineacion`) y marca de imputación; `build_phase_windows` respeta el horizonte de corte (4.10.4) truncando o excluyendo fases posteriores |
| Construcción de tensores | `src/features/tensor_builder.py` (implementado) | Ninguna transformación aprendida en esta etapa (solo reestructuración). `build_observation_tensor` exige una sola provincia-campaña ya filtrada y falla explícitamente ante ambigüedad; celdas puntuales faltantes quedan NaN (no error), variables ausentes por completo sí fallan (`ObservacionIncompletaError`). `build_variable_dictionary` valida el rol de cada entrada (secuencial/estatica/dinamica) |
| Split temporal | `src/evaluation/splits.py` (implementado) | Origen móvil (`expanding_window_splits`, confirmatorio principal): entrenamiento y tuning usan solo campañas anteriores a la campaña externa evaluada. LOCO (`leave_one_campaign_out_splits`, complementario) y LODO (`leave_one_department_out_splits`, robustez espacial) también implementados, con exclusión completa del bloque evaluado en todos los casos (4.12.1) |
| Estandarización/imputación | `src/features/scalers.py` | Medias/desviaciones e imputación estimadas solo con el conjunto de entrenamiento de cada pliegue, aplicadas sin recalcular al conjunto externo (4.12.2) |
| Línea base (B2) | `src/models/benchmarks.py` | Tendencia ajustada exclusivamente con campañas de entrenamiento (4.11) |
| Tratamiento de la variable objetivo (RF13) | `src/features/target_transform.py` (implementado) | Ambas representaciones (nivel y anomalía) se derivan de la línea base ajustada solo con entrenamiento; las métricas confirmatorias siempre se reportan sobre el rendimiento reconstruido en kg/ha (4.11). `fit_provincial_baseline` usa regla de respaldo (media simple) si la historia de una provincia tiene menos de `HISTORIA_MINIMA_CAMPANAS` observaciones |

## 5. Plan de modelado y evaluación

1. **Entrenamiento**: los 5 modelos ajustables (CNN-1D, Elastic-Net, Random Forest, XGBoost, SVR) reciben el mismo número máximo de evaluaciones de Optuna (presupuesto equiparado, 4.12.3); los 3 benchmarks (B1, B2, B3) se calculan sin optimización.
2. **Validación**: esquema principal de origen móvil sobre las campañas externas; leave-one-campaign-out como análisis complementario; leave-one-department-out como análisis de robustez espacial (4.12.1).
3. **Replicación estocástica**: CNN-1D, Random Forest y XGBoost se reentrenan con 10 semillas tras fijar la configuración por pliegue; Elastic-Net y SVR se ejecutan deterministamente cuando la implementación lo permite (4.12.4).
4. **Agregación**: se agrega por campaña (mediana entre semillas) antes de cualquier contraste estadístico, para no inflar artificialmente el número de unidades independientes (3.3.3).
5. **Contrastes confirmatorios**: HE1a (no inferioridad frente a XGBoost, margen δ=0.05, intervalo unilateral 95%, Wilcoxon de apoyo) y HE1b (superioridad frente a B2) — sección 3.3.1-3.3.2, 4.14.

**Implementado — métricas y contrastes** (`src/evaluation/metrics.py`, `src/evaluation/hypothesis_tests.py`):
- `metrics.py`: RMSE, MAE, rRMSE (falla explícitamente si la media observada está cerca de cero, en vez de devolver un porcentaje engañoso) y R² fuera de muestra (verificado que puede ser negativo, sección 2.4.10).
- `hypothesis_tests.py`: `evaluate_non_inferiority()` (HE1a) y `evaluate_superiority()` (HE1b, frente a B2) comparten la misma mecánica — límite superior unilateral del IC 95% de la mediana vía bootstrap sobre las campañas externas (remuestreo por bloques temporales), Wilcoxon pareado unilateral de apoyo (`scipy.stats.wilcoxon`), estimador de Hodges-Lehmann (mediana de promedios de Walsh) y correlación biserial de rangos pareada. Ambas funciones exigen arreglos de RMSE por campaña ya emparejados y de igual longitud, fallando explícitamente si no lo están.

**Implementado** (`src/models/`): arquitectura CNN-1D en PyTorch:
- `cnn1d.py` — `QuinuaYieldCNN1D`: 2 capas Conv1d + BatchNorm1d + Dropout, `AdaptiveAvgPool1d` (permite T variable entre horizontes sin cambiar la red), covariables concatenadas después del bloque convolucional (4.10.3), inicialización He Normal. `CNN1DConfig` fija los valores por defecto dentro del rango parsimonioso de Sabo et al. (2023, ver §1.1); menos de 5000 parámetros totales verificado por test.
- `cnn1d_trainer.py` — `train_cnn1d`: Adam + `weight_decay` (regularización L2), early stopping sobre partición interna de validación, semilla explícita que re-inicializa **tanto los pesos como el split de validación interna** (necesario para reproducibilidad real entre llamadas — un bug de este tipo se encontró y corrigió durante TDD: fijar la semilla solo dentro del bucle de entrenamiento no bastaba, porque los pesos ya se habían inicializado antes con el generador global en un estado distinto entre corridas).

**Nota de arquitectura (2026)**: la decisión original de framework (TensorFlow/Keras, §1) se revisó a **PyTorch** al construir el código: TensorFlow no tiene build disponible para la versión de Python del entorno de desarrollo (3.14). PyTorch es igualmente válido para CNN-1D + SHAP (precedente: Joshi et al. 2025, `shap.DeepExplainer`) y no requirió ningún otro cambio de diseño.

**Implementado — modelos comparadores** (`src/models/classic_models.py`, `src/features/flatten.py`): los 4 modelos clásicos de la Tabla 8 (Elastic-Net, Random Forest, XGBoost, SVR) comparten una interfaz uniforme `ClassicModel` (`fit`/`predict`) que envuelve directamente sus equivalentes de scikit-learn/XGBoost — necesaria para que el módulo de tuning (Optuna) los trate de forma equiparada sin conocer la librería subyacente de cada uno. Reciben el mismo tensor V×T que la CNN-1D, aplanado a vector tabular por `flatten_tensor` (orden variable→fase, covariables al final; NaN preservados, la imputación es un paso posterior). Random Forest y XGBoost son estocásticos y reciben semilla explícita (verificado: misma semilla → misma predicción, semillas distintas → predicciones distintas); Elastic-Net y SVR son deterministas y la ignoran (sección 4.12.4).

**Implementado — benchmark B3** (`src/models/benchmarks.py`): regresión lineal simple `ndvi_max → rendimiento_kg_ha`, ajustada exclusivamente con entrenamiento (misma disciplina que B2). Junto con B1 y B2 (`src/features/target_transform.py`), completa el trío no ajustable de la Tabla 8 — nunca pasa por el módulo de tuning.

**Implementado — tuning con Optuna** (`src/models/tuning.py`, `src/models/tuning_cnn1d.py`): `tune_classic_model()` ejecuta TPE (`optuna.samplers.TPESampler`) sobre un espacio de búsqueda declarado como `{hiperparametro: (tipo, min, max)}`, evaluando cada trial por RMSE en una partición de validación interna (siempre dentro del conjunto de entrenamiento del pliegue). `tune_cnn1d()` sigue el mismo contrato (mismo `TuningResult`, mismo criterio de RMSE) reutilizando `train_cnn1d` por trial sobre tensores 3D, separando internamente hiperparámetros de arquitectura (`CNN1DConfig`) de los de entrenamiento (`TrainingConfig`) a partir de un único espacio de búsqueda plano. `PRESUPUESTO_TRIALS_POR_MODELO` fija explícitamente el mismo número de trials para los 5 modelos ajustables — el valor concreto (30, por ahora) se confirma en el ensayo piloto H2 según el presupuesto de cómputo real tras H1; lo que la sección 4.12.3 exige es la igualdad entre modelos, no un número específico. `sugerir_hiperparametro()` es la única función que traduce el espacio de búsqueda declarado a llamadas de Optuna, compartida entre ambos módulos de tuning para no duplicar esa lógica.

## 6. Plan de explicabilidad

Declarar **antes** de codificar la evaluación definitiva (registro de preespecificación H2, sección 4.15):
- Variante de SHAP: `DeepExplainer` (principal) / `KernelExplainer` (contingencia) — ver §1.1.
- Conjunto de referencia/background: extraído solo de campañas de entrenamiento.
- Manejo de predictores correlacionados: declarar el supuesto de dependencia adoptado (Aas et al., 2021).
- Unidad de atribución: predictor × paso temporal, agregada por fase fenológica; semillas agregadas por campaña antes de evaluar estabilidad (3.3.3).
- Evaluación de calidad de explicación: estabilidad (W de Kendall ≥ 0.70, Jaccard ≥ 0.50), fidelidad por oclusión (sin reentrenar, con control de igual duración/baja contribución), concordancia agronómica (fase de máxima contribución ∈ {floración, llenado de grano} en el horizonte principal).

**Implementado** (`src/explainability/`):
- `shap_explainer.py` — `explain_with_deep_explainer()` genera valores SHAP con `shap.DeepExplainer` sobre múltiples entradas (tensor secuencial + covariables) simultáneamente, devolviendo ambos por separado en la forma original de cada tensor. `validate_local_accuracy()` implementa la validación explícita de la propiedad fundamental de SHAP (Lundberg y Lee, 2017): la suma de valores SHAP + valor base debe reproducir la predicción real, dentro de tolerancia — es la condición operacional para decidir si `DeepExplainer` es fiable en el ensayo piloto (H2) o si debe activarse la contingencia a `KernelExplainer` (aún no implementada, solo el punto de decisión). **Verificado empíricamente con datos sintéticos** que `DeepExplainer` cumple local accuracy en la arquitectura `QuinuaYieldCNN1D` — no elimina el riesgo documentado con datos reales, pero da evidencia favorable temprana.
- `shap_aggregation.py` — `aggregate_shap_by_phase()` agrega valores SHAP por **contribución absoluta** (no suma con signo, para no cancelar pasos de signo opuesto) desde predictor×paso temporal hacia predictor×fase; `fase_de_maxima_contribucion()` identifica la "ventana de contribución predictiva" (sección 2.4.7).
- `stability.py` (HE2a) — `kendalls_w()` implementa el coeficiente de concordancia de Kendall (no disponible directamente en `scipy`, que solo ofrece `kendalltau` para pares) vía la fórmula estándar W = 12·S/(m²·(n³-n)) sobre rangos por campaña; `jaccard_index()` y `evaluate_stability()` combinan ambos umbrales (W ≥ 0.70 Y Jaccard promedio ≥ 0.50, sección 3.3.3) en un solo veredicto.
- `fidelity.py` (HE2b) — `occlude_phase()` reemplaza pasos temporales por el valor de referencia sin mutar el tensor original; `evaluate_fidelity_by_occlusion()` compara el incremento de RMSE de la ventana de alta contribución contra una de control de igual duración, **verificado con test que los pesos del modelo no cambian** (sin reentrenar, sección 3.3.4) y con un modelo determinista de juguete que confirma que el criterio detecta correctamente cuál ventana es realmente más influyente.
- `agronomic_concordance.py` (HE2c) — `evaluate_agronomic_concordance()` compara la fase de máxima contribución contra `FASES_CRITICAS_PREESPECIFICADAS` ({floración, llenado_grano}), restringiendo el conjunto válido a solo {floración} en el horizonte temprano (sección 3.3.5, ya que llenado_grano no es observable en ese punto de corte).

## 7. Reproducibilidad

- **Manifiesto de datos** (`data/manifest/`): fuente, versión, URL/identificador, fecha de descarga, hash, licencia, cobertura, regla de transformación — por cada archivo (4.18, punto 2).
- **Entorno**: `pyproject.toml` + lockfile; documentar versión de Python, PyTorch, scikit-learn, XGBoost, shap, Optuna y hardware/aceleración usada.
- **Preespecificación (H2)**: registro fechado de hipótesis, márgenes de no inferioridad, comparador confirmatorio, umbrales de HE2, horizontes, espacios de búsqueda de Optuna, reglas de depuración — antes de la evaluación definitiva.
- **Resultados intermedios**: predicciones, métricas y atribuciones SHAP por campaña, modelo y semilla, preservando la separación entrenamiento/prueba (4.18, punto 5).
- **Publicación**: si una licencia impide redistribuir datos crudos, se publican scripts y metadatos suficientes para reconstruirlos desde la fuente oficial (4.18, punto 6).

## 8. `dvc.yaml` real (DAG de orquestación) — implementado

DVC inicializado (`dvc init --subdir` dentro de `service-cnn/`, integrado con git). 5 stages reales, cada uno respaldado por un script CLI en `scripts/` que orquesta las funciones ya construidas en `src/`:

```yaml
stages:
  audit:        # scripts/run_audit.py — EJECUTABLE hoy (verificado con dvc repro)
  ingest:       # scripts/run_ingestion.py — verifica prerrequisitos, falla explícito
  preprocess:   # scripts/run_preprocessing.py — depende de la salida de ingest
  train_eval:   # scripts/run_experiment.py — depende de la salida de preprocess
  explain:      # scripts/run_shap.py — depende de la salida de train_eval
```

`params.yaml` declara `gee_project_id` (interpolado en el comando de `ingest` vía `${gee_project_id}`), con el placeholder `"PENDIENTE-CREAR-CUENTA-GEE"` hasta que exista la cuenta real de Google Earth Engine.

**Estado de ejecución real, verificado con `dvc repro`**:
- `audit` — **se ejecuta de punta a punta** con los placeholders vacíos de `configs/provincias.csv` y `configs/produccion_documentada.csv` (produce N0=N1=N2=N3=0, correcto dado que esos CSV están vacíos; N4/N5 quedan `PENDIENTE`). Generó `dvc.lock` real.
- `ingest` — falla explícitamente (`exit 1`) porque `configs/column_mapping.yaml` aún tiene nombres de columna placeholder no confirmados contra el archivo real de MIDAGRI, y no hay cuenta de GEE. El mensaje de error indica exactamente qué falta.
- `preprocess`, `train_eval`, `explain` — cada uno verifica que la salida de la etapa anterior contenga datos reales (no solo `.gitkeep`) antes de intentar ejecutar la orquestación real (aún no implementada, ya que depende de datos que no existen); fallan con `NotImplementedError` explícito señalando qué componentes de `src/` ya están listos para conectarse.

Cada `stage` se re-ejecuta solo si cambian sus dependencias declaradas (código o datos), dando trazabilidad automática de qué resultado corresponde a qué versión (RNF2). `dvc dag` confirma que el grafo es una cadena lineal sin ciclos: `audit → ingest → preprocess → train_eval → explain`.

**Pendiente para ejecución completa** (no es código adicional, son datos/credenciales):
1. `configs/provincias.csv` y `configs/produccion_documentada.csv` con la clasificación territorial oficial (Tabla 3-4, sección 4.5.3).
2. `configs/column_mapping.yaml` con los nombres de columna reales del archivo MIDAGRI/SIEA.
3. Cuenta de Google Earth Engine + `params.yaml` con el `gee_project_id` real.
4. La orquestación real dentro de `run_preprocessing.py`, `run_experiment.py`, `run_shap.py` (llamar en el orden correcto a las funciones ya implementadas y probadas en `src/`) — trabajo de "cableado", no de diseño nuevo.

## 9. Roadmap de implementación (alineado al cronograma)

| Fase de implementación | Actividad(es) del cronograma | Entregable |
|---|---|---|
| 1. Auditoría de datos (`src/ingestion/audit.py`) — RF1, RF11 | Actividad 1 | H1 · E1 |
| 2. Ingesta + máscara agrícola | Actividades 2-3 | E1b, máscara validada |
| 3. Alineación fenológica + tensor V×T | Actividad 4 | E2 |
| 4. Preprocesamiento y análisis exploratorio | Actividad 5 | E3 |
| 5. Implementación CNN-1D + comparadores + SHAP | Actividad 6 | E4 |
| 6. Ensayo piloto + preespecificación (valida elección de explainer SHAP, ajusta arquitectura a N5 real) | Actividad 7 | H2 · E5 |
| 7. Evaluación predictiva definitiva | Actividad 8 | Resultados por campaña/semilla |
| 8. Evaluación SHAP definitiva | Actividad 9 | E6 |
| 9. Redacción de resultados | Actividad 10 | E7 |
| 10. Publicación de artículo y repositorio | Actividad 11 | E8 |
| 11. Sustentación | Actividad 12 | H3 |

**Siguiente paso real de implementación**: construir `src/ingestion/audit.py` para ejecutar la Actividad 1 (auditoría de disponibilidad de datos, Hito H1) — es el único paso que no depende de decisiones aún pendientes de esa misma auditoría.
