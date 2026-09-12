# Análisis del Sistema de Información

**Proyecto:** Modelo interpretable con CNN-1D y SHAP para predecir el rendimiento de quinua en la sierra altoandina del Perú
**Fuente:** `docs/Proyecto_Tesis_Maestria_UNMSM_WMSR.docx` (Proyecto de Tesis, UNMSM-FISI) y `docs/Cronograma_Proyecto_Tesis_UNMSM_WMSAR.xlsx`
**Autor del análisis:** derivado 1:1 del proyecto de tesis; cada requisito cita su sección/tabla de origen.

## 0. Naturaleza del sistema

Este no es un sistema transaccional de negocio. Es un **pipeline de datos + experimentación científica reproducible**, construido bajo el paradigma de Ciencias del Diseño (Design Science Research Methodology, DSRM; Peffers et al., 2007 — Tabla 1 de la tesis, sección 4.2). El "sistema de información" comprende:

- La adquisición, armonización y transformación de datos agroclimáticos y espectrales heterogéneos.
- La construcción de un artefacto computacional (CNN-1D + SHAP) evaluado bajo un protocolo experimental estadístico preespecificado.
- La generación de entregables documentales trazables (E1-E8) exigidos por el cronograma.

Cualquier decisión de diseño debe ser trazable a una sección de la tesis. No se introducen requisitos no sustentados en el documento fuente.

## 1. Propósito y alcance del sistema

**Problema general** (sección 1.2.1): ¿en qué medida un modelo interpretable con CNN-1D y SHAP permite predecir el rendimiento de quinua en la sierra altoandina del Perú?

**Objetivo general** (sección 1.4.1): determinar en qué medida dicho modelo permite predecir el rendimiento de quinua, evaluando conjuntamente exactitud predictiva (OE1) e interpretabilidad (OE2).

**Delimitación del sistema** (sección 1.5):

| Dimensión | Delimitación |
|---|---|
| Espacial | Provincias con producción registrada de quinua en 8 departamentos altoandinos (Puno, Ayacucho, Apurímac, Arequipa, Junín, Cusco, La Libertad, Huancavelica); 67 provincias preliminares de 80 totales (Tabla 3) |
| Temporal | Campañas agrícolas 2005/2006 a 2023/2024 (año de cosecha 2006-2024), 19 campañas. **Enmendado** (ver `service-cnn/data/manifest/midagri_sisagri.yaml`, sección `enmienda_metodologica`, propuesta 2026-09-12, pendiente de aprobación del asesor): la fuente tabular vigente de rendimiento (MIDAGRI/SIEA) solo cubre campañas completas 2016-2025 (10 campañas) — no existe fuente oficial tabular para 2006-2015 |
| Temática | Predicción de rendimiento en kg/ha y atribución de importancia predictiva por variable/fase fenológica. Fuera de alcance: superficie sembrada, precios, evaluación económica |
| Metodológica | Solo fuentes públicas y de libre acceso; 5 familias de modelos + 3 benchmarks (Tabla 8); sin recolección de datos primarios en campo |
| Horizonte de pronóstico | 30 días antes de la cosecha esperada (principal); cierre de floración (secundario) |

El sistema **no** es de uso operativo continuo: es una plataforma de investigación de ciclo único (una ejecución definitiva tras el ensayo piloto), con entregables documentales como salida principal, no un servicio en producción.

## 2. Actores e interesados

| Actor | Rol | Sección de origen |
|---|---|---|
| Tesista (investigador) | Único operador del sistema: diseña, construye, ejecuta y documenta todo el pipeline | Implícito en todo el Cap. 4; Tabla 11 (presupuesto, "trabajo del investigador") |
| Asesor | Revisa decisiones metodológicas y valida entregables antes de cada hito | Actividad 12 del cronograma; portada de la tesis |
| MIDAGRI / SIEA (Sistema Integrado de Estadística Agraria) | Fuente de rendimiento, producción, superficie cosechada y siembra mensual | Sección 4.9, Tabla 7 |
| INEI / SIRTOD | Fuente de contraste para verificación cruzada de estadística agraria | Sección 4.5.2 |
| SENAMHI | Fuente de validación/sesgo para precipitación y temperatura | Tabla 7 |
| CHIRPS v2.0 (vía Google Earth Engine o descarga directa) | Fuente principal de precipitación | Tabla 7 |
| ERA5-Land (Copernicus/CDS) | Fuente principal de temperatura y radiación solar | Tabla 7 |
| MODIS MOD13Q1 V061 (NASA, vía Google Earth Engine) | Fuente de NDVI | Tabla 7 |
| Gobiernos regionales y sector agrario (usuarios finales indirectos) | No interactúan con el sistema directamente; consumen los hallazgos publicados (E7, E8) para planificación agraria | Sección 1.3.2 (justificación práctica) |

No existen actores con acceso concurrente, roles de autorización granular, ni interfaz de usuario multiusuario: es correcto que el sistema no contemple gestión de usuarios, sesiones ni permisos.

## 3. Requisitos funcionales

Derivados de las Tablas 1, 2, 6, 7, 8, 12 y secciones 4.5-4.15.

| ID | Requisito | Sección de origen |
|---|---|---|
| RF1 | Auditar y cuantificar la cobertura provincia×campaña disponible en las fuentes oficiales, produciendo el conteo escalonado N0-N5 (Tabla 5) | 4.5.2, 4.5.6, Tabla 5 |
| RF2 | Ingerir y armonizar 5 fuentes de datos con resoluciones espaciales/temporales heterogéneas (provincia/campaña, 0.05°/diaria, 0.1°/horaria, 250m/16 días) a una estructura común provincia-campaña-paso temporal | 4.9, Tabla 7 |
| RF3 | Construir una máscara de superficie agrícola (capa de cobertura de suelo + MDE + fenometría NDVI) para excluir coberturas no cultivadas del cálculo de NDVI | 4.10.1 |
| RF4 | Alinear fenológicamente las series temporales mediante jerarquía de 3 niveles (fecha modal de siembra → fenometría NDVI → fecha modal departamental de respaldo) | 4.10.2 |
| RF5 | Construir, por observación provincia-campaña, un tensor V×T (predictores secuenciales × pasos temporales) más covariables estáticas (altitud) y dinámicas (% superficie sembrada con quinua) | 4.10.3 |
| RF6 | Entrenar 5 modelos ajustables (CNN-1D, Elastic-Net, Random Forest, XGBoost, SVR) bajo presupuesto de tuning de hiperparámetros equiparado, más 3 benchmarks no ajustables (B1 media histórica, B2 línea base con tendencia, B3 regresión con NDVI máximo) | 4.12.3, Tabla 8 |
| RF7 | Validar temporalmente con origen móvil (esquema confirmatorio principal), leave-one-campaign-out (complementario) y leave-one-department-out (robustez espacial), bajo un diseño de bloques completos pareados donde cada campaña externa (bloque) recibe los 8 tratamientos (5 modelos + 3 benchmarks) bajo idénticas condiciones, replicando modelos estocásticos con 10 semillas | 4.3, 4.12.1, 4.12.4 |
| RF8 | Generar explicaciones SHAP post-entrenamiento únicamente sobre observaciones externas de prueba, con variante, conjunto de referencia y manejo de correlación declarados antes de la evaluación definitiva | 4.15 |
| RF9 | Evaluar la calidad de las explicaciones mediante estabilidad (W de Kendall, índice de Jaccard de fases), fidelidad por oclusión (sin reentrenar) y concordancia agronómica con fases preespecificadas | 3.3.3-3.3.5, 4.15 |
| RF10 | Registrar la preespecificación fechada (hipótesis, márgenes, umbrales, espacios de búsqueda) antes de la evaluación definitiva (Hito H2), y mantener trazabilidad completa (manifiesto de datos, semillas, versiones, código) | 4.18, Actividad 7 del cronograma |
| RF11 | Aplicar y documentar todos los criterios de elegibilidad/exclusión de forma preespecificada, sin usar información de campañas externas para depurar datos | 4.7 |
| RF12 | Calcular y reportar métricas de exactitud (RMSE, MAE, rRMSE, R²) desagregadas por campaña, departamento, horizonte y modelo, y ejecutar los contrastes estadísticos confirmatorios (no inferioridad HE1a/HE1b, estabilidad/fidelidad/concordancia HE2a-c) | 3.3, 4.13, 4.14, Tabla 9 |
| RF13 | Tratar la variable objetivo en dos representaciones (nivel en kg/ha y anomalía respecto de una línea base provincial ajustada solo con entrenamiento), reportando las métricas confirmatorias sobre el rendimiento reconstruido y las de anomalía como análisis complementario | 4.11 |

## 4. Requisitos no funcionales

| ID | Requisito | Sección de origen |
|---|---|---|
| RNF1 | **Anti-fuga de información temporal**: todo parámetro aprendido (medias, desviaciones, imputación, línea base, alineación fenológica, umbrales) se estima solo con datos de entrenamiento de cada pliegue | 4.12.2 |
| RNF2 | **Reproducibilidad**: repositorio de código versionado, manifiesto de datos (fuente/versión/hash/licencia/fecha de descarga), entorno documentado (versiones Python/librerías), semillas registradas | 4.18 |
| RNF3 | **Presupuesto computacional acotado**: bolsa de crédito de cómputo en la nube de S/1500 (ajustable tras el piloto); sin infraestructura dedicada de producción | Tabla 11 |
| RNF4 | **Licencias de datos**: respetar términos de MIDAGRI, INEI, Copernicus, CHIRPS, MODIS, SENAMHI; publicar datos derivados solo si la licencia lo permite (si no, publicar scripts de reconstrucción) | 4.17, 4.18 |
| RNF5 | **Sin datos personales**: la investigación no trata datos de personas naturales identificables; no requiere consentimiento informado, aunque se verificará la necesidad de exención institucional formal | 4.17 |
| RNF6 | **Equidad de comparación algorítmica**: igual número de evaluaciones de hiperparámetros para los 5 modelos ajustables; benchmarks nunca optimizados | 4.12.3 |
| RNF7 | **Trazabilidad de decisiones**: toda modificación sustantiva de alcance, unidad de análisis o diseño confirmatorio requiere enmienda metodológica documentada antes del entrenamiento definitivo | 6.1 (plan de contingencia) |
| RNF8 | **Uso responsable de IA generativa**: declarar su uso como apoyo de redacción/código; el autor verifica toda salida incorporada; prohibido fabricar datos/resultados/referencias | 4.17 |
| RNF9 | **Gestión de riesgos de disponibilidad de datos**: activar respuestas preespecificadas (reducción de periodo, interpolación limitada, estratificación) ante cobertura insuficiente, sin alterar retroactivamente el diseño confirmatorio | 4.8, Tabla 6 |
| RNF10 | **Mitigación documentada de amenazas a la validez** (constructo, interna, externa, de conclusión): cada amenaza identificada en la tesis debe tener una estrategia de mitigación implementada y verificable en el pipeline, no solo declarada | 4.16, Tabla 10 |

## 5. Modelo de datos conceptual

**Entidad central:** `provincia_campaña` — unidad de análisis (sección 4.4), definida como el rendimiento oficial de una provincia en una campaña agrícola junto con sus series agroclimáticas y espectrales asociadas.

```
PROVINCIA_CAMPAÑA (unidad de análisis)
 ├── id: (provincia_id, campaña_id)                 [clave compuesta]
 ├── rendimiento_kg_ha: decimal                      [variable objetivo, MIDAGRI]
 ├── produccion_ton, superficie_cosechada_ha         [para verificación/reconstrucción]
 ├── altitud_media_agricola: decimal                 [covariable estática]
 ├── prop_superficie_quinua: decimal                 [covariable dinámica]
 ├── fecha_modal_siembra: date                       [derivada, jerarquía 4.10.2]
 └── SERIE_TEMPORAL (1:N, eje T = pasos hasta horizonte)
      ├── paso_temporal: int (día o composición de 16 días)
      ├── precipitacion_mm: decimal        (CHIRPS)
      ├── temp_max_C, temp_min_C: decimal  (ERA5-Land)
      ├── radiacion_solar: decimal         (ERA5-Land)
      ├── ndvi: decimal                    (MODIS, sobre máscara agrícola)
      └── fase_fenologica: enum {emergencia, desarrollo_vegetativo,
                                   floracion, llenado_grano, madurez}

PROVINCIA (dimensión territorial)
 ├── provincia_id, departamento, codigo_ubigeo
 └── clasificacion_territorial: enum {altoandina, transicion, excluida}

CAMPAÑA (dimensión temporal)
 ├── campaña_id (año de cosecha)
 └── periodo: [año_siembra, año_cosecha]

EJECUCION_EXPERIMENTAL (registro de resultados, no del dominio agrícola)
 ├── modelo: enum {CNN-1D, ElasticNet, RandomForest, XGBoost, SVR, B1, B2, B3}
 ├── pliegue_temporal, semilla, horizonte
 ├── metricas: {RMSE, MAE, rRMSE, R2}
 └── atribuciones_shap: (predictor, paso_temporal/fase) → valor SHAP
```

Relación clave: `PROVINCIA_CAMPAÑA` agrega N registros de `SERIE_TEMPORAL` (uno por paso temporal disponible hasta el horizonte de pronóstico), y cada ejecución del experimento genera M registros en `EJECUCION_EXPERIMENTAL` (uno por modelo × pliegue × semilla).

## 6. Diagrama de contexto (nivel 0)

```
                    ┌─────────────────────────────────────┐
   MIDAGRI/SIEA ───▶│                                       │
   INEI/SIRTOD  ───▶│                                       │───▶ Matriz de cobertura
   SENAMHI      ───▶│   SISTEMA DE PREDICCIÓN DE           │      provincia×campaña (E1)
   CHIRPS       ───▶│   RENDIMIENTO DE QUINUA              │───▶ Dataset + diccionario
   ERA5-Land    ───▶│   (pipeline de datos +               │      de variables (E2)
   MODIS/GEE    ───▶│    experimentación reproducible)     │───▶ Modelo CNN-1D entrenado
                    │                                       │      + comparadores (E4)
                    │                                       │───▶ Informe de interpretabilidad
   Tesista      ◀──▶│                                       │      SHAP (E6)
   Asesor       ◀──▶│                                       │───▶ Manuscrito + artículo +
                    │                                       │      repositorio (E7, E8)
                    └─────────────────────────────────────┘
```

## 7. Diagrama de flujo de datos (nivel 1)

```
[1. Auditoría H1] ─▶ [2. Ingesta y armonización de 5 fuentes]
        │                          │
        ▼                          ▼
  N0..N5 (Tabla 5)         [3. Máscara agrícola]
                                    │
                                    ▼
                     [4. Alineación fenológica (jerarquía 3 niveles)]
                                    │
                                    ▼
                  [5. Construcción del tensor V×T + covariables]
                                    │
                                    ▼
                [6. Split temporal anti-fuga: origen móvil / LOCO / LODO]
                                    │
                        ┌───────────┴───────────┐
                        ▼                       ▼
        [7a. Tuning + entrenamiento    [7b. Cálculo de benchmarks
         5 modelos, 10 semillas]        B1/B2/B3 (no ajustables)]
                        │                       │
                        └───────────┬───────────┘
                                    ▼
                  [8. Evaluación predictiva: RMSE/MAE/rRMSE/R²,
                       contrastes HE1a/HE1b]
                                    │
                                    ▼
                  [9. Explicaciones SHAP sobre conjunto externo]
                                    │
                                    ▼
             [10. Evaluación de interpretabilidad: estabilidad,
                   fidelidad por oclusión, concordancia agronómica
                   (HE2a-c)]
                                    │
                                    ▼
                  [11. Reportes E1-E8 + manuscrito + repositorio]
```

## 8. Trazabilidad: Requisitos ↔ Actividad del cronograma ↔ Entregable

| Requisitos | Actividad (cronograma) | Hito/Entregable |
|---|---|---|
| RF1, RF11 | 1. Auditoría de cobertura, elegibilidad y tamaño del conjunto analítico | H1 · E1 |
| — | 2. Revisión y actualización de la literatura (continua) | E1b |
| RF3 | 3. Construcción de la máscara agrícola y descarga documentada de series | Producto intermedio: máscara validada |
| RF2, RF4, RF5 | 4. Integración del conjunto de datos y alineación fenológica | E2 |
| RNF1, RF13 | 5. Preprocesamiento, análisis exploratorio y tratamiento de la variable objetivo | E3 |
| RF6 | 6. Diseño e implementación de CNN-1D, comparadores e integración de SHAP | E4 |
| RF10, RNF6 | 7. Ensayo piloto, variabilidad estocástica y preespecificación | H2 · E5 |
| RF7, RF12 | 8. Evaluación predictiva: origen móvil, XGBoost, B2, robustez espacial, sensibilidad | Predicciones y métricas por campaña/semilla |
| RF8, RF9 | 9. Evaluación SHAP: estabilidad, oclusión, ablación secundaria, concordancia agronómica | E6 |
| — | 10. Redacción de resultados, discusión y conclusiones | E7 |
| RNF2, RNF4 | 11. Comunicación: artículo, repositorio, dataset según licencias | E8 |
| — | 12. Revisión del asesor, control de similitud, sustentación | H3 |

## 9. Fuera de alcance (explícito)

- Predicción a nivel de parcela o unidad productiva individual (falacia ecológica declarada en 1.6).
- Atribución causal de variables sobre el rendimiento (SHAP describe la función aprendida, no el mecanismo agronómico — sección 2.4.9, 2.4.7).
- Interfaz de usuario, API pública o servicio desplegado en producción continua.
- Recomendaciones prescriptivas a productores individuales (sección 4.17).
- Gestión de identidad/autenticación de usuarios (no hay datos personales ni acceso multiusuario).
