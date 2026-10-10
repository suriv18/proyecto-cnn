# F3 · Plan del ensayo piloto (actividad 7, hito H2)

**Estado:** BORRADOR · **Fecha:** 2026-10-10 · **Entregables que habilita:** E5 (preespecificación congelada), H2
**Documentos relacionados:** [F3-modelado](F3-modelado.md) · [E5 (borrador)](E5-preespecificacion-BORRADOR.md) · [E3](E3-informe-depuracion-eda.md) · `configs/preespecificacion.yaml`

## 1. Propósito

El piloto existe para **cerrar las decisiones que la tesis delega en él** (secciones 4.11, 4.12 y 4.15) y para **medir costo y variabilidad**, sin tocar nunca las campañas externas. Al terminar, la preespecificación E5 pasa de borrador a congelada y se alcanza el hito H2. El piloto **no** busca saber si la CNN-1D gana: ese resultado es el contraste confirmatorio y no puede condicionarse.

Decisiones que debe resolver (las 9 marcadas `por_decidir_piloto` en E5):

| # | Decisión | Parámetro en E5 |
|---|---|---|
| D1 | Representación de modelado: nivel o anomalía | `variable_objetivo.representacion_de_modelado` |
| D2 | Forma de la línea base B2 | `variable_objetivo.forma_de_la_linea_base_b2` |
| D3 | Partición del *early stopping* de la CNN-1D (brecha G1) | `modelos.entrenamiento_cnn1d.early_stopping_particion` |
| D4 | Restaurar la mejor época (brecha G2) | `modelos.entrenamiento_cnn1d.restaurar_mejor_epoca` |
| D5 | Épocas máximas | `modelos.entrenamiento_cnn1d.epocas_maximas` |
| D6 | Espacios de búsqueda definitivos | `modelos.espacios_de_busqueda` |
| D7 | Variante de SHAP (Deep o Kernel) | `interpretabilidad.variante_de_shap` |
| D8 | Escala de salida explicada | `interpretabilidad.escala_de_salida_explicada` |
| D9 | Tolerancia de precisión local de SHAP | `interpretabilidad.tolerancia_de_precision_local` |

Además informa, sin decidir, la **ventana inicial de entrenamiento** (`por_decidir_asesor`) y confirma el costo del presupuesto de 30 trials.

## 2. Regla de oro: las campañas externas no se tocan

> **Salvedad sobre la ventana (detectada al implementar la guarda, 2026-10-10).** Las campañas del piloto son las
> primeras `ventana_inicial` campañas, que es lo que el asesor todavía debe fijar. Con ventana 5 son 2016–2020; con
> ventana 4, la campaña 2020 pasaría a ser **externa**. Por eso: (a) el piloto **oficial** espera a que el asesor fije
> la ventana; (b) la **corrida técnica** de desarrollo (no vinculante) usa solo **2016–2019**, la menor ventana plausible,
> y no lee 2020 en ningún caso. Con esa ventana mínima el piloto técnico tiene un único pliego (prueba 2019).

- **Datos del piloto:** únicamente las campañas **2016–2020**: **302 observaciones elegibles** (60, 62, 60, 62 y 58 por campaña; 25 con alguna celda vacía que se imputa con la mediana de entrenamiento). Las campañas 2021–2025 no se cargan, no se grafican y no se usan para ninguna decisión. *(Actualización 2026-10-10: la recomendación vigente es ventana 5; la corrida técnica usa 2016–2019 por prudencia, la menor ventana plausible.)*
- **Control técnico (tarea T1, hecha: `src/evaluation/pilot_guard.py`):** una guarda que lanza `CampanaExternaError` si algún componente del piloto recibe una campaña fuera de las `ventana_inicial` primeras, y que registra en el informe del piloto las campañas efectivamente leídas. Sin esta guarda no se ejecuta el piloto sobre datos reales.
- **Salvedad:** el informe E3 ya muestra la serie de rendimiento por campaña con las 10 campañas (descriptiva, para describir los datos). Eso no influye en ninguna decisión del piloto, pero queda declarado.

## 3. Diseño

### 3.1 Pliegues del piloto

Origen móvil con **ventana inicial de 3 campañas** dentro de 2016–2020:

| Pliego | Entrenamiento | Prueba del piloto | Obs. de entrenamiento |
|---|---|---|---|
| P-A | 2016–2018 | 2019 (62 obs.) | 182 |
| P-B | 2016–2019 | 2020 (58 obs.) | 244 |

La ventana de 3 (no la de 5 del confirmatorio) es la mayor que deja al menos dos pliegues en 5 campañas. Las campañas de prueba del piloto son campañas de **entrenamiento** en el confirmatorio, por lo que usarlas para decidir no contamina la evaluación.

### 3.2 Validación interna (selección de hiperparámetros)

Origen móvil **dentro del entrenamiento** (4.12.2: orden temporal y solo campañas anteriores), con ventana inicial interna de 2 campañas: el pliego P-A tiene 1 pliego interno (valida 2018, 60 obs., entrenando con 2016–2017, 122 obs.) y P-B tiene 2 (valida 2018 con 122 obs. y 2019 con 182 obs.). Con tan pocos pliegues internos el RMSE de validación es ruidoso; el piloto lo documenta como limitación y no la corrige con campañas externas.

### 3.3 Preprocesamiento por pliego

Todo se ajusta **solo con el entrenamiento del pliego**: estandarización z-score por variable y fase; imputación de las celdas vacías residuales (30 de las 601 observaciones, 5 %; 25 de las 302 del piloto, 8 %) con la mediana de entrenamiento por variable y fase, para los cinco modelos (misma información, Tabla 8); línea base provincial B2 y respaldo fenológico departamental (que no se usa: el 100 % de las observaciones se alineó por el nivel primario).

## 4. Experimentos y reglas de decisión

Las reglas se **fijan antes de ejecutar** y se copian al informe del piloto. Ningún experimento usa campañas > 2020.

| ID | Experimento | Qué mide | Regla de decisión (preespecificada) |
|---|---|---|---|
| P0 | **Puerta de entrada** | Verifica la guarda T1, registra el hash de `data/processed` y que el gate F2 esté cerrado | Si falla, no se ejecuta nada más |
| P1 | **Prueba de fuga** | Entrena cada modelo con el rendimiento de entrenamiento **permutado** y evalúa en la campaña de prueba | **Rediseñada (2026-10-10):** pasa si la **mediana** del R² fuera de muestra de ≥ 10 permutaciones es ≤ 0,05 en los cinco modelos (se informa también el máximo). El máximo entre permutaciones daba falsas alarmas por azar: con 8 observaciones de prueba el R² sin información oscila entre −2,4 y +0,48 y con 62 llegó a +0,06 en una permutación. Control positivo: un «modelo» que ve la verdad de la prueba debe ser señalado. No detecta fugas entre entradas coherentes con las etiquetas de entrenamiento (limitación declarada) |
| P2 | **Referencias** | RMSE de B1, B2 y B3 por pliego (kg/ha) | Solo descriptivo: da la escala del error |
| P3 | **Costo** | Tiempo por *trial* de cada modelo (5 trials en P-B); proyección del confirmatorio | Si la proyección supera el presupuesto de cómputo disponible, se reducen las épocas máximas o el costo por trial; **nunca** el número de trials (se perdería la igualdad entre modelos) |
| P4 | **Representación (D1)** | RMSE en kg/ha reconstruido, nivel frente a anomalía, promediado en los pliegos P-A y P-B y en los cinco modelos | Se elige la representación con menor RMSE medio; si la diferencia relativa es < 2 %, se elige **nivel** (más simple) |
| P5 | **Línea base B2 y su historia mínima (D2)** | RMSE de B2 con (a) la media provincial, (b) tendencia lineal con un mínimo de 4 campañas y (c) la media de las 3 últimas campañas, en P-B (4 campañas de historia) | **Rediseñado (2026-10-10):** con menos de 5 campañas B2 cae a la media y coincide con B1, de modo que el piloto no puede evaluar la tendencia con la regla actual (verificado: B1 = B2 = 202 kg/ha con 3 campañas). Menor RMSE medio; **empate (< 2 %): media**, por ser la más estable con 4–5 puntos por provincia. El mínimo de historia queda registrado en E5 (`historia_minima_para_tendencia_campanas`, hoy 5) |
| P6 | **Variabilidad por semilla** | 10 semillas de CNN-1D, Random Forest y XGBoost en P-B: mediana, rango intercuartil | Informativo: se reporta el error estándar de la mediana; las 10 réplicas ya están fijadas por la tesis |
| P7 | **Entrenamiento de la CNN (D3–D5)** | (a) *early stopping* con la partición aleatoria actual frente a (b) la última campaña del entrenamiento con restauración de la mejor época | Se adopta (b) salvo que no converja; las épocas máximas se fijan en 1,5 × el percentil 95 de la mejor época observada, redondeado |
| P8 | **SHAP (D7–D9)** | Precisión local de `DeepExplainer` (suma de valores SHAP + valor base frente a la predicción) y acuerdo con `KernelExplainer` en 50 observaciones | La tolerancia se fija como el 5 % de la desviación estándar de la salida de entrenamiento; si algún caso la supera, se usa **KernelExplainer** en el confirmatorio. El acuerdo entre explicadores se reporta como correlación de Spearman de las contribuciones por fase |
| P9 | **Espacios de búsqueda (D6)** | Se revisa si el óptimo cae en el borde del espacio | Si el óptimo del 20 % de los trials se agrupa en un borde, se amplía ese borde **para los cinco modelos por igual**, sin favorecer a ninguno |
| P10 | **Curva de aprendizaje** (informa la ventana inicial) | RMSE de XGBoost en 2019 y 2020 entrenando con 1, 2, 3 y 4 campañas | Solo informa al asesor: ventana de 4 (6 campañas externas) frente a 5 (5 campañas externas) |
| P11 | **Duración de las fases** (opcional) | Compara las duraciones del INIA con la fenometría NDVI de las campañas de entrenamiento | Solo informa; no cambia la alineación primaria (siembra mensual de MIDAGRI) |

## 5. Brechas del código que el piloto debe cerrar

Se detectaron al preparar E5; las dos primeras afectan a la validez de la CNN-1D.

| ID | Brecha | Dónde | Acción |
|---|---|---|---|
| G1 | El *early stopping* usa una **partición aleatoria del 20 %** de las observaciones de entrenamiento; la sección 4.12.2 exige validación temporal y solo con campañas anteriores. Una partición aleatoria mezcla provincias de la misma campaña (correlacionadas) y detiene el entrenamiento demasiado tarde | `src/models/cnn1d_trainer.py` | Particionar por campaña (la última del entrenamiento) |
| G2 | El entrenamiento **devuelve el modelo de la última época**, no el de mejor validación | ídem | Restaurar los pesos de la mejor época |
| G3 | Entrenamiento de lote completo, sin estandarización interna: las entradas tienen escalas de 10⁻¹ a 10⁷ | ídem; `run_experiment` | La estandarización por pliego (3.3) se aplica fuera, antes de entrenar |
| G4 | `run_experiment.py` y `run_shap.py` son marcadores de posición | `scripts/` | Implementar (tareas T4–T7) |
| G5 | El presupuesto de 30 trials y los espacios de búsqueda son valores de partida sin confirmar | `src/models/tuning.py`, `configs/hyperparams.yaml` | P3 y P9 |

## 6. Tareas de implementación (TDD; sobre datos sintéticos hasta cerrar el gate)

Según la regla del tablero, no se codifica una fase hasta cerrar la anterior salvo decisión explícita. **Decisión registrada (2026-10-10):** se prepara y prueba el código del piloto con datos sintéticos; **no se ejecuta sobre datos reales** hasta que el asesor cierre el gate de F2.

| ID | Tarea | Tamaño | Depende de |
|---|---|---|---|
| T1 | Guarda de campañas del piloto (`pilot_campaigns`, `assert_only_allowed_campaigns`) — **hecha** | S | — |
| T2 | Constructor de pliegos desde `data/processed` (origen móvil, preprocesamiento ajustado solo con el entrenamiento, imputación residual) — **hecha** (`src/evaluation/fold_builder.py`: pliegos externos e internos, `load_pilot_dataset` que descarta las campañas no permitidas al leer; 21 pruebas) | M | T1 |
| T3 | Cerrar G1 y G2 en `train_cnn1d` (partición temporal, restaurar mejor época) — **hecha** (`grupos_temporales`, `restaurar_mejor_epoca`; 8 pruebas) | M | — |
| T4 | Ejecutor por pliego: ajuste de los 5 modelos con presupuesto equiparado, validación interna de origen móvil y semillas — **hecha** (`src/evaluation/experiment_runner.py`: adaptadores de los 5 modelos, `tune_model`, `benchmark_predictions` B1–B3, `run_fold`, `rmse_per_campaign`; 26 pruebas) | L | T2, T3 |
| T5 | Agregación por campaña y conexión con `evaluate_non_inferiority` / `evaluate_superiority` — **hecha** (`src/evaluation/he1_evaluation.py`; además se corrigió el contraste de apoyo de HE1a, que comparaba con cero en vez de con el margen; 10 pruebas) | M | T4 |
| T6 | Paso de SHAP con verificación de precisión local y contingencia a KernelExplainer — **hecha** (`src/explainability/shap_step.py`: fondo reproducible solo de entrenamiento, `explain_with_contingency`, `KernelExplainer` con `l1_reg=0`; 11 pruebas) | M | T4 |
| T7 | `scripts/run_pilot.py` (P0–P10) y plantilla del informe del piloto — **hecha** (`src/evaluation/pilot_runner.py`: orquesta P0–P10 con guarda de campañas, modos técnico y oficial, y genera resultados e informe; P11 queda opcional; 16 pruebas de integración) | M | T1–T6 |
| T8 | Prueba de fuga por permutación (P1) como prueba automática — **hecha** (`src/evaluation/leak_test.py`: `permuted_training_fold`, `run_permutation_test` con mediana y control positivo; 8 pruebas, verificadas contra el código anterior) | S | T4 |

## 7. Criterios de salida (hito H2)

- [ ] Gate F2 cerrado: el asesor aprueba las 8 decisiones `por_decidir_asesor` de E5.
- [ ] Las 9 decisiones `por_decidir_piloto` resueltas con las reglas de la sección 4.
- [ ] Prueba de fuga (P1) superada y guarda de campañas (P0) registrada en el informe.
- [ ] E5 sin ningún parámetro `propuesto` ni `por_decidir_*` (`assert_can_freeze` pasa) y **congelada** con fecha, *commit* y *sha256* del conjunto de datos.
- [ ] Informe del piloto (`docs/crisp/F3-informe-del-piloto.md`) con costo proyectado, variabilidad por semilla y la decisión de SHAP.
- [ ] Suite de pruebas en verde y `dvc repro` hasta el modelado reproducible.

## 8. Riesgos del piloto

| Riesgo | Respuesta |
|---|---|
| Pocos pliegues de validación interna → RMSE ruidoso y elección inestable de hiperparámetros | Se documenta; se prefiere la regla más simple ante empates; no se compensa con campañas externas |
| Cinco campañas de entrenamiento dan un piloto pequeño (2 pliegues) | Las decisiones D1 y D2 usan reglas con empate hacia lo simple; los resultados del piloto no se presentan como evidencia de la tesis |
| Tentación de afinar con el resultado de las campañas externas | La guarda T1 lo impide técnicamente y E5 congelada exige enmienda fechada para cualquier cambio |
| Costo del confirmatorio mayor al disponible | P3 lo mide antes; se reduce el costo por trial, no el número de trials |

## 9. Qué NO se hace en el piloto

No se cargan campañas posteriores a 2020; no se contrastan HE1/HE2; no se ajustan los umbrales de HE2 ni el margen δ; no se elige el modelo ganador; no se modifica la máscara agrícola.

## 10. Resultados de la corrida técnica (2026-10-10, **no vinculantes**)

Corrida de desarrollo sobre datos reales **solo de 2016–2019** (la menor ventana plausible, §2): un pliego (prueba 2019, 182 obs. de entrenamiento, 62 de prueba) y un pliego interno (valida 2018 con 122 obs.). No fija ningún parámetro de E5; el piloto oficial se repite cuando el asesor cierre el gate.

**Costo (P3).** 30 trials para los 5 modelos y 10 semillas en los 3 estocásticos proyectan **≈ 0,5 h** para todo el confirmatorio (5 pliegos externos, una representación): CNN-1D ≈ 13 min, Random Forest ≈ 11 min, XGBoost ≈ 3 min, Elastic-Net y SVR < 1 min. El presupuesto de 30 trials **no necesita recortarse** y hay margen para ampliarlo si el piloto lo justifica (siempre igual para los 5 modelos).

**Adelanto de P4 (un solo pliego; ilustrativo).** RMSE en kg/ha sobre la campaña 2019 (desviación estándar del rendimiento de prueba: 446):

| Modelo | Nivel | Anomalía |
|---|---|---|
| B1 / B2 (media provincial) | 202 | 202 |
| SVR | 322 | 203 |
| Random Forest | 389 | 202 |
| Elastic-Net | 395 | 202 |
| XGBoost | 425 | 213 |
| CNN-1D | 450 | 199 |
| B3 (regresión con NDVI máximo) | 466 | 466 |

Lectura prudente: (i) **en nivel todos los modelos son peores que la línea base**, porque ninguno recibe la identidad de la provincia y el 90 % de la varianza es entre provincias; (ii) **en anomalía los modelos quedan prácticamente en B2** (199–213), coherente con las correlaciones débiles dentro de la provincia del E3 (|r| ≤ 0,28); la regla de P4 elegiría la anomalía; (iii) B1 y B2 coinciden por la regla de historia mínima (P5 rediseñado); (iv) con un solo pliego **no** puede concluirse nada sobre HE1; estos números solo dimensionan la dificultad: superar a B2 exigirá una señal interanual que hoy es modesta.

**Adelanto de P8 (SHAP; mismo pliego técnico, 30 observaciones de la campaña 2019, fondo de 100 obs. de entrenamiento; no vinculante).**
- `DeepExplainer` cumple la **precisión local** con error ≈ 0 (la tolerancia propuesta, 5 % de la desviación de la salida, es 0,016): la contingencia a `KernelExplainer` no se activa y D7 apunta a `deep_explainer`. Tarda 0,4 s frente a 4 s del Kernel.
- Con el **mismo fondo**, `KernelExplainer` por defecto daba atribuciones 2,2 veces menores (correlación con Deep 0,87) por la regularización L1 automática de `shap`; con `l1_reg=0` la proporción baja a 1,24 y la correlación sube a **0,95**. Por eso la contingencia se implementó con `l1_reg=0` (registrado en E5).
- La contribución por fase es **casi plana** (0,10–0,15 con Deep) con máximo en floración; el acuerdo por fase entre explicadores es ρ = 0,60–0,70 sobre solo 5 fases. Es prudente esperar que la estabilidad entre campañas (W de Kendall ≥ 0,70) sea exigente; un solo pliego no permite concluir.

**Corrida completa P0–P10 (reports/piloto_tecnico/, no vinculante).** Resumen:
- **P1 (fuga por permutación): pasa** para los 5 modelos (R² mediano ≤ 0 con objetivo permutado).
- **P4 → anomalía** (389,0 vs 204,7 kg/ha; −47,4 %). **P5 → media** (las tres formas de B2 empatan en 202,39). **P7:** early stopping aleatorio y temporal casi idénticos (mejor época mediana 7; RMSE 203,5 vs 204,3); épocas máximas propuestas 32. **P8:** `deep_explainer` (error local 0,0000; ρ por fase vs Kernel = 1,00; fase máxima floración en ambos).
- **P6 (semillas):** CNN-1D mediana 204,3 (EE 2,23; rango 194,0–211,6); Random Forest 202,2 (EE 0,32); XGBoost salió con 211,4 idéntico en las 10 semillas porque, con los valores por defecto, es determinista (la semilla sí se aplicaba). Se añadió `subsample` y `colsample_bytree` ∈ [0,6; 1,0] al espacio (E5: `submuestreo_de_xgboost`, por_decidir_piloto); tras repetir la corrida XGBoost muestra mediana 212,6 (EE 1,32; rango 208,3–218,6) y en P9 pasa a marcar `subsample` en el borde superior.
- **P9 (bordes):** hay parámetros en el borde del espacio de búsqueda: `cnn1d.dropout` (superior), `elastic_net.alpha` (inferior), `xgboost.max_depth` (inferior), `svr.C` (inferior). Con solo 6 ensayos evaluados por modelo es una alerta, no un diagnóstico; ampliar el espacio, si se decide, debe ser para los 5 modelos por igual y registrarse en E5.
- **P10 (curva de aprendizaje, XGBoost):** el RMSE sube con más campañas (194,7 → 203,3 → 211,4). Es un solo pliego y las campañas añadidas cambian la historia; no se interpreta.
- **Sobre B2:** en la campaña 2019 la media provincial da 202,4 kg/ha, la media de las 2 últimas campañas 200,2 y la persistencia (última campaña) 198,9. En 2018 (historia 2016–2017): media 185,3 vs persistencia 166,9. Referencias interanuales simples **ya igualan a los modelos**; con N = 5 campañas externas, HE1b (CNN < B2 en las 5) será muy exigente. Pregunta para el asesor: ¿se acepta una B2 más fuerte (p. ej. persistencia) y se dan identidad/historial de provincia a los modelos?

**Ensayo técnico de HE2 de extremo a extremo (`scripts/run_he2_tecnico.py`; no vinculante).** Cadena completa sobre campañas de **entrenamiento** 2016–2019: pliegos de origen móvil con ventana 2 (se evalúan 2018 y 2019), CNN-1D con los hiperparámetros y las 32 épocas del piloto, 10 semillas promediadas por campaña, DeepExplainer y oclusión de cada una de las 5 fases (21 s en total).
- Fase máxima: desarrollo vegetativo en 2018 y floración en 2019. Las contribuciones son casi planas (0,12–0,18) y los aumentos de RMSE por oclusión, minúsculos (−0,001 a 0,02 desviaciones estándar): el modelo apenas usa la información de las fases, coherente con que la CNN iguale a B2 sin superarla.
- **W de Kendall = 0,80 y Jaccard = 0,50 cumplen los umbrales de la tesis**, pero el p de permutación de la estabilidad es 0,70: con solo 2 campañas, W alto se obtiene por azar. Es el caso que justifica exigir el valor p además del umbral.
- p de fidelidad 0,76 y de concordancia 0,64; ninguno rechaza y HE2 no se respalda. Con 2 campañas el p mínimo posible de concordancia es 0,16, así que el ensayo solo comprueba que la cadena funciona y su costo, no la hipótesis.

*(Actualización con HE2 definitivo.)* Con las reglas de la tesis v3 el ensayo técnico da: concordancia (primaria) p = 0,64 y HE2 no se respalda; estabilidad y fidelidad (secundarias) no se rechazan; la proporción de contribución en floración + llenado es 0,43, apenas sobre el 0,40 del azar. Es coherente con un modelo que apenas usa la información de las fases.

## 11. Piloto oficial (2026-10-10, vinculante)

Ejecutado con el gate de F2 confirmado por el autor y ventana 5 (campañas 2016–2020; pruebas del piloto: 2019 y 2020). Informe completo en `docs/crisp/Piloto-oficial-informe.md`. Cerró los 11 parámetros `por_decidir_piloto` de E5: representación = anomalía; B2 = tendencia lineal con ≥ 4 campañas (`HISTORIA_MINIMA_CAMPANAS = 4`); partición temporal del early stopping con restauración de la mejor época; 47 épocas máximas; DeepExplainer con tolerancia de 0,05 de la desviación de la salida; submuestreo de XGBoost confirmado; espacios de búsqueda congelados tras una sola ampliación (`alpha`, `C`, dropout). La prueba de fuga pasa en los 5 modelos.

Hallazgo para la discusión: sin tendencia en la línea base los modelos fallaban en 2020 (RMSE 231 a 235) y con ella igualan a B2 (188 a 196 frente a 194); la persistencia (B4, 172) sigue siendo la referencia más difícil. Persisten ensayos en el borde de `alpha`, `C`, `filtros_capa1` y `subsample`: el mejor ajuste es casi la línea base. Queda declarado como limitación.

**E5 sin pendientes (2026-10-10).** Con el piloto oficial se cerraron también los 18 parámetros `propuesto` (valores sin cambio salvo `n_bootstrap`, que pasa a servir solo para los intervalos descriptivos; `benchmark_secundario_persistencia`, que queda como candidata a B*; y la oclusión, que ahora incluye el reemplazo condicional de la tesis v3). Estado de E5: 0 pendientes del asesor, 0 del piloto y 0 propuestos; `assert_can_freeze` pasa. Para llegar al hito H2 solo falta **congelar** el pre-registro (fecha, commit y sha256 de los datos) y que el asesor revise las decisiones.

**Indicadores de las dimensiones (P11, 2026-10-10).** Se implementaron y se reportan los indicadores del Anexo 01 (`src/evaluation/indicators.py`): para Y1, RMSE, MAE, rRMSE y R² por modelo y campaña; para X1, campo receptivo temporal (5 pasos, igual a la longitud del tensor), 100 % de predicciones válidas y 100 % de ejecuciones reproducibles; para X2, fracción de predicciones explicadas y valor SHAP absoluto medio por predictor (en el piloto: ndvi_max 0,146, radiación solar 0,120, precipitación 0,111, temperatura mínima 0,109, ndvi_mean 0,083, temperatura máxima 0,065). Se repitió el piloto oficial y todos los resultados coinciden con la ronda 2 previa.

Dos matices para la discusión: (1) el R² de 0,79 a 0,86 en kg/ha es alto en todos los modelos y también en B1, porque la mayor parte de la varianza es entre provincias; no mide habilidad interanual, por eso la comparación relevante es el skill score frente a B*. (2) El piloto explica solo 30 de 58 predicciones de prueba (52 %) por costo; en la evaluación confirmatoria se explicarán todas las observaciones externas.
