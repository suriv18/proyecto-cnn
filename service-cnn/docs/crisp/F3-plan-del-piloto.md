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

- **Datos del piloto:** únicamente las campañas **2016–2020**: **302 observaciones elegibles** (60, 62, 60, 62 y 58 por campaña; 25 con alguna celda vacía que se imputa con la mediana de entrenamiento). Las campañas 2021–2025 no se cargan, no se grafican y no se usan para ninguna decisión.
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
| P1 | **Prueba de fuga** | Entrena cada modelo con el rendimiento de entrenamiento **permutado** y evalúa en la campaña de prueba | Pasa si el R² fuera de muestra ≤ 0,05 en los cinco modelos; si algún modelo supera ese valor hay fuga y se corrige antes de seguir |
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
| T5 | Agregación por campaña y conexión con `evaluate_non_inferiority` / `evaluate_superiority` | M | T4 |
| T6 | Paso de SHAP con verificación de precisión local y contingencia a KernelExplainer | M | T4 |
| T7 | `scripts/run_pilot.py` (P0–P11) y plantilla del informe del piloto | M | T1–T6 |
| T8 | Prueba de fuga por permutación (P1) como prueba automática | S | T4 |

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
