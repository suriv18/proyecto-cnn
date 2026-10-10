# E5 · Preespecificación (registro fechado)

**Estado:** BORRADOR · **Versión:** 0.1 · **Borrador:** 2026-10-10 · **Congelado:** no

> Generado desde `configs/preespecificacion.yaml` (fuente única de verdad) con `python scripts/render_e5.py`; no se edita a mano. Fase F3 de CRISP-ML(Q), hito H2 (sección 4.18 de la tesis).

Contiene **82 parámetros**. Solo puede congelarse cuando no quede ninguno `propuesto` ni `por_decidir_*`; después, todo cambio exige una enmienda fechada y trazable.

## Decisiones pendientes

| Estado | Parámetros |
|---|---|
| `por_decidir_asesor` | 3 |
| `por_decidir_piloto` | 11 |
| `propuesto` | 18 |

### Requieren al asesor

- `datos.provincias_de_transicion` — incluidas_y_marcadas · _Recomendación firme en rol de asesor (docs/crisp/Decisiones-asesor-simuladas.md); falta ratificación real. 4.5.3; E3 §7 (evidencia SISAGRI por distrito)_
- `datos.ciclos_fuera_de_la_ventana_de_extraccion` — excluir · _Recomendación firme en rol de asesor (docs/crisp/Decisiones-asesor-simuladas.md); falta ratificación real. 4.7 criterio 2; alternativa: ampliar la ventana a diciembre y re-extraer_
- `validacion.ventana_inicial_de_entrenamiento_campanas` — 5 · _Recomendación en rol de asesor (docs/crisp/Decisiones-asesor-simuladas.md); falta ratificación real. 4.12.1; deja 5 campañas externas (2021-2025); se evaluó 4 (6 externas) y se descartó: HE1b pasaría a exigir 6 de 6, B2 con tendencia necesita ≥ 5 campañas de historia y obligaría a recalibrar la máscara; el piloto informa pero no decide con campañas externas_

### Las decide el ensayo piloto (sin campañas externas)

- `modelos.espacios_de_busqueda` — configs/hyperparams.yaml · _4.12.3: se publican y congelan antes de la evaluación; hoy son un marcador_
- `modelos.submuestreo_de_xgboost` — subsample y colsample_bytree en [0.6, 1.0] dentro del espacio de búsqueda · _con los valores por defecto XGBoost es determinista y las 10 semillas de 4.12.4 serían idénticas (corrida técnica, P6); el piloto oficial lo confirma_
- `modelos.entrenamiento_cnn1d.early_stopping_particion` — temporal: la última campaña del entrenamiento · _BRECHA G1: el código usa una partición ALEATORIA del 20 % (4.12.2 exige temporal)_
- `modelos.entrenamiento_cnn1d.restaurar_mejor_epoca` — sí · _BRECHA G2: el código devuelve el modelo de la última época_
- `modelos.entrenamiento_cnn1d.epocas_maximas` — _sin definir_ · _se fija con la curva de pérdida del piloto_
- `variable_objetivo.representacion_de_modelado` — _sin definir_ · _4.11: nivel o anomalía respecto de la línea base provincial; se elige sin campañas externas_
- `variable_objetivo.historia_minima_para_tendencia_campanas` — 5 · _target_transform.HISTORIA_MINIMA_CAMPANAS. Con menos campañas B2 cae a la media provincial y coincide con B1 (verificado en la corrida técnica: B1 = B2 = 202 kg/ha con 3 campañas). Con 5 campañas de entrenamiento la primera campaña externa tiene exactamente 5 puntos por provincia; una tendencia con tan pocos puntos es muy ruidosa_
- `variable_objetivo.forma_de_la_linea_base_b2` — _sin definir_ · _4.11: lineal o suavizada; respaldo = media si hay menos de 5 campañas_
- `interpretabilidad.variante_de_shap` — deep_explainer · _4.15; contingencia: KernelExplainer si falla la precisión local_
- `interpretabilidad.escala_de_salida_explicada` — la de la representación de modelado elegida · _4.15_
- `interpretabilidad.tolerancia_de_precision_local` — _sin definir_ · _se calibra en el piloto como fracción de la desviación estándar de la salida_

### Propuestos, a revisar y fijar

- `hipotesis.n_bootstrap` — 10000 · _con 5 campañas externas solo hay 5^5 = 3125 remuestreos distintos; más de 10 000 no aporta_
- `hipotesis.semilla_bootstrap` — 2026 · _reproducibilidad_
- `datos.clasificaciones_excluidas` — costa_riego, selva · _4.7, criterio de exclusión 4_
- `datos.exclusion_por_altitud_del_cultivo` — no · _E3 §7: la altitud del cultivo no es la de la quinua_
- `datos.criterio_de_inclusion_por_completitud` — >= 80 % de celdas válidas por observación (601 de 601 la cumplen); las celdas vacías residuales se imputan con la mediana de entrenamiento y se marcan · _4.7 (interpretación operacional de «pasos válidos»: celdas V×T); alternativa de sensibilidad: solo el tensor completo (571)_
- `datos.imputacion_de_ndvi_max_dias_de_arrastre` — 16 · _4.7; un compuesto MODIS; solo observaciones previas (sin fuga)_
- `validacion.validacion_interna.esquema` — origen_movil_dentro_del_entrenamiento · _4.12.2: orden temporal y solo campañas anteriores_
- `validacion.validacion_interna.ventana_inicial_interna_campanas` — 2 · _con menos de 2 campañas no hay entrenamiento interno_
- `validacion.semillas.valores` — 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 · _reproducibilidad_
- `modelos.benchmark_secundario_persistencia` — B4 · _recomendación en rol de asesor (docs/crisp/Decisiones-asesor-simuladas.md): repite la última campaña de entrenamiento de la provincia; descriptivo, fuera de la familia de Holm; no se optimiza_
- `modelos.presupuesto_de_trials_por_modelo` — 30 · _src/models/tuning.py; igual para los 5 modelos (4.12.3); el piloto confirma el costo_
- `modelos.entrenamiento_cnn1d.paciencia` — 10 · _TrainingConfig_
- `modelos.imputacion_residual_de_celdas_vacias` — mediana de las campañas de entrenamiento por variable y fase, para los 5 modelos · _4.12.2; la misma información para todos (Tabla 8); afecta al 5 % de las observaciones_
- `modelos.estandarizacion_del_objetivo` — z-score con la media y la desviación del entrenamiento del pliego, para los 5 modelos; las predicciones se invierten a kg/ha · _los espacios de búsqueda (epsilon de SVR, C, alpha) suponen un objetivo estandarizado; el rendimiento está en kg/ha (~1 000)_
- `interpretabilidad.conjunto_de_referencia` — muestra aleatoria de hasta 100 observaciones de las campañas de entrenamiento del pliegue, semilla fija · _4.15; solo entrenamiento_
- `interpretabilidad.tratamiento_de_predictores_correlacionados` — sin supuesto de independencia; se informa la variante y se compara con KernelExplainer en el piloto · _2.3.2 (Aas et al., 2021), 4.15_
- `interpretabilidad.kernel_explainer_regularizacion` — l1_reg = 0 (sin regularización L1) · _piloto técnico (no vinculante): con l1_reg=auto el Kernel encoge las atribuciones (suma |SHAP| a 1/2,2 de Deep, correlación 0,87); con 0 la correlación con Deep es 0,95_
- `interpretabilidad.oclusion.valor_de_referencia` — media del conjunto de referencia por variable · _4.15_

## Registro completo

### hipotesis

| Parámetro | Valor | Estado | Fuente |
|---|---|---|---|
| `nivel_de_significancia` | 0.05 | `fijado_por_tesis` | 3.3 |
| `margen_no_inferioridad_relativo` | 0.05 | `fijado_por_tesis` | 3.3.1 (δ) |
| `comparador_confirmatorio` | xgboost | `fijado_por_tesis` | Tabla 8 |
| `benchmark_de_superioridad` | B* = la mejor entre B2 (línea base provincial) y B4 (persistencia), elegida con la validación interna antes de las campañas externas | `fijado_por_tesis` | Tesis v3 (decisión del autor; el asesor la revisa): 3.3.1 y 3.3.2; antes: B2 |
| `unidad_del_contraste` | campaña externa | `fijado_por_tesis` | 3.3 |
| `estadistico_he1a` | q_f(R) = RMSE(CNN-1D)/RMSE(R) - 1 con R en {XGBoost, B*}; H0: mediana >= 0.05 | `fijado_por_tesis` | Tesis v3 (decisión del autor; el asesor la revisa): 3.3.1; antes solo frente a XGBoost |
| `estadistico_he1b` | exploratorio: skill score SS_f = 1 - MSE(CNN-1D)/MSE(B*) (Murphy, 1988); H0: mediana(SS_f) <= 0 | `fijado_por_tesis` | Tesis v3 (decisión del autor; el asesor la revisa): 3.3.2; antes: Δ_f frente a B2, confirmatorio |
| `intervalo_de_confianza` | regla exacta de signos: q_f(R) < 0.05 en todas las campañas externas (nivel 0.5^n); intervalos por bloques solo descriptivos | `fijado_por_tesis` | Tesis v3 (decisión del autor; el asesor la revisa): 4.14; con 5 bloques el bootstrap tiene 3125 remuestreos distintos |
| `n_bootstrap` | 10000 | `propuesto` | con 5 campañas externas solo hay 5^5 = 3125 remuestreos distintos; más de 10 000 no aporta |
| `semilla_bootstrap` | 2026 | `propuesto` | reproducibilidad |
| `contraste_de_apoyo` | Wilcoxon pareado unilateral, equivalencia práctica bayesiana (ROPE ±5 %), Hodges-Lehmann y correlación biserial de rangos | `fijado_por_tesis` | Tesis v3 (decisión del autor; el asesor la revisa): 3.3.1 y 4.14 |
| `he1_se_respalda_si` | se cumple HE1a (HE1b es exploratoria) | `fijado_por_tesis` | Tesis v3 (decisión del autor; el asesor la revisa): 3.3.2 |
| `he2.estabilidad_w_kendall_minimo` | 0.7 | `fijado_por_tesis` | 3.3.3 |
| `he2.estabilidad_jaccard_minimo` | 0.5 | `fijado_por_tesis` | 3.3.3 |
| `he2.n_fases_principales` | 3 | `fijado_por_tesis` | 3.3.3 |
| `he2.fases_criticas_horizonte_principal` | floracion, llenado_grano | `fijado_por_tesis` | 3.3.5 (Vásquez et al., 2024) |
| `he2.fases_criticas_horizonte_temprano` | floracion | `fijado_por_tesis` | 3.3.5 |
| `he2.correccion_por_multiplicidad` | holm | `fijado_por_tesis` | 3.3.5 |
| `he2.familia_de_contrastes` | estabilidad, fidelidad | `fijado_por_tesis` | Tesis v3 (decisión del autor; el asesor la revisa): 3.3.5; la concordancia es el criterio primario y queda fuera de la familia corregida |
| `he2.valor_p_estabilidad` | permutación de las fases dentro de cada campaña (10 000 permutaciones, semilla 2026); p = max(p de W de Kendall, p del Jaccard medio de las 3 fases principales) | `fijado_por_tesis` | src/explainability/he2_pvalues.py; los umbrales W >= 0.70 y J >= 0.50 se conservan como criterio de magnitud |
| `he2.valor_p_fidelidad` | aleatorización exacta con el rango del aumento de RMSE de la fase de mayor contribución entre las 5 fases ocluidas, sumado entre campañas; los empates van en contra | `fijado_por_tesis` | src/explainability/he2_pvalues.py; amplía el control único de 3.3.4 a las 4 fases restantes (todas de igual duración); tesis v3, 3.3.4: sustituye al control único, que con 5 campañas no pasa Holm |
| `he2.valor_p_concordancia` | binomial exacta unilateral sobre las campañas, p0 = fases del referente / fases totales (0.4 principal, 0.2 temprano) | `fijado_por_tesis` | src/explainability/he2_pvalues.py; con 5 campañas solo 5/5 aciertos da p < 0.0167 |
| `he2.he2_se_respalda_si` | se cumple la concordancia agronómica (primaria); estabilidad y fidelidad son secundarias | `fijado_por_tesis` | Tesis v3 (decisión del autor; el asesor la revisa): 3.3.5 |
| `hipotesis_general_conjuntiva` | sí | `fijado_por_tesis` | 3.1 |

### datos

| Parámetro | Valor | Estado | Fuente |
|---|---|---|---|
| `periodo_de_campanas` | 2016-2025 | `fijado_por_tesis` | Confirmado por el autor (2026-10-10). Enmienda H1: no existe fuente provincial de rendimiento anterior a 2015 (data/manifest/midagri_sisagri.yaml); el asesor la revisa |
| `clasificaciones_excluidas` | costa_riego, selva | `propuesto` | 4.7, criterio de exclusión 4 |
| `provincias_de_transicion` | incluidas_y_marcadas | `por_decidir_asesor` | Recomendación firme en rol de asesor (docs/crisp/Decisiones-asesor-simuladas.md); falta ratificación real. 4.5.3; E3 §7 (evidencia SISAGRI por distrito) |
| `exclusion_por_altitud_del_cultivo` | no | `propuesto` | E3 §7: la altitud del cultivo no es la de la quinua |
| `ciclos_fuera_de_la_ventana_de_extraccion` | excluir | `por_decidir_asesor` | Recomendación firme en rol de asesor (docs/crisp/Decisiones-asesor-simuladas.md); falta ratificación real. 4.7 criterio 2; alternativa: ampliar la ventana a diciembre y re-extraer |
| `valores_extremos_del_rendimiento` | se marcan, no se eliminan | `fijado_por_tesis` | 4.7 criterio de exclusión 3 |
| `criterio_de_inclusion_por_completitud` | >= 80 % de celdas válidas por observación (601 de 601 la cumplen); las celdas vacías residuales se imputan con la mediana de entrenamiento y se marcan | `propuesto` | 4.7 (interpretación operacional de «pasos válidos»: celdas V×T); alternativa de sensibilidad: solo el tensor completo (571) |
| `imputacion_de_ndvi_max_dias_de_arrastre` | 16 | `propuesto` | 4.7; un compuesto MODIS; solo observaciones previas (sin fuga) |
| `horizonte_principal_dias_antes_de_cosecha` | 30 | `fijado_por_tesis` | 1.5, 4.10.4 |
| `horizonte_secundario` | cierre_de_floracion | `fijado_por_tesis` | 1.5, 4.10.4 (si la cobertura lo permite) |
| `umbrales_de_la_mascara.altitud_minima_m` | 2500 | `fijado_por_tesis` | Confirmado por el autor (2026-10-10); el asesor lo revisa. data/manifest/mascara_agricola.yaml (provisional) |
| `umbrales_de_la_mascara.altitud_maxima_m` | 4000 | `fijado_por_tesis` | Confirmado por el autor (2026-10-10); el asesor lo revisa. ídem; E3 §7 |
| `umbrales_de_la_mascara.pendiente_maxima_grados` | 14.04 | `fijado_por_tesis` | Confirmado por el autor (2026-10-10); el asesor lo revisa. ídem (D.S. 005-2022-MIDAGRI) |
| `umbrales_de_la_mascara.amplitud_ndvi_minima` | 0.323 | `fijado_por_tesis` | Confirmado por el autor (2026-10-10); el asesor lo revisa. Sensibilidad en p1 y p10. Calibrado con 2016-2020, que son todas de entrenamiento con ventana 5. percentil 5 de las campañas 2016-2020; sesgo regional en Arequipa y Puno (E3 §7) |

### validacion

| Parámetro | Valor | Estado | Fuente |
|---|---|---|---|
| `esquema_confirmatorio` | origen_movil | `fijado_por_tesis` | 4.12.1 |
| `ventana_inicial_de_entrenamiento_campanas` | 5 | `por_decidir_asesor` | Recomendación en rol de asesor (docs/crisp/Decisiones-asesor-simuladas.md); falta ratificación real. 4.12.1; deja 5 campañas externas (2021-2025); se evaluó 4 (6 externas) y se descartó: HE1b pasaría a exigir 6 de 6, B2 con tendencia necesita ≥ 5 campañas de historia y obligaría a recalibrar la máscara; el piloto informa pero no decide con campañas externas |
| `esquema_complementario_loco` | dejar una campaña fuera (secundario) | `fijado_por_tesis` | 4.12.1 |
| `esquema_de_robustez_espacial` | dejar un departamento fuera (sensibilidad) | `fijado_por_tesis` | 4.12.1 |
| `validacion_interna.esquema` | origen_movil_dentro_del_entrenamiento | `propuesto` | 4.12.2: orden temporal y solo campañas anteriores |
| `validacion_interna.ventana_inicial_interna_campanas` | 2 | `propuesto` | con menos de 2 campañas no hay entrenamiento interno |
| `semillas.n_replicas` | 10 | `fijado_por_tesis` | 4.12.4 |
| `semillas.modelos_estocasticos` | cnn1d, random_forest, xgboost | `fijado_por_tesis` | 4.12.4 |
| `semillas.modelos_deterministas` | elastic_net, svr | `fijado_por_tesis` | 4.12.4 |
| `semillas.valores` | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 | `propuesto` | reproducibilidad |
| `semillas.agregacion_por_campana` | mediana | `fijado_por_tesis` | 4.12.4 y 3.3 |
| `campanas_externas` | derivadas de la ventana inicial: de la (ventana+1)-ésima campaña a la última | `fijado_por_tesis` | 4.12.1 |

### modelos

| Parámetro | Valor | Estado | Fuente |
|---|---|---|---|
| `factor_algoritmo` | cnn1d, elastic_net, random_forest, xgboost, svr | `fijado_por_tesis` | Tabla 1 y 8 |
| `benchmarks` | B1, B2, B3 | `fijado_por_tesis` | Tabla 8; no se optimizan |
| `benchmark_secundario_persistencia` | B4 | `propuesto` | recomendación en rol de asesor (docs/crisp/Decisiones-asesor-simuladas.md): repite la última campaña de entrenamiento de la provincia; descriptivo, fuera de la familia de Holm; no se optimiza |
| `presupuesto_de_trials_por_modelo` | 30 | `propuesto` | src/models/tuning.py; igual para los 5 modelos (4.12.3); el piloto confirma el costo |
| `muestreador_de_hiperparametros` | tpe | `fijado_por_tesis` | 4.12.3 (Optuna, Sabo et al. 2023) |
| `espacios_de_busqueda` | configs/hyperparams.yaml | `por_decidir_piloto` | 4.12.3: se publican y congelan antes de la evaluación; hoy son un marcador |
| `submuestreo_de_xgboost` | subsample y colsample_bytree en [0.6, 1.0] dentro del espacio de búsqueda | `por_decidir_piloto` | con los valores por defecto XGBoost es determinista y las 10 semillas de 4.12.4 serían idénticas (corrida técnica, P6); el piloto oficial lo confirma |
| `entrenamiento_cnn1d.optimizador` | adam | `fijado` | src/models/cnn1d_trainer.py |
| `entrenamiento_cnn1d.early_stopping_particion` | temporal: la última campaña del entrenamiento | `por_decidir_piloto` | BRECHA G1: el código usa una partición ALEATORIA del 20 % (4.12.2 exige temporal) |
| `entrenamiento_cnn1d.restaurar_mejor_epoca` | sí | `por_decidir_piloto` | BRECHA G2: el código devuelve el modelo de la última época |
| `entrenamiento_cnn1d.epocas_maximas` | _sin definir_ | `por_decidir_piloto` | se fija con la curva de pérdida del piloto |
| `entrenamiento_cnn1d.paciencia` | 10 | `propuesto` | TrainingConfig |
| `imputacion_residual_de_celdas_vacias` | mediana de las campañas de entrenamiento por variable y fase, para los 5 modelos | `propuesto` | 4.12.2; la misma información para todos (Tabla 8); afecta al 5 % de las observaciones |
| `estandarizacion_del_objetivo` | z-score con la media y la desviación del entrenamiento del pliego, para los 5 modelos; las predicciones se invierten a kg/ha | `propuesto` | los espacios de búsqueda (epsilon de SVR, C, alpha) suponen un objetivo estandarizado; el rendimiento está en kg/ha (~1 000) |
| `estandarizacion` | z-score por variable y fase, ajustado solo con el entrenamiento del pliegue | `fijado_por_tesis` | 4.12.2 |

### variable_objetivo

| Parámetro | Valor | Estado | Fuente |
|---|---|---|---|
| `metrica_confirmatoria` | RMSE sobre el rendimiento reconstruido en kg/ha | `fijado_por_tesis` | 4.11 y 4.13 |
| `representacion_de_modelado` | _sin definir_ | `por_decidir_piloto` | 4.11: nivel o anomalía respecto de la línea base provincial; se elige sin campañas externas |
| `historia_minima_para_tendencia_campanas` | 5 | `por_decidir_piloto` | target_transform.HISTORIA_MINIMA_CAMPANAS. Con menos campañas B2 cae a la media provincial y coincide con B1 (verificado en la corrida técnica: B1 = B2 = 202 kg/ha con 3 campañas). Con 5 campañas de entrenamiento la primera campaña externa tiene exactamente 5 puntos por provincia; una tendencia con tan pocos puntos es muy ruidosa |
| `forma_de_la_linea_base_b2` | _sin definir_ | `por_decidir_piloto` | 4.11: lineal o suavizada; respaldo = media si hay menos de 5 campañas |
| `metricas_complementarias` | mae, rrmse, r2_fuera_de_muestra | `fijado_por_tesis` | Tabla 9 |

### interpretabilidad

| Parámetro | Valor | Estado | Fuente |
|---|---|---|---|
| `variante_de_shap` | deep_explainer | `por_decidir_piloto` | 4.15; contingencia: KernelExplainer si falla la precisión local |
| `escala_de_salida_explicada` | la de la representación de modelado elegida | `por_decidir_piloto` | 4.15 |
| `conjunto_de_referencia` | muestra aleatoria de hasta 100 observaciones de las campañas de entrenamiento del pliegue, semilla fija | `propuesto` | 4.15; solo entrenamiento |
| `tolerancia_de_precision_local` | _sin definir_ | `por_decidir_piloto` | se calibra en el piloto como fracción de la desviación estándar de la salida |
| `tratamiento_de_predictores_correlacionados` | sin supuesto de independencia; se informa la variante y se compara con KernelExplainer en el piloto | `propuesto` | 2.3.2 (Aas et al., 2021), 4.15 |
| `kernel_explainer_regularizacion` | l1_reg = 0 (sin regularización L1) | `propuesto` | piloto técnico (no vinculante): con l1_reg=auto el Kernel encoge las atribuciones (suma |SHAP| a 1/2,2 de Deep, correlación 0,87); con 0 la correlación con Deep es 0,95 |
| `agregacion` | por predictor y paso temporal, y por fase; semillas promediadas dentro de cada campaña antes de evaluar estabilidad | `fijado_por_tesis` | 4.15 |
| `oclusion.valor_de_referencia` | media del conjunto de referencia por variable | `propuesto` | 4.15 |
| `oclusion.ventana_de_control` | fase de igual duración y menor contribución | `fijado_por_tesis` | 3.3.4 |
| `oclusion.sin_reentrenar` | sí | `fijado_por_tesis` | 3.3.4 (ablación = con reentrenamiento, análisis secundario) |

### sensibilidad

| Parámetro | Valor | Estado | Fuente |
|---|---|---|---|
| `analisis` | horizonte_de_pronostico, nivel_frente_a_anomalia, participacion_de_quinua_y_representatividad_de_la_mascara, calidad_de_la_alineacion_fenologica, fuente_climatica, campanas_extremas, percentil_de_amplitud_p1_p5, provincias_de_transicion, ventana_de_extraccion, imputacion_de_ndvi, exclusion_por_altitud, solo_tensor_completo_571 | `fijado_por_tesis` | 4.14 y 4.10.1; E3 §7 |

### politica_de_desviaciones

| Parámetro | Valor | Estado | Fuente |
|---|---|---|---|
| `regla` | toda desviación posterior al congelamiento se registra como enmienda fechada y trazable; no se ajustan umbrales, márgenes ni comparador tras ver resultados de las campañas externas | `fijado_por_tesis` | 4.18 y 6 |

**Política de desviaciones:** toda desviación posterior al congelamiento se registra como enmienda fechada y trazable; no se ajustan umbrales, márgenes ni comparador tras ver resultados de las campañas externas
