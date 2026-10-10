# E5 · Preespecificación (registro fechado)

**Estado:** BORRADOR · **Versión:** 0.1 · **Borrador:** 2026-10-10 · **Congelado:** no

> Generado desde `configs/preespecificacion.yaml` (fuente única de verdad) con `python scripts/render_e5.py`; no se edita a mano. Fase F3 de CRISP-ML(Q), hito H2 (sección 4.18 de la tesis).

Contiene **76 parámetros**. Solo puede congelarse cuando no quede ninguno `propuesto` ni `por_decidir_*`; después, todo cambio exige una enmienda fechada y trazable.

## Decisiones pendientes

| Estado | Parámetros |
|---|---|
| `por_decidir_asesor` | 8 |
| `por_decidir_piloto` | 10 |
| `propuesto` | 16 |

### Requieren al asesor

- `datos.periodo_de_campanas` — 2016-2025 · _enmienda H1 (2006-2024 -> 2016-2025), estado PROPUESTA en data/manifest/midagri_sisagri.yaml_
- `datos.provincias_de_transicion` — incluidas_y_marcadas · _4.5.3; E3 §7 (evidencia SISAGRI por distrito)_
- `datos.ciclos_fuera_de_la_ventana_de_extraccion` — excluir · _4.7 criterio 2; alternativa: ampliar la ventana a diciembre y re-extraer_
- `datos.umbrales_de_la_mascara.altitud_minima_m` — 2500 · _data/manifest/mascara_agricola.yaml (provisional)_
- `datos.umbrales_de_la_mascara.altitud_maxima_m` — 4000 · _ídem; E3 §7_
- `datos.umbrales_de_la_mascara.pendiente_maxima_grados` — 14.04 · _ídem (D.S. 005-2022-MIDAGRI)_
- `datos.umbrales_de_la_mascara.amplitud_ndvi_minima` — 0.323 · _percentil 5 de las campañas 2016-2020; sesgo regional en Arequipa y Puno (E3 §7)_
- `validacion.ventana_inicial_de_entrenamiento_campanas` — 5 · _4.12.1; deja 5 campañas externas (2021-2025); el piloto informa pero no decide con campañas externas_

### Las decide el ensayo piloto (sin campañas externas)

- `modelos.espacios_de_busqueda` — configs/hyperparams.yaml · _4.12.3: se publican y congelan antes de la evaluación; hoy son un marcador_
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
- `modelos.presupuesto_de_trials_por_modelo` — 30 · _src/models/tuning.py; igual para los 5 modelos (4.12.3); el piloto confirma el costo_
- `modelos.entrenamiento_cnn1d.paciencia` — 10 · _TrainingConfig_
- `modelos.imputacion_residual_de_celdas_vacias` — mediana de las campañas de entrenamiento por variable y fase, para los 5 modelos · _4.12.2; la misma información para todos (Tabla 8); afecta al 5 % de las observaciones_
- `modelos.estandarizacion_del_objetivo` — z-score con la media y la desviación del entrenamiento del pliego, para los 5 modelos; las predicciones se invierten a kg/ha · _los espacios de búsqueda (epsilon de SVR, C, alpha) suponen un objetivo estandarizado; el rendimiento está en kg/ha (~1 000)_
- `interpretabilidad.conjunto_de_referencia` — muestra aleatoria de hasta 100 observaciones de las campañas de entrenamiento del pliegue, semilla fija · _4.15; solo entrenamiento_
- `interpretabilidad.tratamiento_de_predictores_correlacionados` — sin supuesto de independencia; se informa la variante y se compara con KernelExplainer en el piloto · _2.3.2 (Aas et al., 2021), 4.15_
- `interpretabilidad.oclusion.valor_de_referencia` — media del conjunto de referencia por variable · _4.15_

## Registro completo

### hipotesis

| Parámetro | Valor | Estado | Fuente |
|---|---|---|---|
| `nivel_de_significancia` | 0.05 | `fijado_por_tesis` | 3.3 |
| `margen_no_inferioridad_relativo` | 0.05 | `fijado_por_tesis` | 3.3.1 (δ) |
| `comparador_confirmatorio` | xgboost | `fijado_por_tesis` | Tabla 8 |
| `benchmark_de_superioridad` | B2 | `fijado_por_tesis` | 3.3.2 |
| `unidad_del_contraste` | campaña externa | `fijado_por_tesis` | 3.3 |
| `estadistico_he1a` | mediana de q_f = RMSE(CNN-1D)/RMSE(XGBoost) - 1; H0: mediana >= 0.05 | `fijado_por_tesis` | 3.3.1 |
| `estadistico_he1b` | mediana de Δ_f = RMSE(CNN-1D) - RMSE(B2); H0: mediana >= 0 | `fijado_por_tesis` | 3.3.2 |
| `intervalo_de_confianza` | límite superior unilateral al 95 % por bootstrap de bloques de campaña | `fijado_por_tesis` | 4.14 |
| `n_bootstrap` | 10000 | `propuesto` | con 5 campañas externas solo hay 5^5 = 3125 remuestreos distintos; más de 10 000 no aporta |
| `semilla_bootstrap` | 2026 | `propuesto` | reproducibilidad |
| `contraste_de_apoyo` | Wilcoxon pareado unilateral, Hodges-Lehmann y correlación biserial de rangos | `fijado_por_tesis` | 3.3.1 y 4.14 |
| `he1_se_respalda_si` | se cumplen HE1a y HE1b | `fijado_por_tesis` | 3.3.2 |
| `he2.estabilidad_w_kendall_minimo` | 0.7 | `fijado_por_tesis` | 3.3.3 |
| `he2.estabilidad_jaccard_minimo` | 0.5 | `fijado_por_tesis` | 3.3.3 |
| `he2.n_fases_principales` | 3 | `fijado_por_tesis` | 3.3.3 |
| `he2.fases_criticas_horizonte_principal` | floracion, llenado_grano | `fijado_por_tesis` | 3.3.5 (Vásquez et al., 2024) |
| `he2.fases_criticas_horizonte_temprano` | floracion | `fijado_por_tesis` | 3.3.5 |
| `he2.correccion_por_multiplicidad` | holm | `fijado_por_tesis` | 3.3.5 |
| `he2.familia_de_contrastes` | estabilidad, fidelidad, concordancia | `fijado_por_tesis` | 3.3.5 |
| `he2.he2_se_respalda_si` | se cumplen estabilidad, fidelidad y concordancia a la vez | `fijado_por_tesis` | 3.3.5 |
| `hipotesis_general_conjuntiva` | sí | `fijado_por_tesis` | 3.1 |

### datos

| Parámetro | Valor | Estado | Fuente |
|---|---|---|---|
| `periodo_de_campanas` | 2016-2025 | `por_decidir_asesor` | enmienda H1 (2006-2024 -> 2016-2025), estado PROPUESTA en data/manifest/midagri_sisagri.yaml |
| `clasificaciones_excluidas` | costa_riego, selva | `propuesto` | 4.7, criterio de exclusión 4 |
| `provincias_de_transicion` | incluidas_y_marcadas | `por_decidir_asesor` | 4.5.3; E3 §7 (evidencia SISAGRI por distrito) |
| `exclusion_por_altitud_del_cultivo` | no | `propuesto` | E3 §7: la altitud del cultivo no es la de la quinua |
| `ciclos_fuera_de_la_ventana_de_extraccion` | excluir | `por_decidir_asesor` | 4.7 criterio 2; alternativa: ampliar la ventana a diciembre y re-extraer |
| `valores_extremos_del_rendimiento` | se marcan, no se eliminan | `fijado_por_tesis` | 4.7 criterio de exclusión 3 |
| `criterio_de_inclusion_por_completitud` | >= 80 % de celdas válidas por observación (601 de 601 la cumplen); las celdas vacías residuales se imputan con la mediana de entrenamiento y se marcan | `propuesto` | 4.7 (interpretación operacional de «pasos válidos»: celdas V×T); alternativa de sensibilidad: solo el tensor completo (571) |
| `imputacion_de_ndvi_max_dias_de_arrastre` | 16 | `propuesto` | 4.7; un compuesto MODIS; solo observaciones previas (sin fuga) |
| `horizonte_principal_dias_antes_de_cosecha` | 30 | `fijado_por_tesis` | 1.5, 4.10.4 |
| `horizonte_secundario` | cierre_de_floracion | `fijado_por_tesis` | 1.5, 4.10.4 (si la cobertura lo permite) |
| `umbrales_de_la_mascara.altitud_minima_m` | 2500 | `por_decidir_asesor` | data/manifest/mascara_agricola.yaml (provisional) |
| `umbrales_de_la_mascara.altitud_maxima_m` | 4000 | `por_decidir_asesor` | ídem; E3 §7 |
| `umbrales_de_la_mascara.pendiente_maxima_grados` | 14.04 | `por_decidir_asesor` | ídem (D.S. 005-2022-MIDAGRI) |
| `umbrales_de_la_mascara.amplitud_ndvi_minima` | 0.323 | `por_decidir_asesor` | percentil 5 de las campañas 2016-2020; sesgo regional en Arequipa y Puno (E3 §7) |

### validacion

| Parámetro | Valor | Estado | Fuente |
|---|---|---|---|
| `esquema_confirmatorio` | origen_movil | `fijado_por_tesis` | 4.12.1 |
| `ventana_inicial_de_entrenamiento_campanas` | 5 | `por_decidir_asesor` | 4.12.1; deja 5 campañas externas (2021-2025); el piloto informa pero no decide con campañas externas |
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
| `presupuesto_de_trials_por_modelo` | 30 | `propuesto` | src/models/tuning.py; igual para los 5 modelos (4.12.3); el piloto confirma el costo |
| `muestreador_de_hiperparametros` | tpe | `fijado_por_tesis` | 4.12.3 (Optuna, Sabo et al. 2023) |
| `espacios_de_busqueda` | configs/hyperparams.yaml | `por_decidir_piloto` | 4.12.3: se publican y congelan antes de la evaluación; hoy son un marcador |
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
