# Piloto oficial: informe (ronda 2, vinculante)

Copia del informe generado por `scripts/run_pilot.py --modo oficial --gate-f2-aprobado --ventana-confirmatoria 5` el 2026-10-10. Solo se cargaron las campañas 2016–2020; 2021–2025 se descartaron al leer. Es la **ronda 2**: la ronda 1 (espacios de búsqueda originales) se ejecutó antes y se repitió tras ensanchar `alpha`, `C` y el dropout, y tras fijar la historia mínima de B2 en 4 campañas. Los valores de E5 que cierra se muestran en `E5-preespecificacion-BORRADOR.md`. sha256 (16 primeros) de `configs/hyperparams.yaml`: `94071bbe66b3550f`.

**Cambios entre rondas** (todos decididos sin campañas externas): con B2 de tendencia lineal desde 4 campañas, los cuatro modelos dejan de verse penalizados en 2020 (CNN-1D 196,4 frente a 234,6 kg/ha en la ronda 1; Random Forest 188,7 frente a 230,7; XGBoost 187,6 frente a 235,4). La caída de error se debe a la línea base, no al modelo: en la ronda 1 la anomalía se medía respecto de la media, que no recoge la deriva temporal del rendimiento.

**Lectura prudente.** Frente a las referencias de 2020: B4 (persistencia) 171,8; B2 lineal 193,8; modelos 187,6 a 196,4. La CNN-1D queda +1,3 % sobre B2 y +14,3 % sobre B4. Si B* resulta ser B4, la no inferioridad exigida (menos de 5 %) fallaría; si es B2, se cumpliría en estos dos pliegos. El criterio de elección de B* (validación interna) es lo que decide cuál es la vara.

---


> Piloto oficial: gate de F2 cerrado.

## P0 · Puerta de entrada
- Campañas permitidas: [2016, 2017, 2018, 2019, 2020]; **descartadas al cargar (solo sus identificadores): 2021, 2022, 2023, 2024, 2025**.
- Campañas de prueba del piloto: [2019, 2020] · observaciones: 302.
- sha256 de `tensores.npz`: `3a44df550568e9af…`
- Pendientes de E5: {'por_decidir_asesor': 0, 'por_decidir_piloto': 0, 'propuesto': 0} · discrepancias con el código: []

## P1 · Prueba de fuga por permutación — PASA
| modelo | r2_mediano | r2_maximo | pasa |
|---|---|---|---|
| cnn1d | -0.217 | -0.087 | True |
| elastic_net | -0.006 | -0.006 | True |
| random_forest | -0.088 | 0.018 | True |
| xgboost | -0.219 | 0.015 | True |
| svr | -0.579 | -0.320 | True |

## P2 · Referencias (RMSE en kg/ha)
| modelo | campana_prueba | rmse |
|---|---|---|
| B1 | 2019 | 202.392 |
| B1 | 2020 | 231.115 |
| B2 | 2019 | 202.392 |
| B2 | 2020 | 193.791 |
| B3 | 2019 | 466.077 |
| B3 | 2020 | 507.595 |
| B4 | 2019 | 198.893 |
| B4 | 2020 | 171.840 |

## P3 · Costo y presupuesto (igual para los 5 modelos)
| representacion | modelo | n_trials | segundos_ajuste | segundos_semillas | rmse_validacion_interna |
|---|---|---|---|---|---|
| nivel | cnn1d | 30 | 37.77 | 8.58 | 371.37 |
| nivel | elastic_net | 30 | 0.35 | 0.01 | 376.96 |
| nivel | random_forest | 30 | 29.74 | 5.32 | 346.30 |
| nivel | xgboost | 30 | 18.22 | 3.53 | 333.50 |
| nivel | svr | 30 | 0.33 | 0.02 | 348.32 |
| nivel | cnn1d | 30 | 54.51 | 10.86 | 355.91 |
| nivel | elastic_net | 30 | 0.28 | 0.01 | 388.19 |
| nivel | random_forest | 30 | 110.83 | 39.73 | 374.80 |
| nivel | xgboost | 30 | 37.46 | 6.48 | 343.80 |
| nivel | svr | 30 | 0.61 | 0.02 | 329.76 |
| anomalia | cnn1d | 30 | 7.84 | 2.20 | 179.76 |
| anomalia | elastic_net | 30 | 0.19 | 0.00 | 184.78 |
| anomalia | random_forest | 30 | 43.48 | 20.23 | 180.60 |
| anomalia | xgboost | 30 | 9.36 | 5.32 | 176.84 |
| anomalia | svr | 30 | 0.30 | 0.01 | 184.26 |
| anomalia | cnn1d | 30 | 15.32 | 3.57 | 188.23 |
| anomalia | elastic_net | 30 | 0.32 | 0.00 | 193.50 |
| anomalia | random_forest | 30 | 101.20 | 30.69 | 191.16 |
| anomalia | xgboost | 30 | 29.11 | 6.48 | 190.24 |
| anomalia | svr | 30 | 0.65 | 0.02 | 193.41 |

## P4 · Representación del objetivo
Nivel 394.6 kg/ha · anomalía 198.2 kg/ha (diferencia relativa -49.8 %) → **anomalia**.

## P5 · Forma de B2
| forma | campana_prueba | rmse |
|---|---|---|
| media | 2019 | 202.392 |
| lineal_min4 | 2019 | 202.392 |
| media_ultimas3 | 2019 | 202.392 |
| media | 2020 | 231.115 |
| lineal_min4 | 2020 | 193.791 |
| media_ultimas3 | 2020 | 218.640 |
Decisión: **lineal_min4** (regla: menor RMSE medio; si la diferencia es menor que 2 % se conserva la media).

## P6 · Variabilidad por semilla
| modelo | campana_prueba | n_semillas | mediana | q1 | q3 | minimo | maximo | error_estandar_mediana |
|---|---|---|---|---|---|---|---|---|
| cnn1d | 2019 | 10 | 200.356 | 197.105 | 204.419 | 192.597 | 208.048 | 1.952 |
| cnn1d | 2020 | 10 | 196.368 | 195.548 | 197.200 | 190.675 | 201.820 | 1.207 |
| random_forest | 2019 | 10 | 202.171 | 201.439 | 202.726 | 200.765 | 202.953 | 0.324 |
| random_forest | 2020 | 10 | 188.650 | 188.393 | 188.803 | 187.306 | 189.234 | 0.220 |
| xgboost | 2019 | 10 | 212.558 | 210.798 | 215.348 | 208.344 | 218.560 | 1.317 |
| xgboost | 2020 | 10 | 187.600 | 186.599 | 189.302 | 185.876 | 191.614 | 0.778 |

## P7 · Entrenamiento de la CNN-1D
| particion | mejor_epoca_mediana | rmse_mediano |
|---|---|---|
| aleatoria | 7.000 | 198.163 |
| temporal | 10.000 | 196.368 |
Épocas máximas propuestas: **47**.

## P8 · SHAP
Variante: **deep_explainer** · error de precisión local de DeepExplainer 0.0000 (tolerancia 0.0139) · Spearman por fase contra Kernel 0.90 · fase máxima: llenado_grano (Kernel: floracion).

## P9 · Bordes de los espacios de búsqueda
| modelo | parametro | lado | fraccion_en_borde | n_ensayos_evaluados |
|---|---|---|---|---|
| cnn1d | filtros_capa1 | inferior | 0.667 | 12 |
| elastic_net | alpha | inferior | 0.667 | 12 |
| xgboost | subsample | superior | 0.667 | 12 |
| svr | C | inferior | 0.833 | 12 |

## P10 · Curva de aprendizaje (XGBoost)
| n_campanas_entrenamiento | n_observaciones_entrenamiento | rmse_xgboost |
|---|---|---|
| 1 | 62 | 176.053 |
| 2 | 122 | 210.394 |
| 3 | 184 | 217.950 |
| 4 | 244 | 187.450 |

## P11 · Indicadores de las dimensiones X1, X2 e Y1 (Anexo 01)
**Y1, exactitud predictiva** (mediana entre semillas; rRMSE vacío si la media observada es cercana a cero):

| modelo | campana_prueba | rmse | mae | rrmse | r2 |
|---|---|---|---|---|---|
| B1 | 2019 | 202.392 | 136.367 | 15.157 | 0.794 |
| B1 | 2020 | 231.115 | 152.698 | 16.977 | 0.783 |
| B2 | 2019 | 202.392 | 136.367 | 15.157 | 0.794 |
| B2 | 2020 | 193.791 | 146.804 | 14.235 | 0.848 |
| B3 | 2019 | 466.077 | 344.016 | 34.904 | -0.094 |
| B3 | 2020 | 507.595 | 392.966 | 37.286 | -0.045 |
| B4 | 2019 | 198.893 | 142.169 | 14.895 | 0.801 |
| B4 | 2020 | 171.840 | 118.192 | 12.623 | 0.880 |
| cnn1d | 2019 | 200.356 | 136.911 | 15.004 | 0.798 |
| cnn1d | 2020 | 196.368 | 148.842 | 14.424 | 0.844 |
| elastic_net | 2019 | 202.192 | 136.031 | 15.142 | 0.794 |
| elastic_net | 2020 | 193.516 | 146.511 | 14.215 | 0.848 |
| random_forest | 2019 | 202.171 | 133.903 | 15.140 | 0.794 |
| random_forest | 2020 | 188.650 | 140.431 | 13.857 | 0.856 |
| svr | 2019 | 203.442 | 138.242 | 15.235 | 0.791 |
| svr | 2020 | 195.589 | 150.138 | 14.367 | 0.845 |
| xgboost | 2019 | 212.558 | 144.852 | 15.918 | 0.772 |
| xgboost | 2020 | 187.600 | 140.982 | 13.780 | 0.857 |

**X1, modelo CNN-1D:** campo receptivo temporal de 5 pasos · predicciones válidas 100% · ejecuciones reproducibles 100% (semillas [0, 1, 2]).

**X2, SHAP:** predicciones explicadas 30 de 58 (52%). Valor SHAP absoluto medio por predictor, de mayor a menor: ndvi_max (0.146), radiacion_solar (0.120), precipitacion (0.111), temperatura_minima (0.109), ndvi_mean (0.083), temperatura_maxima (0.065).
