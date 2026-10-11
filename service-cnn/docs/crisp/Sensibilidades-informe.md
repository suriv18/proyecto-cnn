# Sensibilidades sobre las campañas de entrenamiento (exploratorio, 2026-10-10)

Generado con `scripts/run_sensibilidades.py --ventana-confirmatoria 5`. Los conjuntos alternos se produjeron con `scripts/run_preprocessing.py` cambiando un solo parámetro: `--max-dias-arrastre-ndvi 0` y `--excluir-por-altitud`.

## Lectura prudente

Dos pliegos de prueba (2019 y 2020) y campañas de entrenamiento solamente: **no deciden ninguna hipótesis**.

1. **La conclusión base es robusta al preprocesamiento.** Sin arrastre de NDVI o con exclusión por altitud, el RMSE de la CNN-1D cambia menos de 0,5 %. La imputación limitada de NDVI y la exclusión por altitud no condicionan los resultados.
2. **El horizonte secundario cuesta casi nada.** Con solo las tres primeras fases (hasta la floración) la CNN-1D da 205,0 y 195,3 kg/ha frente a 200,4 y 196,4 con las cinco fases (+2,3 % y −0,6 %), y XGBoost y Random Forest muestran lo mismo. Las fases tardías aportan poca señal, coherente con la contribución SHAP casi plana. Un pronóstico al cierre de la floración conserva la exactitud y llega antes.
3. **La conclusión depende de la composición de la muestra.** Sin las provincias de transición, B2 y los demás modelos mejoran en 2019 (de 202 a 185 kg/ha), pero la CNN-1D no (200,5): queda +8,4 % sobre B2, por encima del margen de 5 %. Con solo tensores completos, todos los modelos y las líneas base empeoran por igual (RMSE de 276 en 2019) porque cambia la población, no el modelo; el `q` frente a B2 se mantiene en 0,2 % y 2,9 %.
4. **Frente a B4 la CNN-1D no cumple en 2020 en ninguna variante** (+13,6 % a +30,8 %), igual que en la base (+14,3 %).
5. **Por participación de quinua** (terciles de `prop_quinua`) ningún modelo mejora a las líneas base en ningún estrato; el estrato medio es el de mayor error para todos (224 a 230 kg/ha) y la persistencia (B4) es la mejor donde la participación es alta (147 kg/ha).

**Sensibilidades no ejecutadas:** fuente climática, percentiles p1 y p10 de la máscara y ventana de extracción ampliada (requieren re-extraer de Earth Engine); campañas extremas (las relevantes son externas); calidad de la alineación (las 601 observaciones se alinearon por el nivel primario, sin variación).

---


Campañas 2016-2020; pliegos de prueba 2019 y 2020. Mismo protocolo del piloto oficial (anomalía, 30 trials para los 5 modelos, 10 semillas). **No usa campañas externas y no decide ninguna hipótesis.**

## RMSE (kg/ha) por variante, modelo y campaña de prueba
| variante | modelo | 2019 | 2020 |
|---|---|---|---|
| base | B2 | 202.4 | 193.8 |
| base | B4 | 198.9 | 171.8 |
| base | cnn1d | 200.4 | 196.4 |
| base | elastic_net | 202.2 | 193.5 |
| base | random_forest | 202.2 | 188.6 |
| base | svr | 203.4 | 195.6 |
| base | xgboost | 212.6 | 187.6 |
| con_exclusion_por_altitud | B2 | 204.1 | 195.5 |
| con_exclusion_por_altitud | B4 | 200.6 | 173.3 |
| con_exclusion_por_altitud | cnn1d | 200.3 | 197.3 |
| con_exclusion_por_altitud | elastic_net | 203.9 | 195.1 |
| con_exclusion_por_altitud | random_forest | 202.8 | 190.2 |
| con_exclusion_por_altitud | svr | 204.6 | 197.5 |
| con_exclusion_por_altitud | xgboost | 208.2 | 192.2 |
| horizonte_temprano | B2 | 202.4 | 193.8 |
| horizonte_temprano | B4 | 198.9 | 171.8 |
| horizonte_temprano | cnn1d | 205.0 | 195.3 |
| horizonte_temprano | elastic_net | 201.3 | 192.5 |
| horizonte_temprano | random_forest | 201.6 | 190.2 |
| horizonte_temprano | svr | 204.0 | 194.0 |
| horizonte_temprano | xgboost | 206.3 | 191.1 |
| sin_arrastre_ndvi | B2 | 202.4 | 193.8 |
| sin_arrastre_ndvi | B4 | 198.9 | 171.8 |
| sin_arrastre_ndvi | cnn1d | 200.7 | 196.5 |
| sin_arrastre_ndvi | elastic_net | 200.9 | 194.1 |
| sin_arrastre_ndvi | random_forest | 202.8 | 190.4 |
| sin_arrastre_ndvi | svr | 203.1 | 193.5 |
| sin_arrastre_ndvi | xgboost | 211.3 | 190.9 |
| sin_transicion | B2 | 185.0 | 195.3 |
| sin_transicion | B4 | 183.8 | 168.7 |
| sin_transicion | cnn1d | 200.5 | 197.7 |
| sin_transicion | elastic_net | 184.9 | 194.8 |
| sin_transicion | random_forest | 185.3 | 192.8 |
| sin_transicion | svr | 184.0 | 192.4 |
| sin_transicion | xgboost | 195.6 | 192.8 |
| tensor_completo | B2 | 275.7 | 243.7 |
| tensor_completo | B4 | 275.6 | 191.7 |
| tensor_completo | cnn1d | 276.4 | 250.8 |
| tensor_completo | elastic_net | 275.7 | 243.7 |
| tensor_completo | random_forest | 275.1 | 248.1 |
| tensor_completo | svr | 281.3 | 248.4 |
| tensor_completo | xgboost | 275.4 | 243.3 |

## CNN-1D frente a las líneas base, por variante
`q_frente_a_*` es RMSE(CNN-1D)/RMSE(referencia) − 1; `cambio_vs_base_pct` es el cambio del RMSE respecto de la variante base.

| variante | campana_prueba | rmse | q_frente_a_b2 | q_frente_a_b4 | cambio_vs_base_pct |
|---|---|---|---|---|---|
| base | 2019 | 200.356 | -0.010 | 0.007 | 0.000 |
| base | 2020 | 196.368 | 0.013 | 0.143 | 0.000 |
| sin_transicion | 2019 | 200.497 | 0.084 | 0.091 | 0.070 |
| sin_transicion | 2020 | 197.657 | 0.012 | 0.172 | 0.656 |
| tensor_completo | 2019 | 276.374 | 0.002 | 0.003 | 37.941 |
| tensor_completo | 2020 | 250.814 | 0.029 | 0.308 | 27.727 |
| horizonte_temprano | 2019 | 205.034 | 0.013 | 0.031 | 2.335 |
| horizonte_temprano | 2020 | 195.267 | 0.008 | 0.136 | -0.561 |
| sin_arrastre_ndvi | 2019 | 200.656 | -0.009 | 0.009 | 0.150 |
| sin_arrastre_ndvi | 2020 | 196.509 | 0.014 | 0.144 | 0.072 |
| con_exclusion_por_altitud | 2019 | 200.318 | -0.018 | -0.002 | -0.019 |
| con_exclusion_por_altitud | 2020 | 197.269 | 0.009 | 0.139 | 0.459 |

## Error por estrato de participación de la quinua (variante base; terciles de `prop_quinua`)
| modelo | bajo | medio | alto |
|---|---|---|---|
| B2 | 187.4 | 226.3 | 174.2 |
| B4 | 194.3 | 207.1 | 147.1 |
| cnn1d | 189.5 | 223.6 | 176.9 |
| xgboost | 188.5 | 229.9 | 178.9 |
