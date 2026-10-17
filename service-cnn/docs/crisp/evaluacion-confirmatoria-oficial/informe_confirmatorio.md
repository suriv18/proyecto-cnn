# Evaluación confirmatoria (oficial)

> Evaluación confirmatoria oficial: E5 congelada, campañas externas evaluadas una sola vez.

## Puerta de entrada
- E5: CONGELADO v1.1 (congelada el 2026-10-17, commit `47c43a7`).
- SHA-256 de `tensores.npz`: `15d7f9446cb3717e…` (coincide con E5).
- Campañas cargadas: [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025] · descartadas al cargar: ninguna · evaluadas como externas: [2021, 2022, 2023, 2024, 2025].
- Parámetros (de E5): ventana inicial 5, 30 trials por modelo, 10 semillas, 47 épocas, representación `anomalia`, δ = 0.05.

## B*
Elegida con la validación interna del pliego inicial (solo entrenamiento): RMSE de B2 193.8 kg/ha, de B4 179.2 kg/ha → **B4**.

## HE1a · No inferioridad (δ = 5%) — NO RESPALDADA
Regla exacta de signos: q < δ en todas las campañas (nivel 0.031).

| Campaña | q frente a XGBoost | q frente a B4 | Skill score frente a B4 |
|---|---|---|---|
| 2021 | +0.035 | +0.143 | -0.305 |
| 2022 | -0.019 | +0.034 | -0.069 |
| 2023 | -0.008 | +0.059 | -0.121 |
| 2024 | -0.007 | -0.017 | +0.034 |
| 2025 | -0.022 | -0.038 | +0.074 |

No inferior a XGBoost: sí · a B4: no. HE1b (exploratoria): ganancia en 2 de 5 campañas (p de signos 0.812).

## HE2c · Concordancia agronómica — NO RESPALDADA
Binomial exacta, p = 0.3174. Fase de máxima contribución por campaña:

| Campaña | Fase máxima |
|---|---|
| 2021 | llenado_grano |
| 2022 | llenado_grano |
| 2023 | floracion |
| 2024 | desarrollo_vegetativo |
| 2025 | emergencia |

Proporción de contribución del referente agronómico: 0.426 (supera el azar: sí). Secundarias descriptivas: W de Kendall 0.49, Jaccard 0.80.

## Hipótesis general (conjuntiva: HE1 y HE2)
**NO RESPALDADA**.
