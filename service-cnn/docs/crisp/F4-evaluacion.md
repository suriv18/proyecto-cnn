# F4 · Evaluación

- **DSRM**: A5 (evaluación).
- **Cronograma**: Act. 8 evaluación predictiva (nov→dic 2026); Act. 9 evaluación SHAP (dic 2026).
- **Entregables**: predicciones y métricas por campaña, modelo y semilla (producto intermedio); E6 (informe de interpretabilidad).
- **Secciones de la tesis**: 3.3, 4.12–4.16, Tablas 9–10.
- **Estado de la fase**: código con datos sintéticos. **Gate: bloqueado por H2.**

## Requisitos y métricas de calidad

| ID | Requisito / métrica | Estado | Evidencia |
|---|---|---|---|
| Q-F4-1 | RMSE, MAE, rRMSE, R² fuera de muestra por campaña, departamento, horizonte y modelo, sobre rendimiento reconstruido (kg/ha) | Parcial | `src/evaluation/metrics.py` |
| Q-F4-2 | HE1a: límite superior unilateral IC 95 % de mediana(q_f) < 0,05 (no inferioridad vs XGBoost) | Parcial | `src/evaluation/hypothesis_tests.py` (bootstrap por bloques, Wilcoxon, Hodges-Lehmann); **corregido (2026-10-10)** el Wilcoxon de apoyo, que contrastaba superioridad (p = 1,0 para una CNN dentro del margen) en vez de q_f frente a δ; `he1_evaluation.evaluate_he1` combina HE1a y HE1b y informa el menor p posible |
| Q-F4-3 | HE1b: límite superior de mediana(Δ_f) < 0 frente a B2 | Parcial | idem |
| Q-F4-4 | HE2a estabilidad: W de Kendall ≥ 0,70 y Jaccard ≥ 0,50 | Parcial | `src/explainability/stability.py` |
| Q-F4-5 | HE2b fidelidad por oclusión (D_f > 0, con Holm) | Parcial | `src/explainability/fidelity.py` |
| Q-F4-6 | HE2c concordancia: fase máxima ∈ {floración, llenado de grano} | Parcial | `src/explainability/agronomic_concordance.py` |
| Q-F4-7 | Sensibilidad: horizonte, nivel vs anomalía, participación de quinua, alineación, fuente climática, campañas extremas | Pendiente | — |
| Q-F4-8 | Robustez espacial (dejar un departamento fuera) y validación complementaria (dejar una campaña fuera) | Pendiente | — |
| Q-F4-9 | Orquestación real | **Pendiente** | `run_experiment.py` y `run_shap.py` son placeholders |

## Riesgos

| ID | Riesgo | Estado |
|---|---|---|
| R-F4-1 | Pocos bloques temporales → baja potencia | **Cuantificado (2026-10-10)**: con N5 = 601 y una ventana inicial de 5 campañas (2016–2020) quedan **5 campañas externas** (2021–2025), con 302–543 observaciones de entrenamiento antes de cada una. Con n = 5, el menor valor p posible del Wilcoxon unilateral es 1/32 ≈ 0,031 (solo se alcanza si las 5 campañas favorecen al mismo modelo) y el IC por bloques será muy ancho. Con ventana inicial de 4 serían 6 campañas externas y 49–109 observaciones de entrenamiento al inicio. **Decisión del asesor/piloto (H2)**: ventana inicial, y reportar la decisión de HE1a/HE1b con magnitud de efecto y no solo con el valor p. **Verificado con `evaluate_superiority` (2026-10-10): con 5 campañas externas HE1b solo se sostiene si las 5 favorecen a la CNN (con 4 de 5, el límite superior del IC95 sigue en positivo); con 6 campañas, 6 de 6.** Holm (3.3.5) se implementó (`src/evaluation/multiplicity.py`, probada contra statsmodels) y se definieron propuestas de valor p para los 3 criterios de HE2 (`src/explainability/he2_pvalues.py`; estado `propuesto` en E5): permutación para estabilidad, aleatorización exacta por rangos para fidelidad y binomial exacta para concordancia. Con 5 campañas, HE2 solo se respalda si las 3 pasan con Holm: concordancia exige 5/5 aciertos (p = 0,0102) y fidelidad 5/5 con la fase alta en primer lugar (p = 0,0003). El umbral J ≥ 0,50 se alcanza por azar en el 22 % de los casos (Jaccard medio bajo H0 = 0,46), por lo que el p de permutación es el criterio informativo. Con una regla tan estricta, un empate o una sola campaña desfavorable basta para no respaldar HE1. **Recomendación en rol de asesor (sin ratificar): mantener ventana 5**; se evaluó 4 y se descartó (docs/crisp/Decisiones-asesor-simuladas.md) |
| R-F4-2 | HE1 y HE2 son conjuntivas: un fracaso en un componente invalida la hipótesis | Aceptado; reportar con honestidad |
| R-F4-3 | Multiplicidad (Holm solo para HE2; comparaciones entre algoritmos exploratorias) | Mitigado por diseño |
| R-F4-4 | SHAP interpretado como causalidad | Mitigado con terminología "contribución predictiva" |
| R-F4-5 | Fuga por transformaciones o respaldo fenológico | Depende de C-F2-4 |

## Tareas de control

| Control | Mitiga | Acción |
|---|---|---|
| C-F4-1 | R-F4-1 | Reportar potencia/IC con el nº real de campañas externas; si resulta inviable, enmienda previa (no ajustar umbrales tras ver resultados) |
| C-F4-2 | R-F4-5 | Auditoría de fuga: test que reentrene con campañas barajadas y verifique que el rendimiento colapsa |
| C-F4-3 | Q-F4-9 | Implementar orquestación; guardar predicciones y atribuciones por campaña, modelo y semilla |
| C-F4-4 | R-F4-2 | Reglas de decisión copiadas literalmente de E5; script de decisión automática sin ajuste manual |

## Criterio de gate F4

Resultados completos y reproducibles (`dvc repro train_eval explain`); E6 con estabilidad, fidelidad y concordancia; decisiones HE1/HE2 tomadas con las reglas de E5; amenazas a la validez (Tabla 10) revisadas con evidencia.

## Actualización: HE1 y HE2 definitivos (tesis v3, decisión del autor)

- **HE1a (confirmatoria):** la CNN-1D no es inferior a XGBoost ni a la mejor línea base B* (B2 o B4, elegida con la validación interna) si la diferencia relativa del RMSE es menor que 5 % en **todas** las campañas externas (regla exacta de signos, nivel 0,5ⁿ = 0,031 con 5 campañas). **HE1b (exploratoria):** skill score de Murphy frente a B* y número de campañas con ganancia; no condiciona la decisión. Implementado en `src/evaluation/he1_evaluation.py`.
- **HE2:** se respalda si se cumple la **concordancia agronómica** (binomial exacta, criterio primario, sin corrección). Fidelidad (aleatorización por rangos de la oclusión de cada fase) y estabilidad (permutación) son secundarias, con Holm entre ellas, y no condicionan la decisión. Implementado en `src/evaluation/he2_evaluation.py`.
- Reemplaza lo indicado antes sobre Holm con tres contrastes y sobre «HE1b confirmatoria». Pendiente: que el asesor lo revise. Si no lo ratifica, la alternativa es volver a la superioridad frente a B2 y a los tres criterios de HE2 con Holm.
