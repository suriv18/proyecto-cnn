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
| Q-F4-2 | HE1a: límite superior unilateral IC 95 % de mediana(q_f) < 0,05 (no inferioridad vs XGBoost) | Parcial | `src/evaluation/hypothesis_tests.py` (bootstrap por bloques, Wilcoxon, Hodges-Lehmann) |
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
| R-F4-1 | Pocos bloques temporales (≤ 10 campañas, menos externas) → baja potencia; IC por bloques inestables | **Alto**, consecuencia de la enmienda 2016–2025 |
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
