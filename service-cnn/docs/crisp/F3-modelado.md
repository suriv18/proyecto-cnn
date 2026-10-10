# F3 · Modelado

- **DSRM**: A3 (diseño y desarrollo) y A4 (demostración).
- **Cronograma**: Act. 6 CNN-1D, comparadores y SHAP (oct→nov 2026); Act. 7 ensayo piloto y preespecificación (nov 2026).
- **Entregables**: E4 (código y configuración reproducible del artefacto), H2 (hito), E5 (registro fechado de hipótesis, márgenes y reglas).
- **Secciones de la tesis**: 4.12, Tabla 8, 4.15 (parte previa), 4.18.
- **Estado de la fase**: código implementado con datos sintéticos. **Gate H2: bloqueado por F2.**

## Requisitos y métricas de calidad

| ID | Requisito / métrica | Estado | Evidencia |
|---|---|---|---|
| Q-F3-1 | Cinco modelos ajustables (CNN-1D, Elastic-Net, RF, XGBoost, SVR) + tres benchmarks B1–B3 | Parcial | `src/models/{cnn1d,classic_models,benchmarks}.py` con tests sintéticos |
| Q-F3-2 | Presupuesto de ajuste igual (mismo nº máximo de evaluaciones) para los cinco modelos; espacios de búsqueda publicados | Parcial | `src/models/tuning*.py`, `configs/hyperparams.yaml`; falta verificar igualdad y fijarlo en E5 |
| Q-F3-3 | Validación de origen móvil con anidamiento; ventana inicial mínima definida en el piloto | Parcial | `src/evaluation/splits.py`; ventana inicial sin definir (depende de N5) |
| Q-F3-4 | Diez semillas solo para modelos estocásticos; agregación por campaña | Parcial | Por conectar en `run_experiment.py` (placeholder) |
| Q-F3-5 | Variante SHAP, escala de salida, conjunto de referencia y tratamiento de correlación **registrados antes** de la evaluación (4.15) | Pendiente | `src/explainability/shap_explainer.py` implementado; decisión no registrada |
| Q-F3-6 | Forma de la línea base (lineal o suavizada) elegida en el piloto sin campañas externas (4.11) | Pendiente | — |
| Q-F3-7 | E5: hipótesis, margen δ = 0,05, comparador XGBoost, umbrales HE2 (W ≥ 0,70; J ≥ 0,50), horizontes, espacios, reglas de depuración, con fecha | **Pendiente** | No existe |
| Q-F3-8 | Estimación de variabilidad en el piloto y de potencia con el nº real de campañas externas | Pendiente | — |

## Riesgos

| ID | Riesgo | Estado |
|---|---|---|
| R-F3-1 | Sobreajuste de CNN-1D con pocas campañas (Sabo et al., 2023); el resultado puede ser desfavorable a la CNN | Latente; la hipótesis es informativa en cualquier desenlace |
| R-F3-2 | Selección retrospectiva del comparador o ajuste asimétrico | Mitigado por XGBoost preespecificado e igual presupuesto; **solo si se registra en E5** |
| R-F3-3 | Piloto contamina la evaluación definitiva (uso de campañas externas) | Latente |
| R-F3-4 | Modelo entrenado antes de cerrar F2 sobre datos provisionales | **Activo si se avanza sin gate F2** |

## Tareas de control

| Control | Mitiga | Acción |
|---|---|---|
| C-F3-1 | R-F3-4 | No ejecutar piloto con datos reales hasta cerrar el gate F2 |
| C-F3-2 | R-F3-3 | Definir campañas reservadas para la evaluación definitiva antes del piloto y bloquearlas por código |
| C-F3-3 | Q-F3-2 | Test que verifique igualdad del nº de configuraciones entre los cinco modelos |
| C-F3-4 | Q-F3-7 | Redactar E5 (`docs/crisp/E5-preespecificacion.md`) con fecha y commit hash; congelarlo antes de F4 |
| C-F3-5 | Q-F3-8 | Piloto en campañas de entrenamiento; reportar variabilidad por semilla |

## Criterio de gate H2

Piloto ejecutado de extremo a extremo con datos reales; E4 reproducible (`dvc repro` hasta el modelado); E5 firmado y fechado **antes** de ver resultados de evaluación; ventana inicial y campañas externas fijadas.
