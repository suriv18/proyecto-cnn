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
| Q-F3-2 | Presupuesto de ajuste igual (mismo nº máximo de evaluaciones) para los cinco modelos; espacios de búsqueda publicados | Parcial | `src/models/tuning.py` (30 trials para los 5) y `configs/hyperparams.yaml`; la **igualdad queda verificada automáticamente** contra E5 (`check_consistency_with_code`). **Costo medido (2026-10-10, corrida técnica no vinculante): ≈ 0,5 h para todo el confirmatorio con 30 trials y 10 semillas**; falta revisar los espacios (P9) |
| Q-F3-3 | Validación de origen móvil con anidamiento; ventana inicial mínima definida en el piloto | Parcial | `src/evaluation/splits.py`; ventana inicial sin definir (depende de N5) |
| Q-F3-4 | Diez semillas solo para modelos estocásticos; agregación por campaña | Parcial | `experiment_runner.run_fold` (semillas solo en CNN-1D, Random Forest y XGBoost; mediana por campaña con `rmse_per_campaign`; 26 pruebas). Falta conectarlo al CLI (T7) |
| Q-F3-5 | Variante SHAP, escala de salida, conjunto de referencia y tratamiento de correlación **registrados antes** de la evaluación (4.15) | **Borrador (2026-10-10)** | Registrados como parámetros de E5 (`interpretabilidad.*`): conjunto de referencia propuesto (≤ 100 obs. de entrenamiento); variante, escala y tolerancia `por_decidir_piloto` (P8) |
| Q-F3-6 | Forma de la línea base (lineal o suavizada) y representación nivel/anomalía elegidas en el piloto sin campañas externas (4.11) | **Plan listo (2026-10-10)** | `docs/crisp/F3-plan-del-piloto.md` P4 y P5, con regla de decisión preespecificada (empate → nivel / lineal). Pendiente ejecutar |
| Q-F3-7 | E5: hipótesis, margen δ = 0,05, comparador XGBoost, umbrales HE2 (W ≥ 0,70; J ≥ 0,50), horizontes, espacios, reglas de depuración, con fecha | **Borrador (2026-10-10)** | `configs/preespecificacion.yaml` (fuente única, 74 parámetros: 41 fijados por la tesis, 15 propuestos, 8 con el asesor, 9 con el piloto) y `docs/crisp/E5-preespecificacion-BORRADOR.md` generado con `scripts/render_e5.py`. `preregistration.check_consistency_with_code` detecta divergencias con el código (13 tests) y `assert_can_freeze` **bloquea el congelamiento** mientras quede algo pendiente |
| Q-F3-8 | Estimación de variabilidad en el piloto y de potencia con el nº real de campañas externas | **Plan listo (2026-10-10)** | P6 (10 semillas) y P10 (curva de aprendizaje); potencia ya cuantificada en F4 (R-F4-1: 5 campañas externas con ventana inicial de 5) |
| Q-F3-9 | Plan del ensayo piloto con reglas de decisión fijadas antes de ejecutar y campañas externas protegidas | **Hecho (2026-10-10)** | `docs/crisp/F3-plan-del-piloto.md`: datos solo 2016–2020 (302 obs.), pliegues P-A (prueba 2019) y P-B (prueba 2020), 12 experimentos P0–P11 con regla de decisión, guarda técnica T1 y criterios de salida de H2 |

## Riesgos

| ID | Riesgo | Estado |
|---|---|---|
| R-F3-1 | Sobreajuste de CNN-1D con pocas campañas (Sabo et al., 2023); el resultado puede ser desfavorable a la CNN | Latente; la hipótesis es informativa en cualquier desenlace |
| R-F3-2 | Selección retrospectiva del comparador o ajuste asimétrico | Mitigado por XGBoost preespecificado e igual presupuesto; **solo si se registra en E5** |
| R-F3-3 | Piloto contamina la evaluación definitiva (uso de campañas externas) | Latente |
| R-F3-4 | Modelo entrenado antes de cerrar F2 sobre datos provisionales | **Activo si se avanza sin gate F2** |
| R-F3-5 | **Early stopping de la CNN-1D con partición aleatoria** del 20 % (brecha G1) y devolución del modelo de la última época (G2): contradicen 4.12.2 y sobreestiman la capacidad de generalización | **Cerrado en código (2026-10-10)**: `train_cnn1d(..., grupos_temporales=)` valida con la última campaña del entrenamiento y restaura la mejor época por defecto (8 pruebas). Falta usarlo en el ejecutor (T4) |
| R-F3-6 | **B2 coincide con B1 con poca historia**: con menos de 5 campañas por provincia la línea base cae a la media provincial; con 5 (la primera campaña externa) la tendencia lineal se ajusta con 5 puntos y es muy ruidosa. Además, en nivel los modelos no reciben la identidad de la provincia y no alcanzan a B2 | **Detectado (2026-10-10)** con la corrida técnica; P5 rediseñado y `historia_minima_para_tendencia_campanas` registrado en E5 (coherente con el código). Decisión del piloto y, si hace falta, del asesor |

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
