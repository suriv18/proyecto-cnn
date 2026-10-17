# Evaluación confirmatoria oficial (2026-10-17)

Registro de la única ejecución oficial de la evaluación confirmatoria de la tesis. **No se editan los archivos de esta carpeta**: son el registro de lo que salió. Si algo debe corregirse, se agrega una nota aparte con fecha; los archivos originales no cambian. `MANIFIESTO.sha256` guarda la huella de cada uno y una prueba del CI (`tests/evaluation/test_evaluacion_oficial_integridad.py`) la comprueba.

## Resultado

La hipótesis general **no se respalda**; ver el informe completo en [`informe_confirmatorio.md`](informe_confirmatorio.md).

| Contraste | Regla (E5) | Resultado |
|---|---|---|
| HE1a, frente a XGBoost | q < 5 % en las 5 campañas externas | Se cumple en las 5 |
| HE1a, frente a B* = B4 | q < 5 % en las 5 campañas externas | No se cumple (2021, 2022 y 2023 quedan por encima) |
| HE1a | conjunción de los dos anteriores | **No respaldada** |
| HE2c | binomial exacta sobre la fase de máxima contribución | **No respaldada** (3 de 5 campañas en el referente; p = 0,317) |
| Hipótesis general | conjunción de HE1 y HE2 | **No respaldada** |

## Procedencia

| Qué | Valor |
|---|---|
| Preespecificación | E5 v1.1, congelada el 2026-10-17 (commit base `47c43a7`) |
| Código ejecutado | commit `085d4d7` de la rama `developer` |
| Datos | `data/processed` no exploratorio; SHA-256 de `tensores.npz`: `15d7f9446cb3717e72514432efc223bb41d5732a7107a7c30b79b61947ee4f44` (el de E5) |
| Comando | `python scripts/run_confirmatorio.py --modo oficial --confirmo-abrir-campanas-externas` |
| Parámetros | los de E5: ventana inicial 5, 30 trials por modelo, 10 semillas, 47 épocas, representación `anomalia`, δ = 0,05 |
| Campañas | 2016-2020 de entrenamiento; 2021-2025 externas (60 observaciones por campaña, 61 en 2021 y 58 en 2025) |
| B\* | B4, elegida con la validación interna del pliego inicial (RMSE de B2 193,8 y de B4 179,2 kg/ha), antes de mirar las campañas externas |

Se ejecutó una sola vez, sin ajuste manual. Ningún umbral, margen ni comparador cambió después de ver los resultados.

## Archivos

| Archivo | Contenido |
|---|---|
| `informe_confirmatorio.md` | Informe generado: puerta de entrada, B\*, HE1a, HE2c y decisión |
| `resultados.json` | Todos los estadísticos de HE1 y HE2, parámetros y verificaciones |
| `rmse_por_campana.csv` | RMSE (kg/ha) por modelo y campaña externa; las semillas se resumen con la mediana |
| `predicciones.csv` | Predicción por observación, modelo y semilla |
| `ajuste.csv` | Presupuesto ejecutado, mejores hiperparámetros y tiempos por modelo y pliego |
| `shap_contribucion_por_fase.csv` | Contribución SHAP absoluta por fase y campaña (promedio de las semillas) |
| `oclusion_aumento_rmse_por_fase.csv` | Aumento de RMSE al ocluir cada fase, por campaña |
| `ejecucion_confirmatoria.json` | Marca de control de la corrida (ver la nota siguiente) |
| `MANIFIESTO.sha256` | SHA-256 de cada archivo anterior |

## Nota sobre `ejecucion_confirmatoria.json`

Los campos `inicio` y `fin` muestran el mismo instante (23:07:23 UTC). El código escribió `inicio` con la fecha del final de la corrida y sobrescribió la hora real de arranque, que fue 22:30 UTC. Es un defecto de registro del control de ejecución; no afecta a ningún resultado. El archivo se conserva tal como salió.

## Cómo comprobar la integridad

Desde `service-cnn`:

```sh
cd docs/crisp/evaluacion-confirmatoria-oficial && sha256sum -c MANIFIESTO.sha256
```

La misma comprobación corre en `pytest tests/evaluation/test_evaluacion_oficial_integridad.py`.
