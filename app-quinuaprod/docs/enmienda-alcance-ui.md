# Enmienda de alcance: visor de resultados (app-quinuaprod)

**Estado:** BORRADOR del autor, sin revisión de terceros. No modifica la tesis ni el registro E5; propone el texto de una enmienda.

## Qué cambia

`docs/01-analisis-sistema.md` (sección de alcance, línea 202) deja fuera del proyecto la «interfaz de usuario, API pública o servicio desplegado en producción continua». RNF7 exige que toda modificación sustantiva de alcance se documente como enmienda metodológica antes del entrenamiento definitivo.

Se propone añadir **app-quinuaprod**, un visor web de **solo lectura** de los resultados del piloto, más una pantalla de predicción para observaciones ya evaluadas. Sigue sin entrar en el alcance: un servicio en producción continua, la predicción a escala parcela, las recomendaciones a productores y cualquier atribución causal.

## Lo que la aplicación no altera

- **Diseño confirmatorio.** La aplicación no entrena, no ajusta umbrales y no toca E5. Solo presenta artefactos ya generados por `service-cnn/`.
- **Campañas externas 2021-2025.** No se leen, no se exportan a los datos de la aplicación y no se muestran. Una prueba e2e verifica que ninguna pantalla de datos las muestre. El backend deberá mantener la restricción (ver `integracion-backend.md`).
- **Honestidad de los resultados.** La franja superior indica siempre el modo del piloto, si es vinculante y la versión de E5. La ejecución técnica de HE2 se rotula como no vinculante.
- **Interpretación.** La pantalla de explicabilidad advierte que SHAP describe la función aprendida, no relaciones causales, y que no existen valores SHAP por provincia ni por celda en los resultados exportados.
- **Datos personales.** No hay: la unidad es provincia-campaña (RNF5).

## Pantallas y su origen

| Pantalla | Artefacto de `service-cnn/` |
|---|---|
| Resumen y Cobertura | `reports/E1_matriz_cobertura_enmendado_2016_2025.json`, `configs/provincias.csv`, `data/processed/exclusiones.csv` |
| Predicción vs observado | `reports/piloto_oficial/predicciones.csv` |
| Indicadores (X1, X2, Y1) | `reports/piloto_oficial/p11_indicadores_y1.csv`, `resultados.json`, `p6_variabilidad_por_semilla.csv` |
| Explicabilidad | `reports/piloto_oficial/resultados.json` (p8), `reports/he2_tecnico/resultados.json` |
| Sensibilidades | `reports/sensibilidades/*.csv` |
| Registro E5 | `configs/preespecificacion.yaml` |
| Predicción interactiva | Predicciones del piloto; el tensor editado requiere un modelo serializado que hoy no existe |

## Decisiones abiertas del autor

1. ¿Se acepta ampliar el alcance a un visor de solo lectura? ¿Con qué audiencia (asesor, jurado, tomadores de decisión)?
2. ¿Puede la pantalla de predicción interactiva aceptar un tensor editado? Exige serializar el CNN-1D tras el entrenamiento definitivo y decidir cómo se versiona el modelo.
3. ¿Se mantiene la aplicación restringida a las campañas de entrenamiento y prueba del piloto hasta congelar E5 (H2)?
4. ¿Qué rotulado debe llevar un resultado definitivo cuando se evalúen las campañas externas?
