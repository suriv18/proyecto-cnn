# Resumen para el asesor: estado del proyecto y decisiones que necesito

Tesis: *Modelo CNN-1D interpretable con SHAP para predecir el rendimiento de quinua en la sierra peruana* (UNMSM). Reorganizada bajo CRISP-ML(Q) + DSRM. Fecha: 2026-10-10.

**Lo que pido:** cerrar las **8 decisiones** de la sección 3 para poder aprobar el gate de F2 y ejecutar el piloto oficial. Hasta entonces todo lo provisional está marcado como tal y es reversible. Además, 3 consultas de diseño (sección 4).

## 1. Estado

| Fase | Estado |
|---|---|
| F1 Comprensión del negocio y los datos | Hecha |
| F2 Preparación de datos | Hecha en código y datos; **gate abierto** a la espera de sus decisiones |
| F3 Modelado | Código del piloto completo (T1–T8); piloto oficial **bloqueado por el gate de F2** |
| F4 Evaluación / F5 Empaquetado | Fichas listas; no iniciadas |

- Unidad de análisis: provincia-campaña. Periodo propuesto 2016–2025 (10 campañas).
- Embudo de observaciones: 800 → 674 tras filtros de calidad → **601 elegibles** (≥ 80 % de celdas válidas; peor caso 86,7 %), 571 con tensor completo (sensibilidad).
- Tensor: 6 variables (precipitación, temperatura máxima y mínima, radiación solar, NDVI medio y máximo) × 5 fases BBCH, más altitud media y proporción de quinua como covariables.
- Fuentes: ESA WorldCover (cultivos), SRTM (pendiente), MODIS MOD13Q1 (NDVI), CHIRPS, ERA5-Land, SISAGRI (rendimiento).
- Control de fugas: todo lo que aprende (imputación, estandarización, línea base, fenología de respaldo) se ajusta solo con el entrenamiento; validación de origen móvil; las campañas externas no se cargan en el piloto (prueba de fuga por permutación: **pasa** en los 5 modelos). Suite de tests: 540 pasan.
- Informe de depuración y EDA: `docs/crisp/E3-informe-depuracion-eda.md`. Pre-registro en borrador: `docs/crisp/E5-preespecificacion-BORRADOR.md`. Plan y resultados del piloto: `docs/crisp/F3-plan-del-piloto.md`.

## 2. Resultados de la corrida técnica (NO VINCULANTE)

Corrida de desarrollo con datos reales de **solo 2016–2019** (un pliego, prueba 2019, 62 observaciones). No fija nada de E5 ni cuenta como evidencia.

- Predecir la **anomalía** respecto a la provincia reduce el RMSE de 389 a 205 kg/ha frente a predecir el nivel (el 90 % de la varianza es entre provincias).
- En anomalía, los 5 modelos quedan entre 199 y 213 kg/ha; la línea base B2 (media provincial) da 202,4. **Referencias simples ya igualan a los modelos:** persistencia (repetir la última campaña) 198,9 en 2019 y 166,9 en 2018 (media 185,3).
- Consecuencia para las hipótesis: con 5 campañas externas, HE1b (CNN mejor que B2) exige que la CNN gane las **5 de 5**. Las correlaciones dentro de la provincia son débiles (|r| ≤ 0,28), así que la señal interanual es modesta.
- SHAP: DeepExplainer cumple la precisión local (error 0) y coincide con KernelExplainer; la contribución por fase es casi plana, con máximo en floración.
- Costo estimado de todo el confirmatorio (30 trials y 10 semillas): ≈ 0,5 h.
- Hallazgo corregido: XGBoost era determinista y sus 10 semillas eran idénticas; se añadió submuestreo a su espacio de búsqueda (propuesto, a confirmar en el piloto).

## 3. Decisiones que necesito (estado `por_decidir_asesor`)

| # | Decisión | Propuesta actual (provisional) | Qué está en juego |
|---|---|---|---|
| 1 | Periodo de estudio | 2016–2025 (enmienda H1 aún "PROPUESTA") | Define cuántas campañas hay para entrenar y probar |
| 2 | Provincias de transición | Incluidas y marcadas (se analizan con y sin ellas como sensibilidad; evidencia SISAGRI por distrito en E3 §7) | Qué provincias quedan dentro del universo |
| 3 | Ciclos fuera de la ventana de extracción | Se excluyen (29 observaciones; ciclos calendario mayo–junio en Arequipa y La Libertad) | Pérdida de cobertura en esos departamentos |
| 4 | Altitud mínima de la máscara agrícola | 2500 m | Qué superficie se considera cultivo de quinua |
| 5 | Altitud máxima | 4000 m | Ídem |
| 6 | Pendiente máxima | 14,04° (D.S. 005-2022-MIDAGRI) | Ídem |
| 7 | Amplitud mínima de NDVI | 0,323 (percentil 5 de 2016–2020; sesgo regional en Arequipa y Puno) | Ídem; es el umbral más sensible |
| 8 | Ventana inicial de entrenamiento | 5 campañas → quedan 5 externas (2021–2025) | Poder de los contrastes: con 5 externas la prueba de Wilcoxon no puede dar p < 1/32 |

## 4. Consultas de diseño

1. **Línea base más fuerte.** Dado que la persistencia iguala a los modelos, ¿se acepta incorporarla como referencia adicional (o sustituir B2 por ella)? Cambia lo que significa "superar a B2".
2. **Identidad de la provincia.** En la representación de nivel ningún modelo recibe la identidad de la provincia y por eso no alcanza a B2. ¿Se permite una covariable de historial (p. ej. media provincial de entrenamiento) o se mantiene la representación de anomalía?
3. **Espacios de búsqueda.** El piloto marcó parámetros en el borde (dropout de la CNN; alpha de Elastic-Net; C de SVR). ¿Se amplían, siempre igual para los 5 modelos, antes de congelar E5?

## 5. Qué ocurre después de su respuesta

1. Cierro el gate de F2 con sus decisiones y repito la preparación si cambia algún umbral.
2. Ejecuto el **piloto oficial** (solo las campañas de entrenamiento) y cierro los parámetros `por_decidir_piloto` (10 pendientes).
3. Congelo E5 (el sistema bloquea el congelamiento mientras queden pendientes) y alcanzo el hito H2.
4. Solo entonces se evalúan las campañas externas y se contrastan HE1 y HE2.
