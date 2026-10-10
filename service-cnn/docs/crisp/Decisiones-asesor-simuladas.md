# Decisiones del gate de F2 en rol de asesor (SIMULADAS)

> **Esto NO es la decisión del asesor.** Es la recomendación que Claude emite al ponerse en su rol, a petición del tesista, con la evidencia del proyecto. Sirve como propuesta firme para llevar a la reunión. Mientras el asesor real no la ratifique, los 8 parámetros siguen en `por_decidir_asesor`, el gate de F2 sigue abierto y el piloto oficial sigue bloqueado.
>
> Para ratificar: el asesor (o el tesista, con su visto bueno) cambia el estado a `fijado` en `configs/preespecificacion.yaml` y se ejecuta el piloto con `--gate-f2-aprobado --ventana-confirmatoria 4`.

Fecha: 2026-10-10. Criterio general: decidir lo que se pueda **antes de ver las campañas externas**, minimizar grados de libertad del investigador y dejar todo lo discutible como análisis de sensibilidad declarado.

## Decisiones

| # | Parámetro | Decisión | Cambio |
|---|---|---|---|
| 1 | Periodo | **2016–2025** (10 campañas). Formalizar la enmienda H1 en el documento de tesis | No |
| 2 | Provincias de transición | **Incluidas y marcadas**; resultados principales con ellas y sensibilidad sin ellas | No |
| 3 | Ciclos fuera de la ventana de extracción | **Excluir** (29 observaciones, 3,6 % de 800); no re-extraer. Informar la cobertura por departamento | No |
| 4 | Altitud mínima | **2500 m** | No |
| 5 | Altitud máxima | **4000 m** | No |
| 6 | Pendiente máxima | **14,04°** (D.S. 005-2022-MIDAGRI) | No |
| 7 | Amplitud mínima de NDVI | **Percentil 5**, con sensibilidad en p1 y p10; **recalibrar solo con 2016–2019** | Recalibrar |
| 8 | Ventana inicial de entrenamiento | **4 campañas** (2016–2019); **6 externas** (2020–2025) | 5 → 4 |

## Razones

1. **Periodo.** Las cuatro fuentes cubren 2016–2025 y las 10 campañas dan la mejor relación entre cobertura y homogeneidad de la serie MODIS. Mantener 2006–2024 obligaría a otra fuente de NDVI.
2. **Transición.** Excluirlas es una decisión del investigador con consecuencias sobre el universo; incluirlas y marcarlas es transparente y permite mostrar que las conclusiones no dependen de ellas.
3. **Ciclos fuera de ventana.** Ampliar la ventana obliga a re-extraer y a mover la definición de campaña después de haber visto los datos. Se pierden 29 observaciones; el riesgo es que la pérdida se concentre en Arequipa y La Libertad, por lo que se informa la cobertura por departamento.
4. **Umbrales de altitud y pendiente.** Tienen respaldo agronómico y normativo, no se ajustaron con el rendimiento y deben congelarse. Cambiarlos después de ver resultados sería un grado de libertad ilegítimo.
5. **Amplitud de NDVI.** El percentil 5 es un criterio, no un valor mágico. Se congela el percentil y se reporta p1 y p10. Con ventana 4, la campaña 2020 pasa a ser externa, así que el umbral (hoy 0,323, calibrado con 2016–2020) debe recalcularse solo con 2016–2019 antes del piloto oficial. Tarea abierta, no ejecutada.
6. **Ventana.** Con 5 externas el p mínimo de Wilcoxon es 1/32 = 0,031, que no sobrevive a Holm con 2 contrastes (0,05/2 = 0,025). Con 6 externas es 1/64 = 0,016. El piloto técnico indicó además que más campañas de entrenamiento no mejoran el error (P10, un pliego), de modo que sacrificar una de entrenamiento cuesta poco. **Contrapartida:** HE1b por bootstrap pasa a exigir que la CNN gane las 6 campañas (con 5 exigía 5 de 5, verificado con `evaluate_superiority`), de modo que ganar poder en Wilcoxon no hace más fácil confirmar HE1b. Además la corrección de Holm figura en la tesis (3.3.5) pero **no está implementada en el código**: la familia exacta de contrastes debe definirse e implementarse antes de congelar.

## CORRECCIÓN (2026-10-10, al implementar Holm): el argumento de la decisión 8 estaba mal

La razón nº 6 decía que con 5 externas el p mínimo de Wilcoxon (1/32) «no sobrevive a Holm con 2 contrastes». **Es incorrecto.** Revisado el E5 (3.3.5):
- **HE1a y HE1b forman una conjunción** (HE1 se respalda si se cumplen ambas): cada una se contrasta al nivel 0,05 sin corrección. Con 5 externas, 1/32 = 0,031 < 0,05 es alcanzable.
- **Holm solo corresponde a la familia de HE2** (estabilidad, fidelidad, concordancia), y esos tres criterios hoy son **reglas con umbral, sin valor p**; no hay nada que corregir hasta que se defina un p para cada uno.

Lo que queda a favor de la ventana de 4 es poco (un bloque más para el bootstrap y p mínimo 1/64). Lo que queda en contra es más: HE1b exige 6 de 6 en vez de 5 de 5 (más difícil si la CNN solo iguala a B2, como en el piloto técnico), B2 con tendencia lineal necesita ≥ 5 campañas de historia y con ventana 4 la primera campaña externa (2020) cae a la media, y obliga a recalibrar la máscara. **La decisión 8 queda abierta; con este argumento corregido recomendaría volver a la ventana de 5.** No se revierte en los archivos hasta que el tesista lo confirme.

## Consultas de diseño

| Consulta | Decisión |
|---|---|
| Línea base más fuerte | **Mantener B2 como referencia confirmatoria** (es la definida en la tesis y en la Tabla 8). **Añadir la persistencia como referencia secundaria B4**, reportada de forma descriptiva y fuera de la familia de Holm. Pendiente de implementar y registrar en E5 |
| Identidad de la provincia | **Mantener la anomalía como representación principal**: al restar la media provincial de entrenamiento la identidad ya está implícita. No añadir covariable de historial. Nivel queda como sensibilidad |
| Espacios de búsqueda | **No se ensanchan ahora.** Lo decide el piloto oficial (P9), una sola vez, igual para los 5 modelos, y luego se congela. Prioridad: el dropout de la CNN, cuyo rango actual [0,001; 0,01] es prácticamente cero |

## Tareas que se derivan (no ejecutadas)

1. ~~Recalibrar la amplitud de NDVI con 2016–2019~~ **Hecho** (mismas celdas y semilla, sin red): p5 = **0,319** (antes 0,323; p1 0,246; p10 0,359; p25 0,417). Los datos extraídos usan 0,323. **Decidir:** re-extraer el NDVI con 0,319 (horas de GEE) o declarar la desviación de 0,004, que es un 1,2 % del umbral y cae dentro de la sensibilidad p1–p10. Recomendación como asesor: declarar la desviación y no re-extraer, salvo que la sensibilidad muestre sensibilidad al umbral.
2. Re-ejecutar `run_preprocessing.py` solo si se re-extrae.
3. Implementar B4 (persistencia) con pruebas y registrarla en E5.
4. Actualizar las referencias a «5 externas, 2021–2025» en el resumen para el asesor, el plan del piloto y las fichas.
