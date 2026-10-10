# Decisiones del gate de F2 en rol de asesor (SIMULADAS)

> **Esto NO es la decisión del asesor.** Es la recomendación que Claude emite al ponerse en su rol, a petición del tesista, con la evidencia del proyecto. Sirve como propuesta firme para llevar a la reunión. Mientras el asesor real no la ratifique, los 8 parámetros siguen en `por_decidir_asesor`, el gate de F2 sigue abierto y el piloto oficial sigue bloqueado.
>
> Para ratificar: el asesor (o el tesista, con su visto bueno) cambia el estado a `fijado` en `configs/preespecificacion.yaml` y se ejecuta el piloto con `--gate-f2-aprobado --ventana-confirmatoria 5`.

Fecha: 2026-10-10. Criterio general: decidir lo que se pueda **antes de ver las campañas externas**, minimizar grados de libertad del investigador y dejar todo lo discutible como análisis de sensibilidad declarado.

## Decisiones

| # | Parámetro | Decisión | Cambio |
|---|---|---|---|
| 1 | Periodo | **2016–2025** (10 campañas). **Confirmado por el autor (2026-10-10).** La enmienda H1 ya está en la tesis v3 | No |
| 2 | Provincias de transición | **Incluidas y marcadas**; resultados principales con ellas y sensibilidad sin ellas | No |
| 3 | Ciclos fuera de la ventana de extracción | **Excluir** (29 observaciones, 3,6 % de 800); no re-extraer. Informar la cobertura por departamento | No |
| 4 | Altitud mínima | **2500 m** | No |
| 5 | Altitud máxima | **4000 m** | No |
| 6 | Pendiente máxima | **14,04°** (D.S. 005-2022-MIDAGRI) | No |
| 7 | Amplitud mínima de NDVI | **Percentil 5**, con sensibilidad en p1 y p10; **recalibrar solo con 2016–2019** | Recalibrar |
| 8 | Ventana inicial de entrenamiento | **5 campañas** (2016–2020); **5 externas** (2021–2025). Se evaluó 4 y se descartó (ver corrección) | No (se revierte la propuesta intermedia de 4) |

## Razones

1. **Periodo.** Las cuatro fuentes cubren 2016–2025 y las 10 campañas dan la mejor relación entre cobertura y homogeneidad de la serie MODIS. Mantener 2006–2024 obligaría a otra fuente de NDVI.
2. **Transición.** Excluirlas es una decisión del investigador con consecuencias sobre el universo; incluirlas y marcarlas es transparente y permite mostrar que las conclusiones no dependen de ellas.
3. **Ciclos fuera de ventana.** Ampliar la ventana obliga a re-extraer y a mover la definición de campaña después de haber visto los datos. Se pierden 29 observaciones; el riesgo es que la pérdida se concentre en Arequipa y La Libertad, por lo que se informa la cobertura por departamento.
4. **Umbrales de altitud y pendiente.** Tienen respaldo agronómico y normativo, no se ajustaron con el rendimiento y deben congelarse. Cambiarlos después de ver resultados sería un grado de libertad ilegítimo.
5. **Amplitud de NDVI.** El percentil 5 es un criterio, no un valor mágico. Se congela el percentil y se reporta p1 y p10. Con ventana 5, las campañas 2016–2020 son de entrenamiento y el umbral 0,323 es válido sin recalibrar. (Con ventana 4 habría que usar 0,319, calculado con 2016–2019.)
6. **Ventana.** Se mantiene 5 (la de E5 y la tesis). Se evaluó 4 y se descartó: HE1b exigiría 6 de 6 en vez de 5 de 5, B2 con tendencia lineal necesita ≥ 5 campañas de historia, y obligaría a recalibrar la máscara. La razón inicial a favor de 4 (Holm) era incorrecta; ver la corrección más abajo.

## CORRECCIÓN (2026-10-10, al implementar Holm): el argumento de la decisión 8 estaba mal

La razón nº 6 decía que con 5 externas el p mínimo de Wilcoxon (1/32) «no sobrevive a Holm con 2 contrastes». **Es incorrecto.** Revisado el E5 (3.3.5):
- **HE1a y HE1b forman una conjunción** (HE1 se respalda si se cumplen ambas): cada una se contrasta al nivel 0,05 sin corrección. Con 5 externas, 1/32 = 0,031 < 0,05 es alcanzable.
- **Holm solo corresponde a la familia de HE2** (estabilidad, fidelidad, concordancia), y esos tres criterios hoy son **reglas con umbral, sin valor p**; no hay nada que corregir hasta que se defina un p para cada uno.

Lo que queda a favor de la ventana de 4 es poco (un bloque más para el bootstrap y p mínimo 1/64). Lo que queda en contra es más: HE1b exige 6 de 6 en vez de 5 de 5 (más difícil si la CNN solo iguala a B2, como en el piloto técnico), B2 con tendencia lineal necesita ≥ 5 campañas de historia y con ventana 4 la primera campaña externa (2020) cae a la media, y obliga a recalibrar la máscara. **Decisión final: se vuelve a la ventana de 5** (revertido el 2026-10-10 por indicación del tesista).

## Consultas de diseño

| Consulta | Decisión |
|---|---|
| Línea base más fuerte | **Mantener B2 como referencia confirmatoria** (es la definida en la tesis y en la Tabla 8). **Añadir la persistencia como referencia secundaria B4**, reportada de forma descriptiva y fuera de la familia de Holm. Pendiente de implementar y registrar en E5 |
| Identidad de la provincia | **Mantener la anomalía como representación principal**: al restar la media provincial de entrenamiento la identidad ya está implícita. No añadir covariable de historial. Nivel queda como sensibilidad |
| Espacios de búsqueda | **No se ensanchan ahora.** Lo decide el piloto oficial (P9), una sola vez, igual para los 5 modelos, y luego se congela. Prioridad: el dropout de la CNN, cuyo rango actual [0,001; 0,01] es prácticamente cero |

## Tareas que se derivan (no ejecutadas)

1. ~~Recalibrar la amplitud de NDVI con 2016–2019~~ Ya no hace falta con ventana 5. El cálculo queda en `reports/calibracion_mascara_amplitud_2016_2020.json` (valores con 2016–2020; con 2016–2019 el p5 sería 0,319).
2. ~~Implementar B4~~ Hecho. ~~Holm~~ Hecho como utilidad; falta definir el valor p de cada criterio de HE2 para conectarlo.
3. Actualizar las referencias a la ventana en el resumen y las fichas: hecho.
