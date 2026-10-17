# Cambios de la tesis v3 a la v4

`Proyecto_Tesis_Maestria_UNMSM_WMSR_v4.docx` es una **copia** de la v3 con los cambios de `docs/Propuesta-coherencia-problema-objetivos-hipotesis.md` aplicados como **cambios con control de versiones** de Word (102 revisiones, autor «Claude (propuesta de coherencia)»). La v3 no se modificó. Desde el 2026-10-17 lleva además una **segunda capa** de 34 revisiones (autor «Claude (alineación con los datos)») que actualiza el texto con los datos reales 2016-2025; está descrita en la sección «Segunda capa» más abajo.

En Word: pestaña *Revisar* → *Control de cambios* → *Panel de revisiones* para ver cada cambio; *Aceptar* o *Rechazar* uno por uno o todos. Se puede filtrar por autor para aceptar o rechazar una capa completa sin tocar la otra. Si rechazas todo, el documento vuelve a ser la v3.

## Qué se cambió

| Dónde | Cambio |
|---|---|
| §1.2 (problema general, PE1, PE2) y §1.4 (objetivo general, OE1, OE2) | Redacción alineada con la hipótesis: mismos comparadores (XGBoost y B*), mismo margen (5 %), horizonte principal y criterios primario y secundarios. Los objetivos pasan de «Determinar» a «Evaluar si» |
| §1.5 (delimitación metodológica) | «Tres benchmarks» pasa a «cuatro (B1 a B4)»; se define B* y el rol de cada benchmark una sola vez |
| §1.5 (horizonte de pronóstico) | Se justifica el horizonte principal de 30 días por los usos poscosecha y se aclara que no permite actuar sobre el manejo del cultivo |
| §3.1 (hipótesis general) | Redacción nueva; el párrafo siguiente reporta los secundarios como cifras, sin valor p |
| §3.2 (HE1, HE2) | Redacción alineada; HE1 declara el horizonte principal. Se define «interpretable» (explicaciones que superan el protocolo de §4.15) |
| §3.3 | Intro: «rango por campaña» en lugar de intervalos de confianza. §3.3.1: «línea base provincial (B2)» (decía «media provincial»); se retiran Wilcoxon, equivalencia bayesiana, Hodges-Lehmann y correlación biserial. §3.3.3: se retiran H₀/H₁ y la prueba de permutación. §3.3.4: se retiran R, H₀/H₁ y el valor p. §3.3.5: se retira la corrección de Holm |
| §4.3 | «Los tres benchmarks» pasa a «Los cuatro benchmarks» |
| §4.14 | La decisión de HE1 es solo la regla exacta de signos; la estabilidad y la fidelidad se reportan como cifras descriptivas |
| §4.15 (estabilidad) y §4.18 (preespecificación) | Se quita «contrastadas por permutación» y «valores p de HE2» (queda el valor p de la concordancia) |
| Tabla 1, Tabla 10 (amenazas a la validez, §4.16) y Anexo 01 | Se quitan las menciones al valor p de permutación, a la corrección de Holm y a los intervalos por bloques |
| Tabla 8 | B4 pasa de «secundaria y descriptiva» a «confirmatoria, junto con B2» (define B*) |
| §2 (marco teórico) | Se aclara que, con cinco campañas, la no inferioridad se lee con la regla exacta de signos y no con un intervalo de confianza |
| Anexo 02 (matriz de consistencia) | Reproduce literalmente la redacción nueva de los capítulos 1 y 3, y la decisión y los análisis simplificados |
| Referencias | Se eliminan Benavoli et al. (2017), Holm (1979) y Phipson y Smyth (2010), que dejan de citarse |

## Qué no se cambió, a propósito

- **El nombre de la dimensión Y2** («Interpretabilidad de la predicción»): se repite literalmente en §2, §3.4 y los anexos, y la matriz debe reproducir las mismas denominaciones. Cambiarla exige tocar todos esos lugares; se puede hacer después si se decide.
- **HE1a, HE2c y la conjunción HG**: criterios, margen (5 %) y reglas de decisión idénticos.
- **HE1b**: sigue exploratoria, con su «intervalo exacto por signos».
- **El título y el periodo de campañas.**

## Segunda capa: alineación con los datos reales (2026-10-17)

La tesis está redactada como protocolo, en futuro («se informará al cierre de H1», «se definirá en el piloto»). Estas 34 revisiones (17 párrafos) reemplazan lo que quedó pendiente por lo que los datos y el piloto ya resolvieron. **Solo contienen hechos de datos y decisiones de diseño tomadas con campañas de entrenamiento; no se redactó ningún resultado de la evaluación confirmatoria.**

| Dónde | Antes | Ahora | Procedencia |
|---|---|---|---|
| §1.5, «Tamaño temporal efectivo» | «La cobertura definitiva … limitarán el número de campañas externas» | Con ventana de cinco campañas (2016-2020) quedan cinco externas (2021-2025); la regla de signos alcanza 0,5⁵ = 0,031 | E5 (`ventana_inicial_de_entrenamiento_campanas` = 5) |
| §4.4, «Tiempo» | «se confirma antes del entrenamiento» | El periodo quedó confirmado: 601 observaciones de 66 provincias | E3 §1 |
| §4.5.3 | «La clasificación definitiva se verificará» | Se añade que la clasificación usada (63 altoandinas, 4 de transición —Carabaya, Sandia, La Mar y Huanta—, 9 costeras bajo riego y 3 de selva) coincide con la Tabla 4; Gran Chimú sigue pendiente y se conservó marcada, con una observación | `service-cnn/configs/provincias.csv`; `data/processed` |
| Tabla 5, N2 | «diez provincias costeras bajo riego y Gran Chimú» | **Nueve** provincias costeras bajo riego y Gran Chimú (diez en total, que son las que dan 67 × 10 = 670). En la depuración final se excluyeron las 44 celdas de las nueve costeras; Gran Chimú se conservó, marcada, con una observación | Idem |
| §4.5.6 y Tabla 5, N4 | 604 («97,4 % de cobertura real») | **674** celdas de 76 provincias; supera a N3 porque la producción abarca más provincias que las 62 documentadas; el 604 de la auditoría H1 se conserva como el recuento dentro de esas 62 | E3 §3.1; recalculado con `run_preprocessing.py` y el caché del SISAGRI |
| Tabla 5, N5 | «Por determinar» | **601** observaciones de 66 provincias, 571 con tensor completo; de N4 se excluyeron 44 fuera del ámbito altoandino y 29 por ciclo fuera de la ventana (674 − 44 − 29 = 601) | E3 §1 y §3; `data/processed/exclusiones.csv` |
| Nota de la Tabla 5 | «Se informarán … al cierre de H1» | Observaciones por campaña: 60, 62, 60, 62, 58, 61, 60, 60, 60 y 58 (2016 a 2025) | `data/processed/observaciones.parquet` |
| §4.5.7 y §4.6 | «El tamaño … se ajustarán … después de H1» | N5 = 601; 302 observaciones de entrenamiento (2016-2020) y 299 de las cinco campañas externas | Idem |
| §4.10.1 | umbrales «se fijarán y registrarán» | Amplitud mínima de NDVI calibrada solo con 2016-2020 y fijada en 0,323 | E5 (`amplitud_ndvi_minima`), `data/manifest/mascara_agricola.yaml` |
| §4.11, representación | «se utilizará … cuando mejore» | En el piloto la anomalía redujo el RMSE de 394,6 a 198,2 kg/ha (−49,8 %); se adoptó | E5 (`representacion_de_modelado`), piloto oficial P4 |
| §4.11, línea base | «se seleccionará en el piloto» | Tendencia lineal por provincia con al menos cuatro campañas de entrenamiento; con menos, media provincial | E5 (`forma_de_la_linea_base_b2`) |
| §4.12, validación temporal | «se definirá en el piloto» | Ventana inicial de cinco campañas, fijada en la preespecificación | E5 |

Cada cifra se puede reproducir con el repositorio: `scripts/build_e3_report.py` (E3, regenerado el 2026-10-17 con el conjunto no exploratorio), `configs/preespecificacion.yaml` (E5 v1.1) y `data/processed`. El informe E3 se corrigió en el mismo cambio: usaba por defecto la calibración de la máscara con 2016-2019 (p5 = 0,319) en vez de la de 2016-2020 que fijó E5 (0,323), y seguía rotulado como exploratorio.

### Qué no se alineó (queda pendiente)

- **Resultados de la evaluación confirmatoria.** Existen (`service-cnn/docs/crisp/evaluacion-confirmatoria-oficial/`) pero no se redactaron en la tesis; depende de que decidas cómo y cuándo.
- **Rango plausible del rendimiento (§4.7, criterio de exclusión 3):** el texto dice que «el rango plausible y la tolerancia por redondeo se definirán antes de H1». El análisis no definió un rango numérico: marcó los extremos y los conservó (máximo 4 311 y mínimo 282 kg/ha; el 5 000 kg/ha previo era Chepén 2024, costa con riego, ya excluida). Decide si el texto debe decir eso o si quieres fijar un rango; no lo reescribí.
- **Gran Chimú** sigue sin clasificación definitiva (`pendiente_verificacion` en `configs/provincias.csv`); el texto lo dice, pero la verificación con la fuente territorial oficial la tienes que hacer tú.
- **Cronograma, hitos y presupuesto:** H2 (cierre del piloto y registro de preespecificación) y los costos «tras el ensayo piloto» no se actualizaron.
- **Parámetros de ajuste** (30 ensayos por modelo, 47 épocas máximas, diez semillas): están en E5; el texto solo menciona las diez semillas y no se añadieron los otros.

## Qué revisar tú

1. **Ábrelo en Word.** No pude abrirlo ni renderizarlo (no hay Word ni LibreOffice en el equipo de desarrollo). Lo verifiqué así: el validador de esquema del skill de `.docx` pasó; al rechazar todas las revisiones, los 1 234 párrafos son idénticos a los de la v3; y no hay identificadores de revisión repetidos. Con la segunda capa lo repetí: el validador pasó, al rechazar solo esa capa vuelve exactamente el texto anterior (1 234 párrafos iguales) y los ids de revisión no se repiten. No he visto el resultado en pantalla.
   **Revisión mecánica ya hecha (2026-10-17)** sobre el texto con todos los cambios aceptados: ningún resto de lo retirado fuera de las frases que explican por qué no se usa (Holm, remuestreo, Wilcoxon, valor p, permutación, «tres benchmarks»); todas las menciones dicen cinco campañas externas, nivel 0,5⁵ = 0,031, y los valores p binomiales de §3.3.5 (0,010 con 5 de 5 y 0,087 con 4 de 5) los recalculé y son correctos; las cifras N0 a N5, 302 + 299 = 601 y 674 − 44 − 29 = 601 cuadran. Lo que esa revisión no puede juzgar es el estilo, la redacción en pantalla ni los números de página.
2. **Actualiza los campos** (índice, referencias cruzadas) con *Ctrl+A* y *F9*: se editaron párrafos y el índice podría cambiar de página.
3. **Revisa las citas.** Se quitaron tres referencias porque dejaron de citarse; si alguna se cita en otro lugar que no encontré, hay que restaurarla.
4. **Marca temporal de los cambios.** Todas las revisiones llevan la fecha 2026-10-17.
5. **Coherencia con E5.** `configs/preespecificacion.yaml` ya refleja la misma simplificación.
