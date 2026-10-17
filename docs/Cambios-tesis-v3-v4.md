# Cambios de la tesis v3 a la v4

`Proyecto_Tesis_Maestria_UNMSM_WMSR_v4.docx` es una **copia** de la v3 con los cambios de `docs/Propuesta-coherencia-problema-objetivos-hipotesis.md` aplicados como **cambios con control de versiones** de Word (102 revisiones, autor «Claude (propuesta de coherencia)»). La v3 no se modificó.

En Word: pestaña *Revisar* → *Control de cambios* → *Panel de revisiones* para ver cada cambio; *Aceptar* o *Rechazar* uno por uno o todos. Si rechazas todo, el documento vuelve a ser la v3.

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

## Qué revisar tú

1. **Ábrelo en Word.** No pude abrirlo ni renderizarlo (no hay Word ni LibreOffice en el equipo de desarrollo). Lo verifiqué así: el validador de esquema del skill de `.docx` pasó; al rechazar todas las revisiones, los 1 234 párrafos son idénticos a los de la v3; y no hay identificadores de revisión repetidos. No he visto el resultado en pantalla.
2. **Actualiza los campos** (índice, referencias cruzadas) con *Ctrl+A* y *F9*: se editaron párrafos y el índice podría cambiar de página.
3. **Revisa las citas.** Se quitaron tres referencias porque dejaron de citarse; si alguna se cita en otro lugar que no encontré, hay que restaurarla.
4. **Marca temporal de los cambios.** Todas las revisiones llevan la fecha 2026-10-17.
5. **Coherencia con E5.** `configs/preespecificacion.yaml` ya refleja la misma simplificación.
