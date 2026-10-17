# Coherencia entre problema, objetivos e hipótesis

**Estado:** propuesta de trabajo del autor. No modifica `Proyecto_Tesis_Maestria_UNMSM_WMSR_v3.docx`. La simplificación de la sección 7 ya se aplicó a `configs/preespecificacion.yaml` el 2026-10-17; la redacción de problema, objetivos e hipótesis sigue sin pasar a la tesis.
**Base de la revisión:** tesis v3 (capítulos 1, 3 y 4, anexos 01 y 02), `docs/01-analisis-sistema.md`, `docs/02-arquitectura-tecnica.md`, y los resultados del piloto de `service-cnn/reports/` (campañas de entrenamiento 2016-2020; **no vinculantes**).

## 1. Regla para cambiar algo

La preespecificación (E5) todavía no está congelada y las campañas externas 2021-2025 no se han visto. Por eso este es el único momento en que las hipótesis se pueden mejorar sin sesgo. Se permite cambiar por tres motivos y por ningún otro:

1. **Coherencia:** que problema, objetivo e hipótesis hablen de lo mismo, con las mismas palabras.
2. **Validez:** que lo que se contrasta sea lo que se dice contrastar.
3. **Medibilidad:** que el criterio sea estimable con los datos que existen.

**No** es motivo cambiar un margen, un comparador o un criterio porque el piloto muestre que será difícil cumplirlo. Eso sería ajustar la hipótesis al resultado. Todo cambio queda como enmienda fechada (RNF7) y se hace sin mirar 2021-2025.

## 2. Lo que ya es coherente (verificado)

- Los enunciados de PG, PE1, PE2, OG, OE1, OE2, HG, HE1 y HE2 son **idénticos** en §1.2, §1.4, §3.1-§3.2 y en el Anexo 02.
- La cadena es 1 a 1: PE1-OE1-HE1 vinculan X1 con Y1; PE2-OE2-HE2 vinculan X2 con Y2. HE1a-b y HE2a-c son contrastes de las dos hipótesis, no problemas nuevos.
- Las cifras estadísticas de §3.3 son correctas: 0,5⁵ = 0,031; 0,4⁵ = 0,010; 4 de 5 aciertos da p = 0,087; el p mínimo de la fidelidad es 1/3 125; y J ≥ 0,50 se cumple por azar en ≈ 22 % (simulé 400 000 casos: 22,7 %; W ≥ 0,70 por azar: 0,6 %).
- La hipótesis general es conjuntiva y lo dice de forma explícita; los criterios secundarios no compensan a los primarios.

## 3. Incoherencias encontradas

| # | Dónde | Qué pasa | Gravedad |
|---|---|---|---|
| H1 | PG y OG (§1.2.1, §1.4.1) frente a HG (§3.1) | PG y OG preguntan «en qué medida» se puede predecir. HG responde con una conjunción de dos criterios, con comparadores (XGBoost, B*), margen (5 %) y un criterio de explicación. Ni PG ni OG mencionan nada de eso. | Alta |
| H2 | PE1 y OE1 frente a HE1 | PE1 y OE1 piden una **magnitud** («exactitud alcanzada»). HE1 afirma **no inferioridad** frente a dos referencias, dentro de un margen. Los comparadores, el margen y el horizonte solo aparecen en la hipótesis. | Alta |
| H3 | PE2 y OE2 frente a HE2 e Y2 | PE2 y OE2 hablan de «interpretabilidad». HE2 la operacionaliza con un criterio **primario** (concordancia agronómica) y dos **secundarios** (fidelidad, estabilidad). El problema y el objetivo no distinguen primario de secundario. | Media |
| H4 | §1.1 frente a HG y HE1 | §1.1 exige anticipar «con la antelación que la decisión agraria requiere». El horizonte principal es 30 días antes de la cosecha (§1.5). HG y HE1 no declaran horizonte; solo HE2c lo hace. | Media |
| H5 | §1.5, §4.3, `docs/01` (RF6, RF7) frente a Tablas 1 y 8, HG y código | Se dice «tres benchmarks» y «8 tratamientos», pero la Tabla 1 y la Tabla 8 listan cuatro (B1-B4) y B* = mejor entre B2 y B4. | Media |
| H6 | Tabla 8 | B2 figura como «referencia confirmatoria, junto con B4» y B4 como «referencia secundaria y descriptiva». B4 entra en B*, así que es confirmatorio. | Media |
| H7 | HG y HE1 | «La mejor línea base provincial» designa a B* (B2 o B4). B4 es persistencia de la última campaña: provincial, pero no es «línea base» en el mismo sentido que B2. | Baja |
| H8 | Título, PG | «Sierra peruana» es más amplio que el alcance real: provincias altoandinas de ocho departamentos (§1.5). «Interpretable» no se define hasta HE2. La unidad provincia-campaña solo aparece en prosa. | Baja |
| H9 | `docs/01` y `docs/02` | Siguen la versión anterior de la tesis: HE1a solo frente a XGBoost, HE1b como superioridad frente a B2 y confirmatoria, tres benchmarks, `evaluate_superiority()` frente a B2. La v3 cambió todo eso. | Media |

## 4. Reformulación propuesta (texto listo para pegar)

Los criterios, márgenes y reglas de decisión de §3.3 **no cambian**. Cambia la redacción de problema, objetivos e hipótesis para que digan lo mismo, y se declara el horizonte.

**Problema general.** ¿Un modelo CNN-1D interpretable con SHAP predice el rendimiento de quinua por provincia y campaña en la sierra peruana con una exactitud no inferior a la de XGBoost y la de la mejor línea base (B*), y con explicaciones concordantes con las fases críticas del cultivo?

**PE1.** ¿La exactitud predictiva del modelo CNN-1D en las campañas externas es no inferior, dentro de un margen relativo de 5 % en el RMSE, a la de XGBoost y a la de la mejor línea base B*, en el horizonte principal?

**PE2.** ¿Son las explicaciones SHAP de las predicciones del CNN-1D concordantes con las fases críticas preespecificadas del cultivo y, como criterios secundarios, fieles al comportamiento del modelo y estables entre campañas?

**Objetivo general.** Evaluar si un modelo CNN-1D interpretable con SHAP permite predecir el rendimiento de quinua por provincia y campaña en la sierra peruana con una exactitud no inferior a la de XGBoost y la de la mejor línea base (B*), y con explicaciones SHAP concordantes con las fases críticas preespecificadas.

**OE1.** Evaluar si la exactitud predictiva (RMSE, MAE, rRMSE y R²) del modelo CNN-1D es no inferior, dentro de un margen relativo de 5 % en el RMSE, a la de XGBoost y a la de la mejor línea base B*, en las campañas externas y en el horizonte principal.

**OE2.** Evaluar la interpretabilidad de las predicciones del CNN-1D mediante la concordancia agronómica de las explicaciones SHAP con las fases críticas preespecificadas (criterio primario), y su fidelidad por oclusión y su estabilidad entre campañas (criterios secundarios).

**Hipótesis general.** El modelo CNN-1D interpretable con SHAP permite predecir el rendimiento de quinua por provincia y campaña en la sierra peruana: en el horizonte principal, su exactitud en las campañas externas no es inferior (margen relativo de 5 %) a la de XGBoost ni a la de la mejor línea base B*, y sus explicaciones SHAP son concordantes con las fases críticas preespecificadas (floración y llenado de grano). Se considera respaldada solo si se cumplen ambos criterios primarios (HE1a y HE2c); la fidelidad y la estabilidad son criterios secundarios y no compensan un incumplimiento.

**HE1.** En el horizonte principal, el modelado temporal mediante CNN-1D alcanza una exactitud predictiva no inferior a la de XGBoost ni a la de la mejor línea base B*, dentro de un margen relativo preespecificado de 5 %. Como análisis exploratorio se estima la ganancia de habilidad predictiva sobre B*.

**HE2.** La explicabilidad mediante SHAP genera interpretaciones del rendimiento de quinua concordantes con las fases fenológicas críticas preespecificadas (criterio primario); su fidelidad al comportamiento del modelo y su estabilidad entre campañas se evalúan como criterios secundarios.

**Y2 (Anexo 01 y 02).** «Interpretabilidad de la predicción: concordancia agronómica (primaria), fidelidad por oclusión y estabilidad entre campañas (secundarias) de las explicaciones SHAP».

**Definiciones que deben escribirse una sola vez (§1.5 o §3.4) y reutilizarse:**
- *B\** = la mejor entre B2 (línea base provincial con tendencia) y B4 (persistencia de la última campaña), elegida con la validación interna antes de evaluar las campañas externas.
- *Horizonte principal* = 30 días antes de la cosecha esperada. *Horizonte secundario* = cierre de la floración.
- *Interpretable* = que sus explicaciones SHAP pasan el protocolo de §4.15.
- *Benchmarks* = cuatro (B1-B4) en todo el documento. B2 y B4 son **confirmatorios** porque definen B*; B1 y B3 son descriptivos.

## 5. Trazabilidad: de la pregunta a la pantalla

| Pregunta | Criterio primario | Regla de decisión | Indicador (Anexo 01) | Código en `service-cnn/src` | Pantalla de app-quinuaprod |
|---|---|---|---|---|---|
| PE1 / OE1 / HE1 | HE1a: q < 0,05 en cada campaña frente a XGBoost y a B* | Regla exacta de signos (nivel 0,5ⁿ) | Y1: RMSE, MAE, rRMSE, R²; SS (exploratorio) | `evaluation/he1_evaluation.py`, `evaluation/indicators.py` | Indicadores (Y1), Predicción vs observado |
| PE2 / OE2 / HE2 | HE2c: la fase de máxima contribución pertenece a {floración, llenado de grano} | Binomial exacta unilateral, p₀ = 0,4 | Y2: coincidencia de la fase máxima | `evaluation/he2_evaluation.py`, `explainability/he2_pvalues.py` | Explicabilidad |
| (secundario) | HE2a estabilidad; HE2b fidelidad | Permutación; aleatorización exacta; Holm | Y2: W de Kendall, J corregido, aumento del RMSE por oclusión | `explainability/fidelity.py`, `evaluation/he2_evaluation.py` | Explicabilidad (HE2 técnico) |
| HG | HE1a **y** HE2c | Conjunción | Todas las anteriores | `evaluation/pilot_runner.py` | Resumen, Indicadores |
| Robustez de decisiones | — | Análisis de sensibilidad | Variantes preespecificadas en E5 (sección `sensibilidad`) | `evaluation/sensitivity.py` | Sensibilidades |
| Preespecificación | — | Registro fechado | — | `evaluation/preregistration.py` | Registro E5 |

La aplicación aún **no tiene una pantalla que muestre esta cadena**. Se propone una pantalla «Hipótesis y objetivos» que presente PG/PE/OE/HE con su criterio y la evidencia disponible.

## 6. Riesgos de diseño que la redacción no resuelve

> Resolución en la sección 7: se declaran, sin trabajo adicional. Las opciones que implican exportar SHAP por semilla o cambiar el criterio de HE2c quedan descartadas.

Estos puntos son decisiones del autor. Los números del piloto son de campañas de entrenamiento y no vinculantes; sirven para ver la forma del problema, no para ajustar el diseño.

**R1. HE1a exige 5 de 5.** Con cinco campañas externas la regla de signos solo se cumple si todas tienen q < 0,05. En el piloto (2019 y 2020) el CNN-1D tuvo q = −5,7 % y +4,7 % frente a XGBoost; −1,0 % y +1,3 % frente a B2; y +0,7 % y **+14,3 %** frente a B4. Una sola campaña con q ≥ 5 % rechaza HE1a. B* se elegirá con la validación interna antes de evaluar 2021-2025; si resulta ser B4, 2020 muestra que persiste un riesgo real.
*Opciones:* (a) mantener la regla y declarar de antemano que el desenlace puede ser el rechazo (la tesis ya afirma que el diseño es informativo en cualquier resultado) — **recomendado**; (b) añadir un análisis de apoyo agrupado además de la regla exacta. **No** se recomienda mover B4 a «descriptivo» ni subir δ: sería elegir un comparador o un margen más fáciles tras ver resultados.

**R2. HE2c depende de una fase máxima casi empatada.** En el piloto, la contribución por fase es casi plana. En 2018 la diferencia entre la primera fase (desarrollo vegetativo, 0,175) y la segunda (floración, 0,171) es de 2,3 %; en 2019, de 10,0 %. Un cambio mínimo de semilla o de fondo puede cambiar el argmax y, con él, la conclusión. Además, con cinco campañas solo 5 de 5 aciertos da p < 0,05; 4 de 5 da p = 0,087.
*Opciones:* (a) mantener el criterio primario tal cual; (b) antes de congelar E5, comprobar con campañas de **entrenamiento** cuán estable es el argmax entre semillas y, si es inestable, justificar por **validez** (no por resultado) un criterio más robusto, por ejemplo basado en la proporción de contribución agregada de {floración, llenado de grano} frente a 0,40 (en el piloto: 0,431). Para (b) hace falta exportar SHAP por semilla, que hoy no se guarda.

**R3. La conjunción HG tiene poca probabilidad de cumplirse a la vez.** HE1a (5/5) y HE2c (5/5) son estrictos. Esto no es un defecto si se declara: el valor de la tesis es la evidencia en cualquier desenlace. Conviene que PG/OG lo digan con esas palabras y que la defensa no dependa de que HG se respalde.

**R4. El referente agronómico es plausibilidad, no estimando.** Vásquez et al. (2024) es un experimento de sombreo en cuatro países, no una predicción provincial. §4.15 ya lo trata como «plausibilidad externa y no causalidad»; PE2/HE2 deberían usar la misma palabra.

**R5. El horizonte principal es tardío para decidir.** §1.1 justifica la tesis por la necesidad de anticipar. A 30 días de la cosecha, las decisiones de asistencia técnica ya no pueden actuar sobre la producción, aunque sí sobre acopio y comercialización. En el piloto, el horizonte temprano casi no cambia el error del CNN-1D (RMSE 205,0 frente a 200,4 en 2019; 195,3 frente a 196,4 en 2020), lo que sugiere que las fases tardías aportan poca señal sobre la línea base. *Decisión:* o se justifica el horizonte principal por los usos de decisión poscosecha (acopio, compras públicas), o se promueve el horizonte temprano a confirmatorio. Cualquiera de las dos debe escribirse en §1.1 o §1.5.

## 7. Simplificación propuesta (para reducir esfuerzo)

**Principio.** Simplificar no es bajar la exigencia. Los criterios que **deciden** (HE1a, HE2c y la conjunción HG) se quedan exactamente como están, con el mismo margen, comparadores y reglas. Lo que se simplifica es todo lo que **no decide nada** y hoy exige pruebas, parámetros y texto.

| Elemento | Hoy | Propuesta | Efecto sobre las decisiones |
|---|---|---|---|
| HE1a: regla exacta de signos | Decide | **Se mantiene** | — |
| HE2c: binomial exacta | Decide | **Se mantiene** | — |
| HG: conjunción HE1a y HE2c | Decide | **Se mantiene** | — |
| Contrastes de apoyo de HE1 (Wilcoxon, equivalencia bayesiana, Hodges-Lehmann, correlación biserial) y bootstrap descriptivo | Apoyo, no deciden | Dejan de ser contrastes. Se reporta la mediana de q y el rango por campaña | Ninguno |
| HE1b: ganancia de habilidad SS | Exploratoria | Se mantiene como cifra descriptiva | Ninguno |
| HE2a estabilidad (W de Kendall, J corregido) | Secundaria, con valor p por permutación (10 000) | Se reportan W y J como números, sin valor p | Ninguno |
| HE2b fidelidad por oclusión | Secundaria, con valor p por aleatorización | Se reporta el aumento del RMSE y el rango de la fase máxima, sin valor p | Ninguno |
| Corrección de Holm y «familia de contrastes» | Corrige 2 contrastes secundarios | **Desaparece**: ya no hay contrastes secundarios | Ninguno |

**Qué se ahorra.** En E5 dejan de hacer falta `n_bootstrap`, `semilla_bootstrap`, `contraste_de_apoyo`, `correccion_por_multiplicidad`, `familia_de_contrastes`, `valor_p_estabilidad` y `valor_p_fidelidad`. En la tesis se acortan §3.3.1, §3.3.3, §3.3.4 y §4.14. No hay que implementar nada nuevo: el código actual ya calcula todo, y simplemente se deja de exigir como contraste.

**Qué se pierde y conviene decir con franqueza.** Estabilidad y fidelidad dejan de tener una afirmación inferencial formal. Con cinco campañas esas pruebas tenían muy poca potencia (en el piloto técnico, los valores p fueron 0,70 y 0,76), de modo que el costo real es pequeño. Pero **este recorte debe justificarse por esa falta de potencia y por simplicidad, no por esos resultados**, y las cifras se siguen reportando completas para que nadie pueda pensar que se retiró una prueba incómoda. Un jurado podría preguntar por qué no hay prueba de estabilidad; la respuesta es que con cinco campañas ninguna prueba puede discriminar.

**Los cuatro riesgos de la sección 6 pasan a declararse, sin trabajo extra:**
- **R1 (HE1a 5 de 5):** se mantiene y se declara que el desenlace puede ser el rechazo.
- **R2 (fase máxima casi empatada):** se mantiene HE2c y se agrega una frase en las limitaciones: la fase máxima puede ser frágil cuando la contribución por fase es casi plana. La ventaja sobre la segunda fase ya se reporta (la UI la muestra). **No** hace falta exportar SHAP por semilla.
- **R3 (conjunción difícil):** se declara en la hipótesis general, como ya propone la sección 4.
- **R4 (referente agronómico):** se usa la palabra «plausibilidad», como ya hace §4.15.

## 8. Decisiones tomadas por el autor

- **Alcance:** solo redacción. No cambian los criterios, márgenes ni reglas de §3.3 (R1 y R2 quedan como riesgos declarados, no como cambios de diseño).
- **Horizonte (R5):** se mantiene 30 días antes de la cosecha como principal, y el cierre de floración como secundario. Texto propuesto para §1.5, tras la definición del horizonte:

  > El horizonte principal se fija en 30 días antes de la fecha esperada de cosecha porque las decisiones que el estudio atiende, como el acopio, la comercialización y las compras públicas, se toman sobre la oferta que ya está en formación y no sobre el manejo del cultivo. El cierre de la floración se analiza como horizonte secundario, para examinar qué parte de la capacidad predictiva se conserva con mayor antelación. El estudio no sostiene que la predicción a 30 días permita actuar sobre el manejo del cultivo.

  Esta redacción es coherente con §1.1, que nombra a los actores de acopio, transformación y exportación, y con §1.6, que ya limita el uso a la orientación agregada. Si una parte de la justificación de la tesis se apoya en intervenir sobre el cultivo, debería moverse al horizonte secundario.

## 9. Cambios derivados

Hecho:
- `docs/01-analisis-sistema.md` y `docs/02-arquitectura-tecnica.md` ya describen cuatro benchmarks (B1-B4), B*, HE1a frente a XGBoost y B*, y HE1b exploratoria. `docs/02` señala que `hypothesis_tests.py` es un módulo heredado de la v2.

Pendiente (requiere editar el docx o la preespecificación):

1. **Tesis (docx):** §1.2, §1.4, §3.1, §3.2, Anexo 02, Anexo 01 (denominación de Y2), §1.5 (horizonte y «cuatro benchmarks»), §4.3, Tabla 8 (rol de B4) y §3.4 (definición de B*); si se acepta la simplificación de la sección 7, también §3.3.1, §3.3.3, §3.3.4 y §4.14. La incoherencia H9 (`docs/01` y `docs/02`) ya está corregida.
2. **`configs/preespecificacion.yaml`:** **hecho (2026-10-17)**: se retiraron los siete parámetros de la sección 7 y se agregaron `reporte_de_apoyo_he1`, `secundarios_descriptivos` y una entrada de enmienda; E5 se regeneró y `check_consistency_with_code` sigue limpio. Se corrigió además `valor_p_concordancia` (decía «p < 0.0167», que era del esquema anterior con Holm; ahora dice p < 0.05). La tesis (docx) debe reflejar el mismo cambio, o E5 y la tesis quedarán distintos.
3. **app-quinuaprod:** pantalla «Hipótesis y objetivos», alineada con esta redacción (no se pidió por ahora).
4. **`app-quinuaprod/docs/enmienda-alcance-ui.md`:** habla de decisiones del asesor; cambiarlo a decisiones abiertas del autor mientras no haya asesor (no se pidió por ahora).
