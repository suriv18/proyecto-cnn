# Propuesta de reformulación de la tesis (borrador para decidir)

Fecha: 2026-10-10. Elaborada a partir de una revisión de literatura y fuentes (cuatro búsquedas independientes) y de los resultados del trabajo técnico. **Nada de esto está aplicado al `.docx` ni a E5 todavía.** Las referencias marcadas «verificada» se abrieron en Crossref, arXiv, JMLR u otra fuente primaria; las marcadas «sin verificar» no se pudieron abrir y hay que comprobarlas antes de citarlas.

## 1. Qué incoherencias hay hoy entre la tesis y el trabajo

| Tema | La tesis (`.docx`) dice | El trabajo real |
|---|---|---|
| Periodo (1.5) | 2006–2024, 19 campañas | 2016–2025, 10 campañas (no hay fuente provincial anterior a 2015) |
| Universo | 62 provincias preespecificadas (Tabla 3) | 76 provincias con producción observada; 601 observaciones elegibles |
| HE1b | CNN **mejor** que B2 en la mediana, límite superior bootstrap < 0 | con 5 campañas externas exige ganar las 5; el ensayo muestra CNN ≈ B2 ≈ persistencia |
| HE1a | bootstrap por bloques de campaña | con 5 bloques solo hay 3125 remuestreos distintos y el límite es poco fiable |
| HE2a | `H₀: W < 0,70` sin estadístico de contraste; Jaccard ≥ 0,50 | el umbral Jaccard se cumple por azar el 22 % de las veces |
| HE2b | un solo control de igual duración, Wilcoxon pareado | con 5 campañas el p mínimo (1/32) no pasa Holm con 3 contrastes |
| Holm (3.3.5) | «corrección de Holm a la familia de HE2» | familia ambigua: estabilidad y concordancia son reglas con umbral |

## 2. Hallazgos de la búsqueda

**Datos (¿se puede volver a 2006–2015?).** No hay fuente tabular provincial o distrital descargable de quinua anterior a 2015. Sí existe una serie **departamental 2004–2014** (paquete `cropdatape`, https://github.com/omarbenites/cropdatape, datos de MINAGRI-SIEA; verificada, licencia sin confirmar). Los anuarios SIEA archivados en Wayback (2015–2017) traen cuadros regionales, no provinciales. La vía realista para provincia es **solicitar** a MIDAGRI-DGESEP (o a la Dirección Regional Agraria de Puno) la base provincial por acceso a la información pública. Con la serie departamental se pierde la variación intra-departamental, que es la que sostiene el diseño provincia-campaña.

**Expectativas de exactitud.** La literatura respalda que el aprendizaje profundo apenas iguala líneas base fuertes con pocos datos:
- Paudel et al. (2021), *Machine learning for large-scale crop yield forecasting*, Agricultural Systems 187. DOI 10.1016/j.agsy.2020.103016 (verificada).
- Meroni et al. (2021), *Agric. For. Meteorol.* 308:108555, arXiv 2104.13246; código `ec-jrc/ml4cast-ml` (verificada): los modelos de referencia superaron a gran parte de los modelos de ML.
- Grinsztajn, Oyallon & Varoquaux (2022), *Why do tree-based models still outperform deep learning on tabular data?*, arXiv 2207.08815 (verificada).
- Shwartz-Ziv & Armon (2022), *Information Fusion* 81. DOI 10.1016/j.inffus.2021.11.011 (verificada).
- «Is deeper always better? … small data» (2023), *Environ. Monit. Assess.*, DOI 10.1007/s10661-023-11609-8 (sin verificar: solo vista en resultados de búsqueda; el pico de NDVI fue una línea base difícil de superar).
- Ploton et al. (2020), *Nat. Commun.*, DOI 10.1038/s41467-020-18321-y (sin verificar el texto; es de biomasa forestal, no de cultivos): la validación no espacial infla el desempeño.

**Quinua.** Lo encontrado es a escala de planta o parcela; no apareció predicción de quinua a escala provincial con NDVI y deep learning, ni quinua con SHAP. Afirmar «no se encontró en la revisión» y no «no existe». Ejemplo verificado en repositorio institucional: Pizarro Carcausto et al. (2026), *Remote Sens. Appl.*, DOI 10.1016/j.rsase.2026.102027 (R² ≤ 0,25, índices espectrales pobres, validación aleatoria). Fenología: Vásquez et al. (2024), *Field Crops Research*, DOI 10.1016/j.fcr.2023.109207 (verificada su existencia; el periodo crítico es una ventana alrededor de la floración, no una fase puntual); Geerts et al. (2008), *Eur. J. Agron.*, DOI 10.1016/j.eja.2007.11.008.

**Contraste estadístico con pocos bloques.** Demšar (2006), JMLR 7:1–30; García & Herrera (2008), JMLR 9; Benavoli et al. (2017), JMLR 18(77) (comparación bayesiana con ROPE); Cameron, Gelbach & Miller (2008), *Rev. Econ. Stat.* 90, DOI 10.1162/rest.90.3.414 (pocos clústeres); Lakens (2017), *SPPS* 8, DOI 10.1177/1948550617697177 (equivalencia/TOST); Diebold & Mariano (1995), DOI 10.1080/07350015.1995.10524599; Giacomini & White (2006), DOI 10.1111/j.1468-0262.2006.00718.x. Murphy (1988), *Mon. Weather Rev.* 116, DOI 10.1175/1520-0493(1988)116<2417:SSBOTM>2.0.CO;2 (skill score basado en el MSE). Validación temporal: Cerqueira et al. (2020), *Machine Learning* 109, DOI 10.1007/s10994-020-05910-7; Roberts et al. (2017), *Ecography* 40, DOI 10.1111/ecog.02881; Bergmeir & Benítez (2012), DOI 10.1016/j.ins.2011.12.028. Todas verificadas. **Ninguna fija un número mínimo de bloques**: eso es inferencia propia y debe declararse como limitación.

**Evaluación de explicaciones.** Fidelidad y oclusión: Petsiuk et al. (2018) RISE, arXiv 1806.07421; Hooker et al. (2019) ROAR, arXiv 1806.10758; Yeh et al. (2019), arXiv 1901.09392; Hedström et al. (2023) Quantus, arXiv 2202.06861; Hase, Xie & Bansal (2021), arXiv 2106.00786 (problema fuera de distribución al reemplazar variables). Estabilidad: **Nogueira, Sechidis & Brown (2018), JMLR 18(174)**, medida de estabilidad corregida por azar con intervalo y prueba de hipótesis: es el respaldo directo para reemplazar el umbral Jaccard ≥ 0,50. SHAP en series temporales: Bento et al. (2021) TimeSHAP, DOI 10.1145/3447548.3467166; Ismail et al. (2020), arXiv 2010.13924. Permutación: Phipson & Smyth (2010), DOI 10.2202/1544-6115.1585. Todas verificadas. No se encontró una referencia revisada por pares sobre pre-registro de concordancia con pocos bloques.

## 3. Propuesta

### Periodo y datos
1. **Mantener 2016–2025** como periodo del modelo (es la única fuente provincial). Reescribir 1.5 con 10 campañas, 76 provincias con producción observada y 601 observaciones elegibles; documentar la enmienda con su motivo (ausencia de fuente anterior a 2015).
2. Pedir a MIDAGRI la base provincial 2006–2015 como gestión paralela. Si llega, se reabre; si no, queda como limitación.
3. Usar `cropdatape` (departamental 2004–2014) solo como **contexto**: tendencia de largo plazo y comprobación de que el periodo cubre variabilidad climática comparable. No entrena el modelo.

### HE1 (exactitud)
- **Comparador de referencia B\***: el mejor entre B2 (media provincial) y B4 (persistencia), elegido con la validación interna y antes de ver las campañas externas.
- **Estadístico**: *skill score* de Murphy, SS = 1 − MSE_CNN / MSE_ref, por campaña externa; resumen por mediana.
- **HE1a (confirmatoria)**: la CNN-1D **no es inferior** a XGBoost (δ = 5 %) **y** a B\*, con cota exacta por permutación de signos (reemplaza el bootstrap con 5 bloques) y ROPE bayesiano de ±5 % como contraste de apoyo.
- **HE1b (exploratoria)**: SS > 0 frente a B\*, informada con magnitud, intervalo y número de campañas favorables. No condiciona la decisión de HE1.
- Declarar que con 5 campañas la potencia es baja y que «equivalente a la línea base» es un resultado válido.

### HE2 (interpretabilidad)
- **Concordancia (criterio primario)**: binomial exacta sobre campañas con p₀ = 0,4 (p = 0,010 con 5/5; 0,087 con 4/5). Pre-registrar además, como secundario, que la proporción agregada de contribución de floración + llenado supere 0,40 con intervalo por campañas. Fijar antes de ver datos si el referente es la fase puntual o la ventana prefloración–llenado (Vásquez et al., 2024).
- **Fidelidad**: ΔRMSE al ocluir cada una de las 5 fases y su relación con la contribución SHAP (dosis-respuesta) o, más simple, la aleatorización exacta por rangos ya implementada; reportar la oclusión con al menos dos reemplazos (fondo marginal y condicional) por el problema fuera de distribución.
- **Estabilidad**: pasar a criterio **descriptivo con valor p de permutación**: W de Kendall contra el nulo de permutación y estabilidad de Nogueira (o Jaccard menos su esperanza al azar) con intervalo; abandonar J ≥ 0,50. Declarar que con contribuciones casi planas la estabilidad del ranking es poco informativa.
- **Multiplicidad**: un único contraste primario (concordancia) y fidelidad y estabilidad como secundarios con Holm entre ellos, o un orden secuencial fijo pre-registrado. Con 5 campañas un p mínimo de 1/32 no pasa Holm de 3; con ventana 4 (6 campañas) el mínimo es 1/64, pero esa ventana se descartó por otras razones.

### Marco teórico
Añadir un párrafo de antecedentes que diga que el aprendizaje profundo con pocos datos suele igualar, no superar, líneas base fuertes (Paudel; Meroni; Grinsztajn), que la validación aleatoria infla el desempeño (Cerqueira; Roberts), y que no se encontró predicción de quinua a escala provincial con series de NDVI y clima ni con SHAP. Replantear el aporte: **evaluación honesta, con validación temporal y línea base explícita, de si un modelo interpretable añade valor sobre la línea base provincial en un cultivo andino de pocos datos.**

## 4. Riesgos de cambiar las hipótesis
- Cambiar HE1 y HE2 después del ensayo técnico exige declarar la enmienda y que las reglas se fijan **antes** de tocar las campañas externas. El ensayo ya vio 2016–2019 de entrenamiento, por lo que esos resultados son exploratorios.
- Elegir B\* con las campañas externas sería *data snooping*.
- Un HE1 de no inferioridad frente a B\* puede parecer menos ambicioso que «superar». Se compensa con el skill score y una discusión franca del valor añadido.
- El margen δ = 5 % es una convención; justificarlo con la variabilidad interanual del rendimiento.
