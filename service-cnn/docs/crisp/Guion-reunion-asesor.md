# Guion para la reunión con el asesor

Duración sugerida: 45 minutos. Orden por impacto: primero lo que puede cambiar el diseño, al final lo que se confirma.

## Qué llevar
- `docs/crisp/Resumen-para-el-asesor.md` (o la página publicada) y `docs/crisp/Cambios-tesis-v2-v3.html`.
- La tesis `docs/Proyecto_Tesis_Maestria_UNMSM_WMSR_v3.docx`.
- El informe del piloto: `docs/crisp/Piloto-oficial-informe.md`.
- El pre-registro: `docs/crisp/E5-preespecificacion-BORRADOR.md`.

## 1. Apertura (3 min)
- Estado: la preparación de datos, el modelado y el piloto están hechos; el código de evaluación está listo y probado. **No se ha tocado ninguna campaña externa (2021–2025).**
- Qué se necesita hoy: revisar las decisiones y autorizar el congelamiento del pre-registro.

## 2. Lo que puede cambiar el diseño (20 min)

**2.1 Reformulación de HE1 y HE2 (10 min).**
- Problema: con 5 campañas externas, exigir superioridad sobre B2 obliga a ganar las 5, y el ensayo muestra que el modelo no se distingue de las líneas base.
- Propuesta (ya en la v3): HE1a = no inferioridad frente a XGBoost y frente a la mejor línea base B*, con regla exacta de signos (menos de 5 % en todas las campañas); la ganancia sobre B* es exploratoria. HE2 = la concordancia agronómica es el criterio primario; fidelidad y estabilidad son secundarias.
- Alternativa si no la acepta: volver a la superioridad frente a B2 y a los tres criterios de HE2 con Holm.
- **Pregunta:** ¿aprueba la reformulación?

**2.2 Qué significa B\* (5 min).**
- La persistencia (repetir la última campaña) fue la referencia más fuerte del piloto: 171,8 kg/ha en 2020 frente a 193,8 de B2 y 196,4 de la CNN-1D.
- Si B* es B4, la CNN-1D queda +14 % en 2020 y HE1a no se cumpliría; si es B2, se cumple.
- **Pregunta:** ¿se acepta elegir B* con la validación interna, antes de ver las campañas externas?

**2.3 Ventana inicial y potencia (5 min).**
- Ventana de 5 campañas deja 5 externas (2021–2025): p mínimo de Wilcoxon 1/32 = 0,031, y HE1a exige que todas cumplan.
- Se evaluó una ventana de 4 (6 externas) y se descartó: la tendencia de B2 necesita 4 campañas de historia y obligaba a recalibrar la máscara.
- **Pregunta:** ¿confirma la ventana de 5?

## 3. Lo ya decidido, para su revisión (10 min)
Las 8 decisiones del gate de F2 están confirmadas por el autor; solo pide su visto bueno:
1. Periodo 2016–2025 (no existe fuente provincial anterior a 2015).
2. Provincias de transición incluidas y marcadas, con sensibilidad sin ellas.
3. Ciclos fuera de la ventana de extracción excluidos (29 observaciones).
4–6. Máscara agrícola: altitud 2500–4000 m y pendiente máxima 14,04°.
7. Amplitud mínima de NDVI: percentil 5 = 0,323 (sesgo regional en Arequipa y Puno).
8. Ventana inicial de 5 campañas.

## 4. Resultados del piloto (5 min)
- La prueba de fuga pasa en los 5 modelos.
- Con una línea base de tendencia lineal, el error de los modelos en 2020 baja de 231–235 a 188–196 kg/ha: la deriva temporal importa.
- SHAP: ndvi_max es el predictor más importante; el orden de las fases es inestable con solo 2 pliegos.
- Limitación a declarar: el mejor ajuste queda cerca de la línea base y varios hiperparámetros quedan en el borde del espacio.

## 5. Pendientes que dependen de su criterio (5 min)
- **Sensibilidades:** por horizonte (cierre de floración), fuente climática, participación de quinua y campañas extremas. Requieren corridas nuevas. **Pregunta:** ¿cuáles son imprescindibles para la tesis?
- **Serie 2006–2015:** solicitar a MIDAGRI la base provincial por acceso a la información pública. **Pregunta:** ¿lo respalda?
- **Cambio de enfoque posible:** si se quisiera capturar la deriva temporal de otra forma (por ejemplo, el año como covariable), cambiaría el modelo y reabriría la comparación.

## 6. Cierre (2 min) — qué se pide
1. Visto bueno a las 8 decisiones, o cuáles cambia.
2. Aprobación de HE1 y HE2 reformulados.
3. Autorización para **congelar el pre-registro** (fecha, commit y sha256 de los datos) y evaluar las campañas externas, que **solo se pueden evaluar una vez**.

## Qué pasa según su respuesta
| Si... | Entonces |
|---|---|
| Aprueba todo | Congelar E5, evaluar 2021–2025 y escribir resultados |
| Cambia HE1 o HE2 | Ajustar el código (ya hay una alternativa documentada), repetir el piloto (minutos) y volver a pasar E5 |
| Cambia la ventana o un umbral de la máscara | Repetir preprocesamiento y piloto; si cambia la máscara, re-extraer el NDVI (horas) |
