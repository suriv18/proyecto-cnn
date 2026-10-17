export type NivelId = 'general' | 'especifico1' | 'especifico2'

export interface Nivel {
  id: NivelId
  etiqueta: string
  problema: string
  objetivo: string
  hipotesis: string
}

export type RedaccionId = 'v3' | 'propuesta'

export interface Redaccion {
  id: RedaccionId
  etiqueta: string
  descripcion: string
  niveles: Nivel[]
}

/**
 * Textos de la matriz de consistencia. `v3` reproduce literalmente la tesis (1.2, 1.4, 3.1, 3.2 y Anexo 02).
 * `propuesta` es la redacción de docs/Propuesta-coherencia-problema-objetivos-hipotesis.md: aún no está en la tesis.
 */
export const REDACCIONES: Record<RedaccionId, Redaccion> = {
  v3: {
    id: 'v3',
    etiqueta: 'Tesis v3',
    descripcion: 'Texto vigente del proyecto de tesis.',
    niveles: [
      {
        id: 'general',
        etiqueta: 'General',
        problema: '¿En qué medida un modelo CNN-1D interpretable con SHAP permite predecir el rendimiento de quinua en la sierra peruana?',
        objetivo: 'Determinar en qué medida un modelo CNN-1D interpretable con SHAP permite predecir el rendimiento de quinua en la sierra peruana.',
        hipotesis:
          'El modelo CNN-1D interpretable con SHAP permite predecir el rendimiento de quinua en la sierra peruana al alcanzar una exactitud predictiva no inferior a la de XGBoost y a la de la mejor línea base provincial, y al generar, mediante SHAP, explicaciones agronómicamente concordantes con las fases críticas preespecificadas para la quinua de secano, cuya fidelidad al comportamiento del modelo y cuya estabilidad entre campañas se evalúan como criterios secundarios.',
      },
      {
        id: 'especifico1',
        etiqueta: 'Específico 1',
        problema: '¿En qué medida el modelado temporal mediante CNN-1D permite alcanzar exactitud predictiva en la estimación del rendimiento de quinua en la sierra peruana?',
        objetivo: 'Determinar la exactitud predictiva alcanzada mediante el modelado temporal con CNN-1D en la estimación del rendimiento de quinua en la sierra peruana.',
        hipotesis:
          'El modelado temporal mediante CNN-1D alcanza una exactitud predictiva no inferior a la de XGBoost ni a la de la mejor línea base provincial, dentro de un margen relativo preespecificado de 5 %, en la predicción del rendimiento de quinua. Como análisis exploratorio se estima la ganancia de habilidad predictiva sobre dicha línea base.',
      },
      {
        id: 'especifico2',
        etiqueta: 'Específico 2',
        problema: '¿En qué medida la explicabilidad mediante SHAP permite interpretar las predicciones del rendimiento de quinua generadas por el modelo CNN-1D en la sierra peruana?',
        objetivo: 'Determinar la interpretabilidad de las predicciones del rendimiento de quinua mediante las explicaciones SHAP generadas por el modelo CNN-1D en la sierra peruana.',
        hipotesis:
          'La explicabilidad mediante SHAP genera interpretaciones de las predicciones del rendimiento de quinua que son concordantes con las fases fenológicas críticas preespecificadas para la quinua de secano; su fidelidad al comportamiento del modelo y su estabilidad entre campañas se evalúan como criterios secundarios.',
      },
    ],
  },
  propuesta: {
    id: 'propuesta',
    etiqueta: 'Propuesta de coherencia',
    descripcion: 'Redacción propuesta, todavía no incorporada a la tesis: problema, objetivo e hipótesis usan los mismos comparadores, margen y horizonte.',
    niveles: [
      {
        id: 'general',
        etiqueta: 'General',
        problema:
          '¿Un modelo CNN-1D interpretable con SHAP predice el rendimiento de quinua por provincia y campaña en la sierra peruana con una exactitud no inferior a la de XGBoost y la de la mejor línea base (B*), y con explicaciones concordantes con las fases críticas del cultivo?',
        objetivo:
          'Evaluar si un modelo CNN-1D interpretable con SHAP permite predecir el rendimiento de quinua por provincia y campaña en la sierra peruana con una exactitud no inferior a la de XGBoost y la de la mejor línea base (B*), y con explicaciones SHAP concordantes con las fases críticas preespecificadas.',
        hipotesis:
          'El modelo CNN-1D interpretable con SHAP permite predecir el rendimiento de quinua por provincia y campaña en la sierra peruana: en el horizonte principal, su exactitud en las campañas externas no es inferior (margen relativo de 5 %) a la de XGBoost ni a la de la mejor línea base B*, y sus explicaciones SHAP son concordantes con las fases críticas preespecificadas (floración y llenado de grano). Se considera respaldada solo si se cumplen ambos criterios primarios (HE1a y HE2c); la fidelidad y la estabilidad son criterios secundarios y no compensan un incumplimiento.',
      },
      {
        id: 'especifico1',
        etiqueta: 'Específico 1',
        problema:
          '¿La exactitud predictiva del modelo CNN-1D en las campañas externas es no inferior, dentro de un margen relativo de 5 % en el RMSE, a la de XGBoost y a la de la mejor línea base B*, en el horizonte principal?',
        objetivo:
          'Evaluar si la exactitud predictiva (RMSE, MAE, rRMSE y R²) del modelo CNN-1D es no inferior, dentro de un margen relativo de 5 % en el RMSE, a la de XGBoost y a la de la mejor línea base B*, en las campañas externas y en el horizonte principal.',
        hipotesis:
          'En el horizonte principal, el modelado temporal mediante CNN-1D alcanza una exactitud predictiva no inferior a la de XGBoost ni a la de la mejor línea base B*, dentro de un margen relativo preespecificado de 5 %. Como análisis exploratorio se estima la ganancia de habilidad predictiva sobre B*.',
      },
      {
        id: 'especifico2',
        etiqueta: 'Específico 2',
        problema:
          '¿Son las explicaciones SHAP de las predicciones del CNN-1D concordantes con las fases críticas preespecificadas del cultivo y, como criterios secundarios, fieles al comportamiento del modelo y estables entre campañas?',
        objetivo:
          'Evaluar la interpretabilidad de las predicciones del CNN-1D mediante la concordancia agronómica de las explicaciones SHAP con las fases críticas preespecificadas (criterio primario), y su fidelidad por oclusión y su estabilidad entre campañas (criterios secundarios).',
        hipotesis:
          'La explicabilidad mediante SHAP genera interpretaciones del rendimiento de quinua concordantes con las fases fenológicas críticas preespecificadas (criterio primario); su fidelidad al comportamiento del modelo y su estabilidad entre campañas se evalúan como criterios secundarios.',
      },
    ],
  },
}

/** Definiciones que la propuesta pide escribir una sola vez y reutilizar. */
export const DEFINICIONES = [
  { termino: 'B*', texto: 'La mejor entre B2 (línea base provincial con tendencia) y B4 (persistencia de la última campaña), elegida con la validación interna antes de evaluar las campañas externas.' },
  { termino: 'Horizonte principal', texto: '30 días antes de la cosecha esperada. El horizonte secundario es el cierre de la floración.' },
  { termino: 'Interpretable', texto: 'Que sus explicaciones SHAP superan el protocolo de interpretabilidad de la sección 4.15.' },
  { termino: 'Benchmarks', texto: 'Cuatro (B1-B4). B2 y B4 son confirmatorios porque definen B*; B1 y B3 son descriptivos.' },
] as const
