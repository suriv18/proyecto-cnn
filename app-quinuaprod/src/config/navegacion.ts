export type GrupoNavegacion = 'Resultados' | 'Interpretación' | 'Referencia' | 'Herramienta'

export interface ItemNavegacion {
  ruta: string
  etiqueta: string
  descripcion: string
  grupo: GrupoNavegacion
}

export const NAVEGACION: readonly ItemNavegacion[] = [
  { ruta: '/', etiqueta: 'Resumen', descripcion: 'Estado del piloto y embudo de depuración', grupo: 'Resultados' },
  { ruta: '/cobertura', etiqueta: 'Cobertura', descripcion: 'Provincias, campañas y exclusiones', grupo: 'Resultados' },
  { ruta: '/predicciones', etiqueta: 'Predicción vs observado', descripcion: 'Rendimiento predicho por modelo', grupo: 'Resultados' },
  { ruta: '/indicadores', etiqueta: 'Indicadores', descripcion: 'X1, X2 e Y1 del Anexo 01', grupo: 'Resultados' },
  { ruta: '/explicabilidad', etiqueta: 'Explicabilidad', descripcion: 'SHAP por fase y predictor', grupo: 'Interpretación' },
  { ruta: '/sensibilidades', etiqueta: 'Sensibilidades', descripcion: 'Variantes y estratos de quinua', grupo: 'Interpretación' },
  { ruta: '/hipotesis', etiqueta: 'Hipótesis y objetivos', descripcion: 'Matriz de consistencia y criterios de decisión', grupo: 'Referencia' },
  { ruta: '/preespecificacion', etiqueta: 'Registro E5', descripcion: 'Parámetros preespecificados', grupo: 'Referencia' },
  { ruta: '/inferencia', etiqueta: 'Predicción interactiva', descripcion: 'Inferencia con el CNN-1D', grupo: 'Herramienta' },
]

export const GRUPOS: readonly GrupoNavegacion[] = ['Resultados', 'Interpretación', 'Referencia', 'Herramienta']
