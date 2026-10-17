import type { Schemas } from '@/lib/api/client'

export type Modelo = Schemas['Modelo']

export const ORDEN_MODELOS: Modelo[] = ['cnn1d', 'xgboost', 'random_forest', 'elastic_net', 'svr', 'B1', 'B2', 'B3', 'B4']

const ETIQUETAS: Record<Modelo, string> = {
  cnn1d: 'CNN-1D',
  xgboost: 'XGBoost',
  random_forest: 'Random Forest',
  elastic_net: 'Elastic-Net',
  svr: 'SVR',
  B1: 'B1',
  B2: 'B2',
  B3: 'B3',
  B4: 'B4',
}

export const etiquetaModelo = (m: Modelo) => ETIQUETAS[m]

/** Las referencias B1-B4 no son modelos entrenados: se distinguen en las tablas. */
export const esReferencia = (m: Modelo) => m.startsWith('B')
