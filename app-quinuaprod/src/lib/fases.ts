/** Las cinco fases fenológicas del tensor, en el orden del ciclo del cultivo, con su color de tema. */
export const FASES = ['emergencia', 'desarrollo_vegetativo', 'floracion', 'llenado_grano', 'madurez'] as const

export type FaseId = (typeof FASES)[number]

export const colorFase = (fase: string): string => {
  const i = FASES.indexOf(fase as FaseId)
  return i < 0 ? 'var(--reference)' : `var(--fase-${i + 1})`
}

const CORTA: Record<FaseId, string> = {
  emergencia: 'Emergencia',
  desarrollo_vegetativo: 'Desarrollo',
  floracion: 'Floración',
  llenado_grano: 'Llenado',
  madurez: 'Madurez',
}

export const etiquetaFaseCorta = (fase: string): string => CORTA[fase as FaseId] ?? fase
