import type { Schemas } from '@/lib/api/client'

/** q_f(R) = RMSE(modelo, f) / RMSE(R, f) − 1 (tesis v3, 3.3.1). Negativo: el modelo se equivoca menos que R. */
export const diferenciaRelativa = (rmseModelo: number, rmseReferencia: number) => rmseModelo / rmseReferencia - 1

/** SS_f = 1 − MSE(modelo) / MSE(R) = 1 − (RMSE(modelo) / RMSE(R))² (Murphy, 1988; tesis v3, 3.3.2). */
export const gananciaDeHabilidad = (rmseModelo: number, rmseReferencia: number) =>
  1 - (rmseModelo / rmseReferencia) ** 2

export interface FaseMaxima {
  fase: string
  /** Diferencia relativa entre la mayor contribución y la segunda: mide cuán fácil es que cambie el orden. */
  brecha: number
}

export function faseMaxima(contribuciones: Record<string, number>): FaseMaxima {
  const orden = Object.entries(contribuciones).sort((a, b) => b[1] - a[1])
  const [fase, mayor] = orden[0]
  return { fase, brecha: orden.length > 1 ? (mayor - orden[1][1]) / mayor : 1 }
}

/** Rango (1 = el mayor) del aumento del RMSE de una fase entre todas; los empates cuentan en su contra (3.3.4). */
export function rangoDelAumento(fase: string, aumentos: Record<string, number>): number {
  const valor = aumentos[fase]
  return Object.values(aumentos).filter((v) => v >= valor).length
}

export interface FilaHE1 {
  campana: number
  qXgboost?: number
  qB2?: number
  qB4?: number
  ssB2?: number
  ssB4?: number
}

/** Evidencia descriptiva de HE1 con los RMSE del piloto. No decide la hipótesis. */
export function evidenciaHE1(y1: Schemas['IndicadorY1'][]): FilaHE1[] {
  const campanas = [...new Set(y1.map((f) => f.campana_prueba))].sort((a, b) => a - b)
  return campanas.map((campana) => {
    const rmse = (m: Schemas['Modelo']) => y1.find((f) => f.modelo === m && f.campana_prueba === campana)?.rmse
    const cnn = rmse('cnn1d')
    const vs = (m: Schemas['Modelo'], f: (a: number, b: number) => number) => {
      const r = rmse(m)
      return cnn === undefined || r === undefined ? undefined : f(cnn, r)
    }
    return {
      campana,
      qXgboost: vs('xgboost', diferenciaRelativa),
      qB2: vs('B2', diferenciaRelativa),
      qB4: vs('B4', diferenciaRelativa),
      ssB2: vs('B2', gananciaDeHabilidad),
      ssB4: vs('B4', gananciaDeHabilidad),
    }
  })
}

export interface FilaHE2 {
  campana: number
  fase: string
  brecha: number
  /** ¿La fase de máxima contribución pertenece al conjunto crítico preespecificado? */
  concuerda: boolean
  rango: number
}

/** Evidencia descriptiva de HE2 (ejecución técnica). `criticas` viene de E5, no del código. */
export function evidenciaHE2(he2: Schemas['Explicabilidad']['he2_tecnico'], criticas: string[]): FilaHE2[] {
  const contribuciones = he2.contribuciones as Record<string, Record<string, number>>
  const aumentos = he2.aumentos_rmse as Record<string, Record<string, number>>
  return Object.keys(contribuciones)
    .sort()
    .map((c) => {
      const { fase, brecha } = faseMaxima(contribuciones[c])
      return { campana: Number(c), fase, brecha, concuerda: criticas.includes(fase), rango: rangoDelAumento(fase, aumentos[c]) }
    })
}

