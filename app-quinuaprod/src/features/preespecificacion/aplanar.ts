export type EstadoE5 = 'fijado_por_tesis' | 'fijado' | 'propuesto' | 'por_decidir_piloto' | 'por_decidir_asesor'

export interface ParametroPlano {
  seccion: string
  /** Ruta dentro de la sección, p. ej. "he2.estabilidad_w_kendall_minimo". */
  ruta: string
  valor: unknown
  estado: EstadoE5
  fuente: string
}

const esObjeto = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v)

/** Una hoja del registro E5 es un objeto {valor, estado, fuente}; lo demás son grupos anidados. */
const esHoja = (v: Record<string, unknown>) => 'valor' in v && typeof v.estado === 'string' && typeof v.fuente === 'string'

export function aplanar(secciones: Record<string, unknown>): ParametroPlano[] {
  const salida: ParametroPlano[] = []
  const recorrer = (seccion: string, ruta: string[], nodo: unknown) => {
    if (!esObjeto(nodo)) return
    if (esHoja(nodo)) {
      salida.push({ seccion, ruta: ruta.join('.'), valor: nodo.valor, estado: nodo.estado as EstadoE5, fuente: nodo.fuente as string })
      return
    }
    for (const [k, v] of Object.entries(nodo)) recorrer(seccion, [...ruta, k], v)
  }
  for (const [seccion, nodo] of Object.entries(secciones)) recorrer(seccion, [], nodo)
  return salida
}

export function contarPorEstado(parametros: ParametroPlano[]): Record<string, number> {
  const cuenta: Record<string, number> = {}
  for (const p of parametros) cuenta[p.estado] = (cuenta[p.estado] ?? 0) + 1
  return cuenta
}

/** Texto legible de un valor del registro (cadenas, números, booleanos, listas). */
export function textoValor(v: unknown): string {
  if (v === null || v === undefined) return '—'
  if (typeof v === 'boolean') return v ? 'sí' : 'no'
  if (Array.isArray(v)) return v.map(textoValor).join(', ')
  if (esObjeto(v)) return JSON.stringify(v)
  return String(v)
}
