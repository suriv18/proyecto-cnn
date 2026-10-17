import { etiquetaDe } from './etiquetas'

const LOCALE = 'es-PE'

const nf = (digits: number) =>
  new Intl.NumberFormat(LOCALE, { minimumFractionDigits: digits, maximumFractionDigits: digits })

const entero = nf(0)
const un = nf(1)
const dos = nf(2)
const tres = nf(3)

export const formatEntero = (v: number) => entero.format(v)
export const formatDecimal = (v: number, digits: 1 | 2 | 3 = 2) =>
  (digits === 1 ? un : digits === 3 ? tres : dos).format(v)
export const formatKgHa = (v: number) => `${entero.format(v)} kg/ha`
export const formatPorcentaje = (fraccion: number) => `${un.format(fraccion * 100)} %`

/**
 * Nombre legible de un identificador. Usa el diccionario de `etiquetas.ts` (con tildes y siglas)
 * y, si el id no está, lo convierte: "llenado_grano" -> "Llenado grano".
 */
export function formatEtiqueta(id: string): string {
  const conocida = etiquetaDe(id)
  if (conocida) return conocida
  const t = id.replaceAll('_', ' ')
  return t.charAt(0).toUpperCase() + t.slice(1)
}
