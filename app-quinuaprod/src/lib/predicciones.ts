import type { Schemas } from '@/lib/api/client'
import { mediana } from './estadistica'

type Modelo = Schemas['Modelo']

export interface FilaProvincia {
  provincia_id: string
  y_real: number
  /** Predicción por modelo: mediana entre semillas (los modelos deterministas tienen una sola). */
  modelos: Partial<Record<Modelo, number>>
}

export function agregarPorProvincia(items: Schemas['Prediccion'][]): FilaProvincia[] {
  const grupos = new Map<string, { y_real: number; porModelo: Map<Modelo, number[]> }>()
  for (const p of items) {
    const g = grupos.get(p.provincia_id) ?? { y_real: p.y_real, porModelo: new Map() }
    g.porModelo.set(p.modelo, [...(g.porModelo.get(p.modelo) ?? []), p.y_pred])
    grupos.set(p.provincia_id, g)
  }
  return [...grupos.entries()]
    .map(([provincia_id, g]) => ({
      provincia_id,
      y_real: g.y_real,
      modelos: Object.fromEntries([...g.porModelo].map(([m, xs]) => [m, mediana(xs)])) as FilaProvincia['modelos'],
    }))
    .sort((a, b) => a.provincia_id.localeCompare(b.provincia_id))
}

export function errorAbsoluto(fila: FilaProvincia, modelo: Modelo): number | undefined {
  const y = fila.modelos[modelo]
  return y === undefined ? undefined : Math.abs(y - fila.y_real)
}
