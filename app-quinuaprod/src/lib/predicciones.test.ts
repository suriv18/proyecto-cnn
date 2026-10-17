import { describe, expect, it } from 'vitest'
import type { Schemas } from '@/lib/api/client'
import { agregarPorProvincia, errorAbsoluto } from './predicciones'

const p = (modelo: Schemas['Modelo'], semilla: number, y_pred: number, provincia_id = 'A'): Schemas['Prediccion'] => ({
  campana_prueba: 2019,
  modelo,
  semilla,
  provincia_id,
  y_real: 1000,
  y_pred,
})

describe('agregarPorProvincia', () => {
  it('toma la mediana entre semillas por modelo', () => {
    const [fila] = agregarPorProvincia([p('cnn1d', 0, 900), p('cnn1d', 1, 1100), p('cnn1d', 2, 1000), p('B2', 0, 950)])
    expect(fila.modelos.cnn1d).toBe(1000)
    expect(fila.modelos.B2).toBe(950)
    expect(fila.y_real).toBe(1000)
  })

  it('separa provincias y las ordena por nombre', () => {
    const filas = agregarPorProvincia([p('B2', 0, 1, 'Z'), p('B2', 0, 2, 'A')])
    expect(filas.map((f) => f.provincia_id)).toEqual(['A', 'Z'])
  })

  it('errorAbsoluto es indefinido si el modelo no predijo esa provincia', () => {
    const [fila] = agregarPorProvincia([p('B2', 0, 800)])
    expect(errorAbsoluto(fila, 'B2')).toBe(200)
    expect(errorAbsoluto(fila, 'cnn1d')).toBeUndefined()
  })
})
