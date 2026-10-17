import { describe, expect, it } from 'vitest'
import { escalaAgradable } from './escala'

describe('escalaAgradable', () => {
  it('redondea el máximo hacia arriba a una marca limpia', () => {
    const { max, marcas } = escalaAgradable(2727.4095195525997)
    expect(max).toBeGreaterThanOrEqual(2727.41)
    expect(marcas[0]).toBe(0)
    expect(marcas.at(-1)).toBe(max)
    expect(marcas.length).toBeGreaterThanOrEqual(4)
    expect(marcas.length).toBeLessThanOrEqual(8)
  })

  it('usa pasos de 1, 2 o 5 por potencias de diez', () => {
    expect(escalaAgradable(800).marcas).toEqual([0, 200, 400, 600, 800])
    const { marcas } = escalaAgradable(0.31)
    expect(marcas).toHaveLength(5)
    marcas.forEach((m, i) => expect(m).toBeCloseTo(i * 0.1))
  })

  it('tolera máximos no válidos', () => {
    expect(escalaAgradable(0)).toEqual({ max: 1, marcas: [0, 1] })
    expect(escalaAgradable(Number.NaN).marcas).toEqual([0, 1])
  })
})
