import { describe, expect, it } from 'vitest'
import explicabilidad from '@/testing/mocks/fixtures/explicabilidad.json'
import indicadoresY1 from '@/testing/mocks/fixtures/indicadores_y1.json'
import type { Schemas } from '@/lib/api/client'
import {
  diferenciaRelativa,
  evidenciaHE1,
  evidenciaHE2,
  faseMaxima,
  gananciaDeHabilidad,
  rangoDelAumento,
} from './hipotesis'

describe('diferenciaRelativa y gananciaDeHabilidad', () => {
  it('q es 0 si el error es igual, negativo si el modelo mejora', () => {
    expect(diferenciaRelativa(100, 100)).toBe(0)
    expect(diferenciaRelativa(95, 100)).toBeCloseTo(-0.05)
    expect(diferenciaRelativa(105, 100)).toBeCloseTo(0.05)
  })

  it('SS usa el cociente de MSE, no de RMSE', () => {
    expect(gananciaDeHabilidad(100, 100)).toBe(0)
    expect(gananciaDeHabilidad(50, 100)).toBeCloseTo(0.75)
    expect(gananciaDeHabilidad(110, 100)).toBeLessThan(0)
  })
})

describe('faseMaxima y rangoDelAumento', () => {
  it('devuelve la fase mayor y su brecha relativa con la segunda', () => {
    const r = faseMaxima({ a: 0.1, b: 0.2, c: 0.15 })
    expect(r.fase).toBe('b')
    expect(r.brecha).toBeCloseTo(0.25)
  })

  it('el rango 1 es el mayor aumento y los empates cuentan en contra', () => {
    const aumentos = { a: 0.3, b: 0.1, c: 0.3, d: 0.2 }
    expect(rangoDelAumento('d', aumentos)).toBe(3)
    expect(rangoDelAumento('a', aumentos)).toBe(2)
    expect(rangoDelAumento('b', aumentos)).toBe(4)
  })
})

describe('con los datos reales del piloto', () => {
  const y1 = indicadoresY1 as Schemas['IndicadorY1'][]

  it('HE1: reproduce q frente a XGBoost, B2 y B4 por campaña', () => {
    const [f2019, f2020] = evidenciaHE1(y1)
    expect([f2019.campana, f2020.campana]).toEqual([2019, 2020])
    expect(f2019.qXgboost).toBeCloseTo(-0.057, 3)
    expect(f2020.qXgboost).toBeCloseTo(0.047, 3)
    expect(f2020.qB4).toBeCloseTo(0.143, 3)
    expect(f2019.ssB2).toBeCloseTo(0.02, 2)
  })

  it('HE2: la fase máxima de 2018 es desarrollo vegetativo y su brecha es mínima', () => {
    const filas = evidenciaHE2(explicabilidad.he2_tecnico as Schemas['Explicabilidad']['he2_tecnico'], ['floracion', 'llenado_grano'])
    const f2018 = filas.find((f) => f.campana === 2018)!
    const f2019 = filas.find((f) => f.campana === 2019)!
    expect(f2018.fase).toBe('desarrollo_vegetativo')
    expect(f2018.concuerda).toBe(false)
    expect(f2018.brecha).toBeLessThan(0.05)
    expect(f2019.fase).toBe('floracion')
    expect(f2019.concuerda).toBe(true)
  })

  it('HE2: la concordancia sale del conjunto crítico recibido, no de una constante', () => {
    const he2 = explicabilidad.he2_tecnico as Schemas['Explicabilidad']['he2_tecnico']
    const solo = evidenciaHE2(he2, ['desarrollo_vegetativo'])
    expect(solo.find((f) => f.campana === 2018)!.concuerda).toBe(true)
    expect(solo.find((f) => f.campana === 2019)!.concuerda).toBe(false)
  })
})
