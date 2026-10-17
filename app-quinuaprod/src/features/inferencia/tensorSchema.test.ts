import { describe, expect, it } from 'vitest'
import { tensorSchema } from './tensorSchema'

const valido = () => ({
  valores: Array.from({ length: 6 }, () => [1, 2, 3, 4, 5]),
  altitud_media_m: 3800,
  prop_quinua: 0.1,
})

describe('tensorSchema', () => {
  it('acepta un tensor 6x5 con covariables', () => {
    expect(tensorSchema.safeParse(valido()).success).toBe(true)
  })

  it('rechaza una forma distinta de 6x5', () => {
    const t = valido()
    t.valores.pop()
    expect(tensorSchema.safeParse(t).success).toBe(false)
    const u = valido()
    u.valores[0] = [1, 2, 3]
    expect(tensorSchema.safeParse(u).success).toBe(false)
  })

  it('rechaza celdas vacías (NaN) con un mensaje claro', () => {
    const t = valido()
    t.valores[2][3] = Number.NaN
    const r = tensorSchema.safeParse(t)
    expect(r.success).toBe(false)
    expect(r.error?.issues[0].message).toBe('Ingrese un número')
  })

  it('rechaza covariables negativas', () => {
    expect(tensorSchema.safeParse({ ...valido(), prop_quinua: -0.1 }).success).toBe(false)
    expect(tensorSchema.safeParse({ ...valido(), altitud_media_m: -1 }).success).toBe(false)
  })
})
