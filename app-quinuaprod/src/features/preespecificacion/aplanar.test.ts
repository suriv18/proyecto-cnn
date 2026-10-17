import { describe, expect, it } from 'vitest'
import preespecificacion from '@/testing/mocks/fixtures/preespecificacion.json'
import { aplanar, contarPorEstado, textoValor } from './aplanar'

describe('aplanar', () => {
  it('encuentra hojas en cualquier profundidad y conserva la ruta', () => {
    const planos = aplanar({
      hipotesis: {
        alfa: { valor: 0.05, estado: 'fijado_por_tesis', fuente: '3.3' },
        he2: { w: { valor: 0.7, estado: 'fijado', fuente: 'x' } },
      },
    })
    expect(planos.map((p) => p.ruta)).toEqual(['alfa', 'he2.w'])
    expect(planos.every((p) => p.seccion === 'hipotesis')).toBe(true)
  })

  it('ignora nodos que no son parámetros', () => {
    expect(aplanar({ s: { lista: [1, 2], texto: 'x' } })).toEqual([])
  })

  it('cuenta por estado', () => {
    const planos = aplanar({ s: { a: { valor: 1, estado: 'fijado', fuente: '' }, b: { valor: 2, estado: 'fijado', fuente: '' } } })
    expect(contarPorEstado(planos)).toEqual({ fijado: 2 })
  })

  it('textoValor formatea listas y booleanos', () => {
    expect(textoValor(['a', 'b'])).toBe('a, b')
    expect(textoValor(true)).toBe('sí')
    expect(textoValor(null)).toBe('—')
  })

  it('el registro real tiene parámetros y ninguno queda pendiente (piloto oficial ronda 2)', () => {
    const planos = aplanar(preespecificacion.secciones)
    expect(planos.length).toBeGreaterThan(50)
    const cuenta = contarPorEstado(planos)
    expect(cuenta.propuesto ?? 0).toBe(0)
    expect(cuenta.por_decidir_asesor ?? 0).toBe(0)
    expect(cuenta.por_decidir_piloto ?? 0).toBe(0)
  })
})
