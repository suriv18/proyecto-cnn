import { afterEach, describe, expect, it, vi } from 'vitest'
import { CLAVE_TEMA, esOscuro, guardarTema, leerTema } from './theme'

afterEach(() => {
  localStorage.clear()
  vi.restoreAllMocks()
})

describe('theme', () => {
  it('esOscuro respeta la preferencia del sistema solo en modo sistema', () => {
    expect(esOscuro('sistema', true)).toBe(true)
    expect(esOscuro('sistema', false)).toBe(false)
    expect(esOscuro('claro', true)).toBe(false)
    expect(esOscuro('oscuro', false)).toBe(true)
  })

  it('guarda y lee la preferencia', () => {
    guardarTema('oscuro')
    expect(localStorage.getItem(CLAVE_TEMA)).toBe('oscuro')
    expect(leerTema()).toBe('oscuro')
  })

  it('ignora valores inválidos y vuelve a sistema', () => {
    localStorage.setItem(CLAVE_TEMA, 'fucsia')
    expect(leerTema()).toBe('sistema')
  })

  it('no falla si localStorage lanza', () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('bloqueado')
    })
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('bloqueado')
    })
    expect(leerTema()).toBe('sistema')
    expect(() => guardarTema('claro')).not.toThrow()
  })
})
