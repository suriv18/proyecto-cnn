import { describe, expect, it } from 'vitest'
import { formatEtiqueta, formatKgHa, formatPorcentaje } from './format'

describe('format', () => {
  it('formatea rendimiento en kg/ha sin decimales', () => {
    expect(formatKgHa(1806.42)).toMatch(/^1\D?806 kg\/ha$/)
  })

  it('formatea fracciones como porcentaje con un decimal', () => {
    expect(formatPorcentaje(0.5172)).toMatch(/^51[.,]7 %$/)
  })

  it('convierte identificadores en etiquetas legibles', () => {
    expect(formatEtiqueta('llenado_grano')).toBe('Llenado de grano')
    expect(formatEtiqueta('ndvi_max')).toBe('NDVI máximo')
    expect(formatEtiqueta('floracion')).toBe('Floración')
  })

  it('convierte ids desconocidos sin romperse', () => {
    expect(formatEtiqueta('algo_nuevo')).toBe('Algo nuevo')
  })
})
