import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { AppProviders } from '@/app/providers'
import type { Schemas } from '@/lib/api/client'
import { MapaShap } from './MapaShap'
import { ResultadoInferencia } from './ResultadoInferencia'

const VARIABLES = ['precipitacion', 'ndvi_max']
const FASES = ['emergencia', 'floracion', 'madurez']
const VALORES = [
  [1.25, -2.5, 0],
  [-0.04, 10, -7.35],
]

describe('MapaShap', () => {
  it('escribe cada valor con su signo, de modo que el color no sea la única señal', () => {
    render(<MapaShap valores={VALORES} variables={VARIABLES} fases={FASES} />)
    const tabla = screen.getByRole('region', { name: 'Mapa de valores SHAP' })
    const filas = within(tabla).getAllByRole('row').slice(1)
    expect(filas).toHaveLength(2)
    expect(filas[0]).toHaveTextContent('Precipitación')
    expect(filas[0]).toHaveTextContent(/\+1[.,]3/) // 1,25 redondeado a un decimal
    expect(filas[0]).toHaveTextContent(/-2[.,]5/)
    expect(filas[1]).toHaveTextContent('NDVI máximo')
    expect(filas[1]).toHaveTextContent(/\+10[.,]0/)
    expect(filas[1]).toHaveTextContent(/-7[.,]4/)
  })

  it('nombra las fases para lectores de pantalla y aclara que no es causalidad', () => {
    render(<MapaShap valores={VALORES} variables={VARIABLES} fases={FASES} />)
    expect(screen.getByRole('columnheader', { name: /Floración/ })).toBeInTheDocument()
    expect(screen.getByText(/no una relación causal/)).toBeInTheDocument()
  })

  it('pinta con más intensidad el valor mayor y deja sin fondo el cero', () => {
    render(<MapaShap valores={VALORES} variables={VARIABLES} fases={FASES} />)
    const celdas = within(screen.getByRole('region', { name: 'Mapa de valores SHAP' })).getAllByRole('cell')
    const porTexto = (t: RegExp) => celdas.find((c) => t.test(c.textContent ?? ''))!
    expect(porTexto(/\+10[.,]0/).style.background).toContain('40%')
    expect(porTexto(/\+10[.,]0/).style.background).toContain('--shap-positivo')
    expect(porTexto(/-7[.,]4/).style.background).toContain('--shap-negativo')
    expect(porTexto(/^0[.,]0$/).style.background).toBe('')
  })

  it('un valor que se redondea a cero se escribe sin signo y sin fondo', () => {
    render(<MapaShap valores={VALORES} variables={VARIABLES} fases={FASES} />)
    const ceros = within(screen.getByRole('region', { name: 'Mapa de valores SHAP' })).getAllByRole('cell').filter((c) => /^0[.,]0$/.test(c.textContent ?? ''))
    expect(ceros).toHaveLength(2) // el 0 exacto y el -0,04
    expect(ceros.every((c) => c.style.background === '')).toBe(true)
  })

  it('no falla si todos los valores son cero', () => {
    render(<MapaShap valores={[[0, 0], [0, 0]]} variables={VARIABLES} fases={FASES.slice(0, 2)} />)
    expect(screen.getAllByRole('cell')).toHaveLength(4)
  })
})

describe('ResultadoInferencia con SHAP', () => {
  const base: Schemas['ResultadoInferencia'] = {
    y_pred: 2000, y_b2: 1900, anomalia: 100, y_real: null, shap: null, modelo_version: 'cnn1d-abc', vinculante: false,
  }

  it('muestra el mapa cuando la respuesta trae SHAP', async () => {
    const shap = Array.from({ length: 6 }, () => [0.5, -0.5, 1, 2, -3])
    render(<AppProviders><ResultadoInferencia r={{ ...base, shap }} /></AppProviders>)
    expect(await screen.findByRole('region', { name: 'Mapa de valores SHAP' })).toBeInTheDocument()
    expect(screen.queryByText(/SHAP por celda no está disponible/)).not.toBeInTheDocument()
  })

  it('avisa que no está disponible cuando shap es null', () => {
    render(<AppProviders><ResultadoInferencia r={base} /></AppProviders>)
    expect(screen.getByText(/SHAP por celda no está disponible/)).toBeInTheDocument()
    expect(screen.queryByRole('region', { name: 'Mapa de valores SHAP' })).not.toBeInTheDocument()
  })
})
