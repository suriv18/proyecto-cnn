import { afterEach, describe, expect, it, vi } from 'vitest'

async function cargar() {
  vi.resetModules()
  return import('./env')
}

afterEach(() => vi.unstubAllEnvs())

describe('config/env', () => {
  it('usa la URL de la API configurada', async () => {
    vi.stubEnv('VITE_API_URL', 'https://api.ejemplo.org/api/v1')
    expect((await cargar()).API_URL).toBe('https://api.ejemplo.org/api/v1')
  })

  it('sin variable, llama a /api/v1 en el mismo origen', async () => {
    vi.stubEnv('VITE_API_URL', undefined as unknown as string)
    expect((await cargar()).API_URL).toBe('/api/v1')
  })

  it('una variable vacía (build de Docker sin argumento) cuenta como sin definir', async () => {
    vi.stubEnv('VITE_API_URL', '')
    expect((await cargar()).API_URL).toBe('/api/v1')
  })

  it('solo el texto «true» activa los mocks', async () => {
    vi.stubEnv('VITE_USE_MOCKS', 'true')
    expect((await cargar()).USE_MOCKS).toBe(true)
    vi.stubEnv('VITE_USE_MOCKS', '1')
    expect((await cargar()).USE_MOCKS).toBe(false)
  })
})
