import { describe, expect, it } from 'vitest'
import { createApiClient } from '@/lib/api/client'
import { mediana } from './handlers'
import meta from './fixtures/meta.json'
import predicciones from './fixtures/predicciones.json'
import exclusiones from './fixtures/exclusiones.json'
import indicadoresY1 from './fixtures/indicadores_y1.json'
import sensibilidades from './fixtures/sensibilidades.json'

const api = createApiClient(new URL('/api/v1', location.origin).href)
const EXTERNAS = [2021, 2022, 2023, 2024, 2025]

describe('regla de honestidad: campañas externas protegidas', () => {
  it('las fixtures no contienen campañas 2021-2025', () => {
    const campanas = new Set<number>([
      ...predicciones.map((p) => p.campana_prueba),
      ...exclusiones.map((e) => e.campana),
      ...indicadoresY1.map((i) => i.campana_prueba),
      ...sensibilidades.map((s) => s.campana_prueba),
    ])
    for (const c of EXTERNAS) expect(campanas.has(c)).toBe(false)
    expect(meta.campanas_permitidas.some((c) => EXTERNAS.includes(c))).toBe(false)
  })

  it('/predicciones rechaza una campaña externa con problem+json', async () => {
    const { data, error, response } = await api.GET('/predicciones', {
      params: { query: { campana_prueba: 2021 } },
    })
    expect(data).toBeUndefined()
    expect(response.status).toBe(422)
    expect(response.headers.get('content-type')).toContain('application/problem+json')
    expect(error?.status).toBe(422)
  })
})

describe('contrato con datos reales', () => {
  it('GET /meta describe un piloto con 5 variables y 5 fases', async () => {
    const { data } = await api.GET('/meta')
    expect(data?.orden_variables).toHaveLength(6)
    expect(data?.orden_fases).toHaveLength(5)
    expect(data?.campanas_de_prueba).toEqual([2019, 2020])
  })

  it('GET /provincias filtra por clasificación', async () => {
    const todas = await api.GET('/provincias')
    expect(todas.data).toHaveLength(80)
    const alto = await api.GET('/provincias', { params: { query: { clasificacion: 'altoandina' } } })
    expect(alto.data?.length).toBeGreaterThan(0)
    expect(alto.data?.every((p) => p.clasificacion === 'altoandina')).toBe(true)
  })

  it('GET /predicciones pagina y filtra', async () => {
    const { data } = await api.GET('/predicciones', {
      params: { query: { modelo: 'B2', campana_prueba: 2019, limit: 10 } },
    })
    expect(data?.items).toHaveLength(10)
    expect(data!.total).toBeGreaterThanOrEqual(10)
    expect(data?.items.every((p) => p.modelo === 'B2' && p.campana_prueba === 2019)).toBe(true)
  })

  it('GET /cobertura/embudo mantiene N0 >= N1 >= ... y N5 nulo', async () => {
    const { data } = await api.GET('/cobertura/embudo')
    const ns = data!.filter((e) => e.n !== null).map((e) => e.n as number)
    expect([...ns].sort((a, b) => b - a)).toEqual(ns)
    expect(data!.at(-1)).toEqual({ nivel: 'N5', n: null })
  })

  it('GET /indicadores/y1 trae RMSE positivo para todos los modelos', async () => {
    const { data } = await api.GET('/indicadores/y1')
    expect(data!.length).toBeGreaterThan(0)
    expect(data!.every((i) => i.rmse > 0)).toBe(true)
  })

  it('GET /preespecificacion expone parámetros con estado y fuente', async () => {
    const { data } = await api.GET('/preespecificacion')
    expect(data?.metadatos).toBeDefined()
    expect(Object.keys(data!.secciones)).toContain('hipotesis')
  })
})

describe('POST /predicciones/inferir', () => {
  const ref = predicciones.find((p) => p.modelo === 'cnn1d' && p.campana_prueba === 2019)!

  it('devuelve la predicción real del piloto, sin SHAP inventado', async () => {
    const { data } = await api.POST('/predicciones/inferir', {
      body: { provincia_id: ref.provincia_id, campana: 2019 },
    })
    const esperado = mediana(
      predicciones
        .filter((p) => p.modelo === 'cnn1d' && p.campana_prueba === 2019 && p.provincia_id === ref.provincia_id)
        .map((p) => p.y_pred),
    )
    expect(data?.y_pred).toBeCloseTo(esperado)
    expect(data?.anomalia).toBeCloseTo(data!.y_pred - data!.y_b2)
    expect(data?.shap).toBeNull()
  })

  const tensor = { valores: Array.from({ length: 6 }, () => [0, 0, 0, 0, 0]), altitud_media_m: 3800, prop_quinua: 0.1 }

  it('501 si se envía un tensor editado (los mocks no incluyen el modelo)', async () => {
    const { response } = await api.POST('/predicciones/inferir', {
      body: { provincia_id: ref.provincia_id, campana: 2020, tensor },
    })
    expect(response.status).toBe(501)
  })

  it('con un tensor exige provincia y campaña, y rechaza campañas externas', async () => {
    const sinProvincia = await api.POST('/predicciones/inferir', { body: { campana: 2020, tensor } })
    expect(sinProvincia.response.status).toBe(422)
    const externa = await api.POST('/predicciones/inferir', { body: { provincia_id: ref.provincia_id, campana: 2022, tensor } })
    expect(externa.response.status).toBe(422)
  })

  it('422 para una campaña no permitida y 404 fuera de las campañas de prueba', async () => {
    const ext = await api.POST('/predicciones/inferir', { body: { provincia_id: ref.provincia_id, campana: 2022 } })
    expect(ext.response.status).toBe(422)
    const fuera = await api.POST('/predicciones/inferir', { body: { provincia_id: ref.provincia_id, campana: 2016 } })
    expect(fuera.response.status).toBe(404)
  })
})

describe('mediana', () => {
  it('maneja cantidad par e impar', () => {
    expect(mediana([3, 1, 2])).toBe(2)
    expect(mediana([4, 1, 3, 2])).toBe(2.5)
  })
})
