import { http, HttpResponse } from 'msw'
import type { Schemas } from '@/lib/api/client'
import { mediana } from '@/lib/estadistica'
import meta from './fixtures/meta.json'
import provincias from './fixtures/provincias.json'
import embudo from './fixtures/embudo.json'
import exclusiones from './fixtures/exclusiones.json'
import predicciones from './fixtures/predicciones.json'
import indicadoresY1 from './fixtures/indicadores_y1.json'
import indicadoresX1X2 from './fixtures/indicadores_x1x2.json'
import variabilidad from './fixtures/variabilidad_semillas.json'
import explicabilidad from './fixtures/explicabilidad.json'
import sensibilidades from './fixtures/sensibilidades.json'
import estratos from './fixtures/sensibilidades_estratos.json'
import preespecificacion from './fixtures/preespecificacion.json'

// Las fixtures son una instantánea REAL de service-cnn/ (scripts/export_fixtures.py).
// Cada una se asigna a su tipo del contrato para que tsc detecte derivas respecto de openapi.yaml.
const DB = {
  meta: meta as Schemas['Meta'],
  provincias: provincias as Schemas['Provincia'][],
  embudo: embudo as Schemas['NivelEmbudo'][],
  exclusiones: exclusiones as Schemas['Exclusion'][],
  predicciones: predicciones as Schemas['Prediccion'][],
  indicadoresY1: indicadoresY1 as Schemas['IndicadorY1'][],
  indicadoresX1X2: indicadoresX1X2 as Schemas['IndicadoresX1X2'],
  variabilidad: variabilidad as Schemas['VariabilidadSemilla'][],
  explicabilidad: explicabilidad as Schemas['Explicabilidad'],
  sensibilidades: sensibilidades as Schemas['Sensibilidad'][],
  estratos: estratos as Schemas['ErrorEstrato'][],
  preespecificacion: preespecificacion as Schemas['Preespecificacion'],
}

const BASE = '/api/v1'
const intParam = (v: string | null) => (v === null || v === '' ? undefined : Number(v))

export function problema(status: number, title: string, detail?: string) {
  const body: Schemas['Problema'] = { type: 'about:blank', title, status, detail }
  return HttpResponse.json(body, {
    status,
    headers: { 'Content-Type': 'application/problem+json' },
  })
}

export { mediana }

export const handlers = [
  http.get(`${BASE}/meta`, () => HttpResponse.json(DB.meta)),

  http.get(`${BASE}/provincias`, ({ request }) => {
    const q = new URL(request.url).searchParams
    const dep = q.get('departamento')
    const cla = q.get('clasificacion')
    return HttpResponse.json(
      DB.provincias.filter((p) => (!dep || p.departamento === dep) && (!cla || p.clasificacion === cla)),
    )
  }),

  http.get(`${BASE}/cobertura/embudo`, () => HttpResponse.json(DB.embudo)),

  http.get(`${BASE}/cobertura/exclusiones`, ({ request }) => {
    const q = new URL(request.url).searchParams
    const campana = intParam(q.get('campana'))
    const motivo = q.get('motivo')
    const prov = q.get('provincia_id')
    return HttpResponse.json(
      DB.exclusiones.filter(
        (e) =>
          (campana === undefined || e.campana === campana) &&
          (!motivo || e.motivo === motivo) &&
          (!prov || e.provincia_id === prov),
      ),
    )
  }),

  http.get(`${BASE}/predicciones`, ({ request }) => {
    const q = new URL(request.url).searchParams
    const campana = intParam(q.get('campana_prueba'))
    if (campana !== undefined && !DB.meta.campanas_de_prueba.includes(campana)) {
      return problema(422, 'Campaña no disponible', `Solo hay campañas de prueba ${DB.meta.campanas_de_prueba.join(', ')}.`)
    }
    const modelo = q.get('modelo')
    const prov = q.get('provincia_id')
    const semilla = intParam(q.get('semilla'))
    const limit = intParam(q.get('limit')) ?? 500
    const offset = intParam(q.get('offset')) ?? 0
    const filtradas = DB.predicciones.filter(
      (p) =>
        (campana === undefined || p.campana_prueba === campana) &&
        (!modelo || p.modelo === modelo) &&
        (!prov || p.provincia_id === prov) &&
        (semilla === undefined || p.semilla === semilla),
    )
    const body: Schemas['PaginaPredicciones'] = {
      total: filtradas.length,
      limit,
      offset,
      items: filtradas.slice(offset, offset + limit),
    }
    return HttpResponse.json(body)
  }),

  // Los mocks no ejecutan el modelo: con un tensor validan igual que el backend (provincia y campaña
  // obligatorias, campaña permitida) y responden 501, porque inventar una predicción sería falsear datos.
  // Para una observación de las campañas de prueba se devuelve la predicción REAL ya calculada por el
  // piloto (mediana de semillas del CNN-1D y B2); SHAP por celda no existe en reports/, por eso es null.
  http.post(`${BASE}/predicciones/inferir`, async ({ request }) => {
    const req = (await request.json()) as Schemas['SolicitudInferencia']
    if (req.tensor) {
      if (!req.provincia_id || req.campana === undefined) {
        return problema(422, 'Solicitud inválida', 'Con un tensor indique también provincia_id y campana: la línea base B2 depende de ellas.')
      }
      if (!DB.meta.campanas_permitidas.includes(req.campana)) {
        return problema(422, 'Campaña no permitida', `Campañas permitidas: ${DB.meta.campanas_permitidas.join(', ')}.`)
      }
      return problema(501, 'No implementado', 'Los datos de demostración no incluyen el modelo CNN-1D; use el backend para predecir sobre un tensor.')
    }
    if (!req.provincia_id || req.campana === undefined) {
      return problema(422, 'Solicitud inválida', 'Indique provincia_id y campana, o un tensor.')
    }
    if (!DB.meta.campanas_permitidas.includes(req.campana)) {
      return problema(422, 'Campaña no permitida', `Campañas permitidas: ${DB.meta.campanas_permitidas.join(', ')}.`)
    }
    const filas = DB.predicciones.filter((p) => p.provincia_id === req.provincia_id && p.campana_prueba === req.campana)
    const cnn = filas.filter((p) => p.modelo === 'cnn1d')
    const b2 = filas.find((p) => p.modelo === 'B2')
    if (!cnn.length || !b2) {
      return problema(404, 'Sin predicción', 'La observación no está entre las campañas de prueba del piloto.')
    }
    const y_pred = mediana(cnn.map((p) => p.y_pred))
    const body: Schemas['ResultadoInferencia'] = {
      y_pred,
      y_b2: b2.y_pred,
      anomalia: y_pred - b2.y_pred,
      y_real: b2.y_real,
      shap: null,
      modelo_version: `piloto-${DB.meta.modo}-${DB.meta.sha256_tensores.slice(0, 8)}`,
      vinculante: DB.meta.vinculante,
    }
    return HttpResponse.json(body)
  }),

  http.get(`${BASE}/indicadores/y1`, () => HttpResponse.json(DB.indicadoresY1)),
  http.get(`${BASE}/indicadores/x1-x2`, () => HttpResponse.json(DB.indicadoresX1X2)),
  http.get(`${BASE}/indicadores/variabilidad-semillas`, () => HttpResponse.json(DB.variabilidad)),
  http.get(`${BASE}/explicabilidad`, () => HttpResponse.json(DB.explicabilidad)),

  http.get(`${BASE}/sensibilidades`, ({ request }) => {
    const q = new URL(request.url).searchParams
    const variante = q.get('variante')
    const modelo = q.get('modelo')
    return HttpResponse.json(
      DB.sensibilidades.filter((s) => (!variante || s.variante === variante) && (!modelo || s.modelo === modelo)),
    )
  }),
  http.get(`${BASE}/sensibilidades/estratos`, () => HttpResponse.json(DB.estratos)),
  http.get(`${BASE}/preespecificacion`, () => HttpResponse.json(DB.preespecificacion)),
]
