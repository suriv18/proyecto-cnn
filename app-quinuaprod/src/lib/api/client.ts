import createClient from 'openapi-fetch'
import { API_URL } from '@/config/env'
import type { components, paths } from './schema'

export type Schemas = components['schemas']

export function createApiClient(baseUrl: string = API_URL) {
  // fetch se resuelve en cada llamada: openapi-fetch lo captura al crearse y asi no veria el parche de MSW.
  return createClient<paths>({ baseUrl, fetch: (request) => globalThis.fetch(request) })
}

export const api = createApiClient()
