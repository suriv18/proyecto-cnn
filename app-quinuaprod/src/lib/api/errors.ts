import type { Schemas } from './client'

/** Error de la API (application/problem+json, RFC 9457) o de red. */
export class ApiError extends Error {
  readonly status: number
  readonly detail?: string

  constructor(problema: Partial<Schemas['Problema']> & { status: number }) {
    super(problema.title ?? `Error ${problema.status}`)
    this.name = 'ApiError'
    this.status = problema.status
    this.detail = problema.detail
  }
}

interface Resultado<T> {
  data?: T
  error?: unknown
  response: Response
}

/** Devuelve `data` o lanza ApiError; así TanStack Query maneja siempre excepciones. */
export function unwrap<T>({ data, error, response }: Resultado<T>): T {
  if (error !== undefined || data === undefined) {
    const problema = (error ?? {}) as Partial<Schemas['Problema']>
    throw new ApiError({ ...problema, status: problema.status ?? response.status })
  }
  return data
}
