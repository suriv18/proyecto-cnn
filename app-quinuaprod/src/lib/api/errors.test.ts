import { describe, expect, it } from 'vitest'
import { ApiError, unwrap } from './errors'

const respuesta = (status: number) => new Response(null, { status })

describe('unwrap', () => {
  it('devuelve data cuando no hay error', () => {
    expect(unwrap({ data: { a: 1 }, response: respuesta(200) })).toEqual({ a: 1 })
  })

  it('lanza ApiError con el problema del backend', () => {
    const problema = { type: 'about:blank', title: 'Campaña no permitida', status: 422, detail: 'solo 2016-2020' }
    try {
      unwrap({ error: problema, response: respuesta(422) })
      expect.unreachable()
    } catch (e) {
      expect(e).toBeInstanceOf(ApiError)
      expect((e as ApiError).status).toBe(422)
      expect((e as ApiError).detail).toBe('solo 2016-2020')
      expect((e as ApiError).message).toBe('Campaña no permitida')
    }
  })

  it('usa el status HTTP cuando el cuerpo de error no es un problema', () => {
    expect(() => unwrap({ error: 'boom', response: respuesta(500) })).toThrow(ApiError)
  })
})
