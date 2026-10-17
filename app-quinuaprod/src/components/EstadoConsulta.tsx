import type { ReactNode } from 'react'
import type { UseQueryResult } from '@tanstack/react-query'
import { ApiError } from '@/lib/api/errors'

type Consulta<T> = Pick<UseQueryResult<T>, 'data' | 'isPending' | 'isError' | 'error'>

/** Resuelve los tres estados de una consulta y entrega los datos ya cargados. */
export function EstadoConsulta<T>({ query, children }: { query: Consulta<T>; children: (datos: T) => ReactNode }) {
  if (query.isPending) {
    return (
      <div role="status" className="flex items-center gap-3 rounded-lg border border-line bg-paper p-4 text-sm text-muted">
        <span className="size-2 animate-pulse rounded-full bg-accent" aria-hidden="true" />
        Cargando…
      </div>
    )
  }
  if (query.isError || query.data === undefined) {
    const detalle = query.error instanceof ApiError ? (query.error.detail ?? query.error.message) : undefined
    return (
      <p role="alert" className="rounded-lg border border-aviso/50 bg-aviso-soft p-4 text-sm text-aviso">
        <strong className="font-semibold">No se pudieron obtener los datos.</strong>
        {detalle ? ` ${detalle}` : ' Revise la conexión con el servidor e intente de nuevo.'}
      </p>
    )
  }
  return <>{children(query.data)}</>
}
