import { useMemo, useState, type ReactNode } from 'react'

export interface Columna<T> {
  clave: string
  etiqueta: string
  /** Unidad bajo la etiqueta: se escribe una vez en la cabecera y no en cada celda. */
  unidad?: string
  render: (fila: T) => ReactNode
  /** Valor con que se ordena; si falta, la columna no es ordenable. */
  valor?: (fila: T) => string | number | undefined
  derecha?: boolean
}

interface Props<T> {
  titulo: string
  columnas: Columna<T>[]
  filas: T[]
  idFila: (fila: T) => string
  /** Origen del dato, visible bajo la tabla (estilo de las tablas estadísticas). */
  fuente?: string
  alturaMaxima?: string
  /** Clases extra para resaltar filas (p. ej. las del modelo en estudio). */
  claseFila?: (fila: T) => string
}

export function Tabla<T>({ titulo, columnas, filas, idFila, fuente, alturaMaxima, claseFila }: Props<T>) {
  const [orden, setOrden] = useState<{ clave: string; asc: boolean } | null>(null)

  const ordenadas = useMemo(() => {
    const col = columnas.find((c) => c.clave === orden?.clave)
    if (!orden || !col?.valor) return filas
    const v = col.valor
    const dir = orden.asc ? 1 : -1
    return [...filas].sort((a, b) => {
      const x = v(a)
      const y = v(b)
      if (x === undefined) return 1
      if (y === undefined) return -1
      return (typeof x === 'number' && typeof y === 'number' ? x - y : String(x).localeCompare(String(y))) * dir
    })
  }, [filas, columnas, orden])

  return (
    <figure className="my-4">
      <div
        className={`relative overflow-auto rounded-lg border border-line bg-paper ${alturaMaxima ?? ''}`}
        tabIndex={0}
        role="region"
        aria-label={titulo}
      >
        <table className="w-full border-collapse text-sm">
          <caption className="sr-only">{titulo}</caption>
          <thead className="sticky top-0 z-10 bg-sunken">
            <tr className="border-b border-line-strong">
              {columnas.map((c) => {
                const activa = orden?.clave === c.clave
                return (
                  <th
                    key={c.clave}
                    scope="col"
                    aria-sort={activa ? (orden.asc ? 'ascending' : 'descending') : c.valor ? 'none' : undefined}
                    className={`whitespace-nowrap px-3 py-2.5 align-bottom text-xs font-semibold text-muted ${c.derecha ? 'text-right' : 'text-left'}`}
                  >
                    {c.valor ? (
                      <button
                        type="button"
                        onClick={() => setOrden({ clave: c.clave, asc: activa ? !orden.asc : true })}
                        className={`group inline-flex items-baseline gap-1 font-semibold hover:text-ink ${activa ? 'text-ink' : ''}`}
                      >
                        {c.etiqueta}
                        <span aria-hidden="true" className={`text-[0.625rem] ${activa ? 'text-accent' : 'opacity-0 group-hover:opacity-60'}`}>
                          {activa && !orden.asc ? '▼' : '▲'}
                        </span>
                      </button>
                    ) : (
                      c.etiqueta
                    )}
                    {c.unidad && <span className="block text-[0.6875rem] font-normal">{c.unidad}</span>}
                  </th>
                )
              })}
            </tr>
          </thead>
          <tbody>
            {ordenadas.map((fila) => (
              <tr key={idFila(fila)} className={`border-b border-line last:border-b-0 hover:bg-sunken/60 ${claseFila?.(fila) ?? ''}`}>
                {columnas.map((c) => (
                  <td key={c.clave} className={`px-3 py-2 ${c.derecha ? 'whitespace-nowrap text-right tabular-nums' : ''}`}>
                    {c.render(fila)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {fuente && (
        <figcaption className="mt-1.5 font-mono text-[0.6875rem] text-muted">Fuente: {fuente}</figcaption>
      )}
    </figure>
  )
}
