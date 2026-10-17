export interface Opcion {
  valor: string
  etiqueta: string
}

export const CLASE_CONTROL =
  'rounded-md border border-line-strong bg-paper px-3 py-2 text-sm text-ink shadow-xs hover:border-ink/40'

export function Campo({
  etiqueta,
  valor,
  onChange,
  opciones,
}: {
  etiqueta: string
  valor: string
  onChange: (v: string) => void
  opciones: Opcion[]
}) {
  return (
    <label className="flex flex-col gap-1 text-sm">
      <span className="text-xs font-medium text-muted">{etiqueta}</span>
      <select value={valor} onChange={(e) => onChange(e.target.value)} className={`${CLASE_CONTROL} min-w-40`}>
        {opciones.map((o) => (
          <option key={o.valor} value={o.valor}>
            {o.etiqueta}
          </option>
        ))}
      </select>
    </label>
  )
}

/** Barra de filtros: agrupa los controles sobre una tabla o gráfico. */
export function Filtros({ children }: { children: React.ReactNode }) {
  return <div className="mb-4 flex flex-wrap items-end gap-4 rounded-lg border border-line bg-paper p-4">{children}</div>
}
