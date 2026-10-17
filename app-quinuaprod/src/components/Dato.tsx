import type { ReactNode } from 'react'

/** Una cifra con su etiqueta. Va dentro de un <dl>. */
export function Dato({ etiqueta, children, destacado }: { etiqueta: string; children: ReactNode; destacado?: boolean }) {
  return (
    <div className={`rounded-lg border p-4 ${destacado ? 'border-accent bg-accent-soft' : 'border-line bg-paper'}`}>
      <dt className="text-xs font-medium text-muted">{etiqueta}</dt>
      <dd className="mt-1 font-display text-xl font-semibold leading-tight tabular-nums">{children}</dd>
    </div>
  )
}

export function Datos({ etiqueta, children, columnas = 'lg:grid-cols-4' }: { etiqueta: string; children: ReactNode; columnas?: string }) {
  return (
    <dl aria-label={etiqueta} className={`grid grid-cols-2 gap-3 ${columnas}`}>
      {children}
    </dl>
  )
}
