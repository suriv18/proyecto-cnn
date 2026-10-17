import type { ReactNode } from 'react'

export function Seccion({ titulo, descripcion, children }: { titulo: string; descripcion?: ReactNode; children?: ReactNode }) {
  return (
    <section className="mt-10 first:mt-0">
      <h2 className="text-xl font-semibold">{titulo}</h2>
      {descripcion && <p className="mt-1 max-w-3xl text-sm text-muted">{descripcion}</p>}
      <div className="mt-4">{children}</div>
    </section>
  )
}
