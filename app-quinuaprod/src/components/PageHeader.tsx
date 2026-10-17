import type { ReactNode } from 'react'

export function PageHeader({
  titulo,
  descripcion,
  eyebrow,
  children,
}: {
  titulo: string
  descripcion?: string
  eyebrow?: string
  children?: ReactNode
}) {
  return (
    <header className="mb-8 border-b border-line pb-6">
      {eyebrow && <p className="mb-1 text-xs font-semibold uppercase tracking-[0.12em] text-accent">{eyebrow}</p>}
      <h1 className="text-3xl font-bold sm:text-4xl">{titulo}</h1>
      {descripcion && <p className="mt-2 max-w-3xl text-muted">{descripcion}</p>}
      {children}
    </header>
  )
}
