import type { ReactNode } from 'react'

const TONOS = {
  neutro: 'border-line-strong bg-sunken text-ink',
  ok: 'border-ok/40 bg-ok-soft text-ok',
  aviso: 'border-aviso/40 bg-aviso-soft text-aviso',
  acento: 'border-accent/40 bg-accent-soft text-accent',
}

export function Chip({ tono = 'neutro', children }: { tono?: keyof typeof TONOS; children: ReactNode }) {
  return (
    <span className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium ${TONOS[tono]}`}>
      {children}
    </span>
  )
}
