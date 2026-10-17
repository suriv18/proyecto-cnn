import type { Nivel } from './contenido'

const COLUMNAS = [
  { clave: 'problema', titulo: 'Problema' },
  { clave: 'objetivo', titulo: 'Objetivo' },
  { clave: 'hipotesis', titulo: 'Hipótesis' },
] as const

/**
 * Matriz de consistencia: cada fila es un nivel (general, específico 1 y 2) y las tres columnas son la
 * cadena problema → objetivo → hipótesis. En pantallas estrechas las columnas se apilan.
 */
export function MatrizConsistencia({ niveles }: { niveles: Nivel[] }) {
  return (
    <div className="space-y-4">
      {niveles.map((nivel) => (
        <article key={nivel.id} aria-label={`Nivel ${nivel.etiqueta.toLowerCase()}`} className="rounded-lg border border-line bg-paper">
          <h3 className="border-b border-line bg-sunken px-4 py-2 text-sm font-semibold">{nivel.etiqueta}</h3>
          <div className="grid gap-px bg-line lg:grid-cols-3">
            {COLUMNAS.map((c) => (
              <section key={c.clave} className="bg-paper p-4">
                <h4 className="mb-1.5 text-xs font-semibold uppercase tracking-[0.12em] text-accent">{c.titulo}</h4>
                <p className="text-sm leading-relaxed">{nivel[c.clave]}</p>
              </section>
            ))}
          </div>
        </article>
      ))}
    </div>
  )
}
