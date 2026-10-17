import { CLASE_CONTROL } from '@/components/Campo'
import { useTema } from '@/lib/useTema'
import type { Tema } from '@/lib/theme'

const OPCIONES: { valor: Tema; etiqueta: string }[] = [
  { valor: 'sistema', etiqueta: 'Sistema' },
  { valor: 'claro', etiqueta: 'Claro' },
  { valor: 'oscuro', etiqueta: 'Oscuro' },
]

export function ThemeToggle() {
  const { tema, setTema } = useTema()

  return (
    <label className="flex items-center gap-2 text-sm">
      <span className="sr-only text-muted sm:not-sr-only">Tema</span>
      <select value={tema} onChange={(e) => setTema(e.target.value as Tema)} className={`${CLASE_CONTROL} py-1`}>
        {OPCIONES.map((o) => (
          <option key={o.valor} value={o.valor}>
            {o.etiqueta}
          </option>
        ))}
      </select>
    </label>
  )
}
