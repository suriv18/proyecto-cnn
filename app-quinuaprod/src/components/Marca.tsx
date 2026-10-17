import { FASES, colorFase } from '@/lib/fases'

/** Marca: cinco granos que crecen, uno por fase del ciclo del cultivo. */
export function Marca({ tamano = 28 }: { tamano?: number }) {
  return (
    <svg width={tamano} height={tamano} viewBox="0 0 28 28" aria-hidden="true" focusable="false">
      {FASES.map((f, i) => {
        const alto = 8 + i * 4
        return <rect key={f} x={1 + i * 5.4} y={26 - alto} width={4} height={alto} rx={2} fill={colorFase(f)} />
      })}
    </svg>
  )
}

export function Logotipo({ compacto = false }: { compacto?: boolean }) {
  return (
    <div className="flex items-center gap-2.5">
      <Marca />
      <div className="leading-none">
        <p className="font-display text-lg font-bold tracking-tight">QuinuaProd</p>
        <p className={`mt-1 text-[0.6875rem] text-muted ${compacto ? "hidden sm:block" : ""}`}>Rendimiento de quinua · sierra</p>
      </div>
    </div>
  )
}
