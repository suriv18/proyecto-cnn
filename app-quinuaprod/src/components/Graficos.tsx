import type { ReactElement } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { escalaAgradable } from '@/lib/escala'
import { colorFase, etiquetaFaseCorta } from '@/lib/fases'
import { formatEntero } from '@/lib/format'

const EJE = { fontSize: 12, fill: 'var(--muted)' } as const
const ETIQUETA_EJE = { fontSize: 12, fill: 'var(--muted)' } as const
const REJILLA = { stroke: 'var(--line)', strokeDasharray: '3 3' } as const
const LINEA_EJE = { stroke: 'var(--line-strong)' } as const
const TOOLTIP = {
  contentStyle: {
    background: 'var(--paper)',
    border: '1px solid var(--line-strong)',
    borderRadius: 8,
    fontSize: 12,
    color: 'var(--ink)',
  },
  labelStyle: { color: 'var(--ink)', fontWeight: 600 },
  itemStyle: { color: 'var(--ink)' },
} as const

function Figura({
  titulo,
  descripcion,
  alto = 'h-72',
  children,
}: {
  titulo: string
  descripcion: string
  alto?: string
  children: ReactElement
}) {
  return (
    <figure className="my-4 rounded-lg border border-line bg-paper p-4">
      <figcaption className="mb-2 text-sm font-semibold">{titulo}</figcaption>
      <div role="img" aria-label={descripcion} className={alto}>
        <ResponsiveContainer width="100%" height="100%" initialDimension={{ width: 480, height: 256 }}>
          {children}
        </ResponsiveContainer>
      </div>
    </figure>
  )
}

export interface Barra {
  etiqueta: string
  valor: number
}

/** Barras horizontales. Cada gráfico va acompañado de una tabla con los mismos valores. */
export function GraficoBarras({
  titulo,
  descripcion,
  datos,
  formato,
}: {
  titulo: string
  descripcion: string
  datos: Barra[]
  formato: (v: number) => string
}) {
  const { max, marcas } = escalaAgradable(Math.max(...datos.map((d) => d.valor)))
  return (
    <Figura titulo={titulo} descripcion={descripcion} alto={datos.length > 5 ? 'h-80' : 'h-64'}>
      <BarChart data={datos} layout="vertical" margin={{ left: 8, right: 24, top: 4 }} barCategoryGap="22%">
        <CartesianGrid {...REJILLA} horizontal={false} />
        <XAxis type="number" tick={EJE} tickFormatter={formato} domain={[0, max]} ticks={marcas} axisLine={LINEA_EJE} tickLine={false} />
        <YAxis type="category" dataKey="etiqueta" tick={EJE} width={120} axisLine={LINEA_EJE} tickLine={false} />
        <Tooltip {...TOOLTIP} cursor={{ fill: 'var(--sunken)' }} formatter={(v) => formato(Number(v))} />
        <Bar dataKey="valor" fill="var(--accent)" radius={[0, 4, 4, 0]} isAnimationActive={false} />
      </BarChart>
    </Figura>
  )
}

export interface BarraFase {
  fase: string
  valor: number
}

/** Una columna por fase, en el orden del ciclo del cultivo y con el color de cada fase. */
export function GraficoFases({
  titulo,
  descripcion,
  datos,
  formato,
  etiquetaEje,
}: {
  titulo: string
  descripcion: string
  datos: BarraFase[]
  formato: (v: number) => string
  etiquetaEje: string
}) {
  const { max, marcas } = escalaAgradable(Math.max(...datos.map((d) => d.valor)))
  const filas = datos.map((d) => ({ ...d, etiqueta: etiquetaFaseCorta(d.fase) }))
  return (
    <Figura titulo={titulo} descripcion={descripcion}>
      <BarChart data={filas} margin={{ left: 4, right: 16, top: 8, bottom: 4 }} barCategoryGap="18%">
        <CartesianGrid {...REJILLA} vertical={false} />
        <XAxis dataKey="etiqueta" tick={EJE} axisLine={LINEA_EJE} tickLine={false} />
        <YAxis
          tick={EJE}
          tickFormatter={formato}
          domain={[0, max]}
          ticks={marcas}
          width={56}
          axisLine={false}
          tickLine={false}
          label={{ value: etiquetaEje, angle: -90, position: 'insideLeft', offset: 8, style: ETIQUETA_EJE }}
        />
        <Tooltip {...TOOLTIP} cursor={{ fill: 'var(--sunken)' }} formatter={(v) => formato(Number(v))} />
        <Bar dataKey="valor" radius={[4, 4, 0, 0]} isAnimationActive={false}>
          {filas.map((f) => (
            <Cell key={f.fase} fill={colorFase(f.fase)} />
          ))}
        </Bar>
      </BarChart>
    </Figura>
  )
}

export interface Punto {
  x: number
  y: number
  etiqueta: string
}

/** Dispersión observado (x) vs predicho (y) con la diagonal de ajuste perfecto. */
export function GraficoDispersion({
  titulo,
  descripcion,
  puntos,
  unidad,
}: {
  titulo: string
  descripcion: string
  puntos: Punto[]
  unidad: string
}) {
  const { max, marcas } = escalaAgradable(Math.max(1, ...puntos.flatMap((p) => [p.x, p.y])))
  const diagonal = [
    { x: 0, y: 0 },
    { x: max, y: max },
  ]
  return (
    <Figura titulo={titulo} descripcion={descripcion} alto="h-96">
      <ScatterChart margin={{ left: 8, right: 20, top: 8, bottom: 20 }}>
        <CartesianGrid {...REJILLA} />
        <XAxis
          type="number"
          dataKey="x"
          name="Observado"
          tick={EJE}
          domain={[0, max]}
          ticks={marcas}
          tickFormatter={formatEntero}
          axisLine={LINEA_EJE}
          tickLine={false}
          label={{ value: `Observado (${unidad})`, position: 'insideBottom', offset: -12, style: ETIQUETA_EJE }}
        />
        <YAxis
          type="number"
          dataKey="y"
          name="Predicho"
          tick={EJE}
          domain={[0, max]}
          ticks={marcas}
          tickFormatter={formatEntero}
          width={64}
          axisLine={LINEA_EJE}
          tickLine={false}
          label={{ value: `Predicho (${unidad})`, angle: -90, position: 'insideLeft', offset: 4, style: ETIQUETA_EJE }}
        />
        <Tooltip
          {...TOOLTIP}
          cursor={{ strokeDasharray: '3 3', stroke: 'var(--line-strong)' }}
          formatter={(v) => `${formatEntero(Number(v))} ${unidad}`}
        />
        <Scatter data={puntos} fill="var(--accent)" fillOpacity={0.75} isAnimationActive={false} />
        <Line data={diagonal} dataKey="y" dot={false} stroke="var(--reference)" strokeDasharray="5 4" isAnimationActive={false} legendType="none" />
      </ScatterChart>
    </Figura>
  )
}
