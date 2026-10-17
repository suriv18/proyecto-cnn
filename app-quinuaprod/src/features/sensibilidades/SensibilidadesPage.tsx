import { useState } from 'react'
import { Campo, Filtros } from '@/components/Campo'
import { EstadoConsulta } from '@/components/EstadoConsulta'
import { PageHeader } from '@/components/PageHeader'
import { Seccion } from '@/components/Seccion'
import { Tabla } from '@/components/Tabla'
import type { Schemas } from '@/lib/api/client'
import { formatDecimal, formatEntero, formatEtiqueta } from '@/lib/format'
import { etiquetaModelo, ORDEN_MODELOS, type Modelo } from '@/lib/modelos'
import { useSensibilidades, useSensibilidadesEstratos } from './api/queries'

type Variante = Schemas['VarianteSensibilidad']

const VARIANTES: Variante[] = [
  'base',
  'sin_transicion',
  'tensor_completo',
  'horizonte_temprano',
  'sin_arrastre_ndvi',
  'con_exclusion_por_altitud',
]
const TODAS = ''

const signo = (v: number) => (v > 0 ? '+' : '')
/** Un RMSE más alto que la base es peor; más bajo, mejor. */
const tonoCambio = (v: number) => (v > 0 ? 'text-aviso' : v < 0 ? 'text-ok' : 'text-muted')

export default function SensibilidadesPage() {
  const [variante, setVariante] = useState<string>(TODAS)
  const [modelo, setModelo] = useState<string>('cnn1d')
  const sens = useSensibilidades({
    ...(variante ? { variante: variante as Variante } : {}),
    ...(modelo ? { modelo: modelo as Modelo } : {}),
  })
  const estratos = useSensibilidadesEstratos()

  return (
    <>
      <PageHeader
        eyebrow="Interpretación"
        titulo="Sensibilidades"
        descripcion="Cómo cambia el error al variar las decisiones del piloto, y cómo se reparte entre provincias con poca y mucha quinua."
      />
      <Filtros>
        <Campo
          etiqueta="Variante"
          valor={variante}
          onChange={setVariante}
          opciones={[{ valor: TODAS, etiqueta: 'Todas' }, ...VARIANTES.map((v) => ({ valor: v, etiqueta: formatEtiqueta(v) }))]}
        />
        <Campo
          etiqueta="Modelo"
          valor={modelo}
          onChange={setModelo}
          opciones={[{ valor: TODAS, etiqueta: 'Todos' }, ...ORDEN_MODELOS.map((m) => ({ valor: m, etiqueta: etiquetaModelo(m) }))]}
        />
      </Filtros>
      <p className="text-sm text-muted">
        <strong className="font-semibold text-ink">q</strong> = RMSE(modelo) / RMSE(referencia) − 1: un valor negativo
        significa menor error que la referencia. «Cambio vs base» es la variación porcentual del RMSE respecto de la
        variante base: en <span className="text-aviso">naranja</span> el error sube, en <span className="text-ok">verde</span> baja.
      </p>

      <EstadoConsulta query={sens}>
        {(filas) => (
          <Tabla
            titulo="RMSE por variante, modelo y campaña"
            alturaMaxima="max-h-[32rem]"
            fuente="reports/sensibilidades/sensibilidades_rmse.csv"
            columnas={[
              { clave: 'variante', etiqueta: 'Variante', render: (f) => formatEtiqueta(f.variante), valor: (f) => VARIANTES.indexOf(f.variante) },
              { clave: 'modelo', etiqueta: 'Modelo', render: (f) => etiquetaModelo(f.modelo), valor: (f) => ORDEN_MODELOS.indexOf(f.modelo) },
              { clave: 'campana', etiqueta: 'Campaña', render: (f) => f.campana_prueba, valor: (f) => f.campana_prueba },
              { clave: 'rmse', etiqueta: 'RMSE', unidad: 'kg/ha', derecha: true, render: (f) => formatDecimal(f.rmse, 1), valor: (f) => f.rmse },
              {
                clave: 'cambio',
                etiqueta: 'Cambio vs base',
                unidad: '%',
                derecha: true,
                render: (f) => (
                  <span className={`font-medium ${tonoCambio(f.cambio_vs_base_pct)}`}>
                    {signo(f.cambio_vs_base_pct)}
                    {formatDecimal(f.cambio_vs_base_pct, 1)}
                  </span>
                ),
                valor: (f) => f.cambio_vs_base_pct,
              },
              { clave: 'qb2', etiqueta: 'q vs B2', derecha: true, render: (f) => formatDecimal(f.q_frente_a_b2, 3), valor: (f) => f.q_frente_a_b2 },
              { clave: 'qb4', etiqueta: 'q vs B4', derecha: true, render: (f) => formatDecimal(f.q_frente_a_b4, 3), valor: (f) => f.q_frente_a_b4 },
            ]}
            filas={filas}
            idFila={(f) => `${f.variante}-${f.modelo}-${f.campana_prueba}`}
          />
        )}
      </EstadoConsulta>

      <Seccion
        titulo="Error por estrato de quinua"
        descripcion="RMSE según el tercil de proporción de quinua de la provincia (bajo, medio, alto)."
      >
        <EstadoConsulta query={estratos}>
          {(filas) => (
            <Tabla
              titulo="Error por estrato de proporción de quinua"
              fuente="reports/sensibilidades/error_por_estrato_de_quinua.csv"
              columnas={[
                { clave: 'modelo', etiqueta: 'Modelo', render: (f) => etiquetaModelo(f.modelo), valor: (f) => ORDEN_MODELOS.indexOf(f.modelo) },
                { clave: 'estrato', etiqueta: 'Estrato', render: (f) => formatEtiqueta(f.estrato) },
                { clave: 'rmse', etiqueta: 'RMSE', unidad: 'kg/ha', derecha: true, render: (f) => formatDecimal(f.rmse, 1), valor: (f) => f.rmse },
                { clave: 'n', etiqueta: 'Observaciones', derecha: true, render: (f) => formatEntero(f.n), valor: (f) => f.n },
              ]}
              filas={filas}
              idFila={(f) => `${f.modelo}-${f.estrato}`}
            />
          )}
        </EstadoConsulta>
      </Seccion>
    </>
  )
}
