import { useMemo, useState } from 'react'
import { Campo, Filtros } from '@/components/Campo'
import { EstadoConsulta } from '@/components/EstadoConsulta'
import { GraficoDispersion } from '@/components/Graficos'
import { PageHeader } from '@/components/PageHeader'
import { Tabla, type Columna } from '@/components/Tabla'
import { useEstadoPiloto } from '@/lib/api/useEstadoPiloto'
import { formatEntero } from '@/lib/format'
import { esReferencia, etiquetaModelo, ORDEN_MODELOS, type Modelo } from '@/lib/modelos'
import { agregarPorProvincia, errorAbsoluto, type FilaProvincia } from '@/lib/predicciones'
import { usePredicciones } from './api/queries'

const clasePorModelo = (m: Modelo) => (m === 'cnn1d' ? 'font-semibold text-accent' : esReferencia(m) ? 'text-muted' : '')

export default function PrediccionesPage() {
  const piloto = useEstadoPiloto()
  const campanas = piloto.data?.campanas_de_prueba ?? []
  const [elegida, setCampana] = useState<string>('')
  const [modelo, setModelo] = useState<Modelo>('cnn1d')
  const campana = elegida || String(campanas[0] ?? '')

  const consulta = usePredicciones(campana ? { campana_prueba: Number(campana), limit: 5000 } : {})
  const filas = useMemo(() => agregarPorProvincia(consulta.data?.items ?? []), [consulta.data])
  const presentes = ORDEN_MODELOS.filter((m) => filas.some((f) => f.modelos[m] !== undefined))

  const columnas: Columna<FilaProvincia>[] = [
    { clave: 'provincia', etiqueta: 'Provincia', render: (f) => <span className="font-medium">{f.provincia_id}</span>, valor: (f) => f.provincia_id },
    { clave: 'real', etiqueta: 'Observado', unidad: 'kg/ha', derecha: true, render: (f) => formatEntero(f.y_real), valor: (f) => f.y_real },
    ...presentes.map<Columna<FilaProvincia>>((m) => ({
      clave: m,
      etiqueta: etiquetaModelo(m),
      unidad: esReferencia(m) ? 'línea base' : 'kg/ha',
      derecha: true,
      render: (f) => (f.modelos[m] === undefined ? '—' : <span className={clasePorModelo(m)}>{formatEntero(f.modelos[m])}</span>),
      valor: (f) => f.modelos[m],
    })),
    {
      clave: 'error',
      etiqueta: `Error abs. ${etiquetaModelo(modelo)}`,
      unidad: 'kg/ha',
      derecha: true,
      render: (f) => {
        const e = errorAbsoluto(f, modelo)
        return e === undefined ? '—' : formatEntero(e)
      },
      valor: (f) => errorAbsoluto(f, modelo),
    },
  ]

  return (
    <>
      <PageHeader
        eyebrow="Resultados"
        titulo="Predicción vs observado"
        descripcion="Rendimiento predicho por cada modelo en las campañas de prueba del piloto, frente al rendimiento observado."
      />
      <Filtros>
        <Campo
          etiqueta="Campaña de prueba"
          valor={campana}
          onChange={setCampana}
          opciones={campanas.map((c) => ({ valor: String(c), etiqueta: String(c) }))}
        />
        <Campo
          etiqueta="Modelo del gráfico"
          valor={modelo}
          onChange={(v) => setModelo(v as Modelo)}
          opciones={ORDEN_MODELOS.map((m) => ({ valor: m, etiqueta: etiquetaModelo(m) }))}
        />
      </Filtros>

      <EstadoConsulta query={consulta}>
        {() => {
          const puntos = filas.flatMap((f) =>
            f.modelos[modelo] === undefined ? [] : [{ x: f.y_real, y: f.modelos[modelo], etiqueta: f.provincia_id }],
          )
          return (
            <>
              <p className="text-sm text-muted">
                Los modelos con varias semillas muestran la mediana.{' '}
                {esReferencia(modelo) ? 'B1-B4 son líneas base, no modelos entrenados.' : ''}
              </p>
              <GraficoDispersion
                titulo={`${etiquetaModelo(modelo)}: observado vs predicho (${campana})`}
                descripcion={`Dispersión de ${puntos.length} provincias: rendimiento observado contra predicho por ${etiquetaModelo(modelo)} en la campaña ${campana}. La línea punteada marca el ajuste perfecto.`}
                puntos={puntos}
                unidad="kg/ha"
              />
              <Tabla
                titulo={`Predicción por provincia, campaña ${campana}`}
                alturaMaxima="max-h-[32rem]"
                fuente="reports/piloto_oficial/predicciones.csv"
                columnas={columnas}
                filas={filas}
                idFila={(f) => f.provincia_id}
              />
            </>
          )
        }}
      </EstadoConsulta>
    </>
  )
}
