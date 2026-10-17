import { useMemo, useState } from 'react'
import { Campo, Filtros } from '@/components/Campo'
import { EstadoConsulta } from '@/components/EstadoConsulta'
import { PageHeader } from '@/components/PageHeader'
import { Tabla, type Columna } from '@/components/Tabla'
import type { Schemas } from '@/lib/api/client'
import { useEstadoPiloto } from '@/lib/api/useEstadoPiloto'
import { formatEntero, formatEtiqueta } from '@/lib/format'
import { useExclusiones, useProvincias } from './api/queries'

type Motivo = Schemas['MotivoExclusion']

const CODIGO: Record<Motivo, string> = {
  fuera_del_ambito_altoandino: 'A',
  ciclo_fuera_de_ventana_de_extraccion: 'V',
  tensor_incompleto: 'T',
}

const TODOS = ''

export default function CoberturaPage() {
  const [departamento, setDepartamento] = useState(TODOS)
  const [clasificacion, setClasificacion] = useState(TODOS)
  const provincias = useProvincias()
  const exclusiones = useExclusiones()
  const piloto = useEstadoPiloto()

  const mapa = useMemo(() => {
    const m = new Map<string, Motivo>()
    for (const e of exclusiones.data ?? []) m.set(`${e.provincia_id}|${e.campana}`, e.motivo)
    return m
  }, [exclusiones.data])

  return (
    <>
      <PageHeader
        eyebrow="Resultados"
        titulo="Cobertura"
        descripcion="Las provincias y campañas del piloto, con las observaciones excluidas y el motivo de cada exclusión."
      />
      <EstadoConsulta query={provincias}>
        {(todas) => {
          const departamentos = [...new Set(todas.map((p) => p.departamento))].sort()
          const clasificaciones = [...new Set(todas.map((p) => p.clasificacion))].sort()
          const filas = todas.filter(
            (p) => (!departamento || p.departamento === departamento) && (!clasificacion || p.clasificacion === clasificacion),
          )
          const campanas = piloto.data?.campanas_permitidas ?? []

          const columnas: Columna<Schemas['Provincia']>[] = [
            { clave: 'provincia', etiqueta: 'Provincia', render: (p) => <span className="font-medium">{p.provincia_id}</span>, valor: (p) => p.provincia_id },
            { clave: 'departamento', etiqueta: 'Departamento', render: (p) => p.departamento, valor: (p) => p.departamento },
            {
              clave: 'clasificacion',
              etiqueta: 'Clasificación',
              render: (p) => formatEtiqueta(p.clasificacion),
              valor: (p) => p.clasificacion,
            },
            ...campanas.map<Columna<Schemas['Provincia']>>((c) => ({
              clave: `c${c}`,
              etiqueta: String(c),
              render: (p) => {
                const motivo = mapa.get(`${p.provincia_id}|${c}`)
                return motivo ? (
                  <span
                    title={formatEtiqueta(motivo)}
                    aria-label={`Excluida: ${formatEtiqueta(motivo)}`}
                    className="mx-auto flex size-6 items-center justify-center rounded bg-aviso-soft text-xs font-bold text-aviso"
                  >
                    {CODIGO[motivo]}
                  </span>
                ) : (
                  <span aria-label="Sin exclusión registrada" className="block text-center text-muted">
                    ·
                  </span>
                )
              },
            })),
          ]

          return (
            <>
              <Filtros>
                <Campo
                  etiqueta="Departamento"
                  valor={departamento}
                  onChange={setDepartamento}
                  opciones={[{ valor: TODOS, etiqueta: 'Todos' }, ...departamentos.map((d) => ({ valor: d, etiqueta: d }))]}
                />
                <Campo
                  etiqueta="Clasificación"
                  valor={clasificacion}
                  onChange={setClasificacion}
                  opciones={[
                    { valor: TODOS, etiqueta: 'Todas' },
                    ...clasificaciones.map((c) => ({ valor: c, etiqueta: formatEtiqueta(c) })),
                  ]}
                />
                <p role="status" className="ml-auto text-sm text-muted">
                  {formatEntero(filas.length)} de {formatEntero(todas.length)} provincias
                </p>
              </Filtros>

              <EstadoConsulta query={exclusiones}>
                {() => (
                  <>
                    <Tabla
                      titulo="Matriz provincia por campaña"
                      alturaMaxima="max-h-[32rem]"
                      fuente="configs/provincias.csv y data/processed/exclusiones.csv"
                      columnas={columnas}
                      filas={filas}
                      idFila={(p) => p.provincia_id}
                    />
                    <p className="text-xs text-muted">
                      <strong className="font-semibold">Leyenda.</strong> A: fuera del ámbito altoandino · V: ciclo fuera
                      de la ventana de extracción · T: tensor incompleto · ·: sin exclusión registrada. Esta marca no
                      implica que la observación pase todos los filtros posteriores de calidad.
                    </p>
                  </>
                )}
              </EstadoConsulta>
            </>
          )
        }}
      </EstadoConsulta>
    </>
  )
}
