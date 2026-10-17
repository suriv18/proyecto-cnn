import { useMemo, useState } from 'react'
import { CLASE_CONTROL, Campo, Filtros } from '@/components/Campo'
import { Chip } from '@/components/Chip'
import { Dato, Datos } from '@/components/Dato'
import { EstadoConsulta } from '@/components/EstadoConsulta'
import { PageHeader } from '@/components/PageHeader'
import { Tabla } from '@/components/Tabla'
import { formatEtiqueta } from '@/lib/format'
import { usePreespecificacion } from './api/queries'
import { aplanar, contarPorEstado, textoValor, type EstadoE5, type ParametroPlano } from '@/lib/aplanar'

const TODOS = ''

/** Tono del estado: lo fijado es sólido; lo que aún se discute llama la atención. */
const TONO_ESTADO: Record<EstadoE5, 'neutro' | 'ok' | 'aviso' | 'acento'> = {
  fijado_por_tesis: 'neutro',
  fijado: 'ok',
  propuesto: 'aviso',
  por_decidir_piloto: 'acento',
  por_decidir_asesor: 'acento',
}

export default function PreespecificacionPage() {
  const consulta = usePreespecificacion()
  const [estado, setEstado] = useState(TODOS)
  const [texto, setTexto] = useState('')

  const parametros = useMemo(() => (consulta.data ? aplanar(consulta.data.secciones) : []), [consulta.data])
  const cuenta = useMemo(() => contarPorEstado(parametros), [parametros])
  const visibles = useMemo(() => {
    const q = texto.trim().toLowerCase()
    return parametros.filter(
      (p) =>
        (!estado || p.estado === estado) &&
        (!q || `${p.seccion} ${p.ruta} ${textoValor(p.valor)} ${p.fuente}`.toLowerCase().includes(q)),
    )
  }, [parametros, estado, texto])

  return (
    <>
      <PageHeader
        eyebrow="Referencia"
        titulo="Registro E5"
        descripcion="Los parámetros que la tesis fijó antes de evaluar las campañas externas, con su estado y la fuente que los respalda."
      />
      <EstadoConsulta query={consulta}>
        {({ metadatos }) => (
          <>
            <Datos etiqueta="Metadatos de E5">
              <Dato etiqueta="Documento">
                {formatEtiqueta(textoValor(metadatos.estado_documento).toLowerCase())} v{textoValor(metadatos.version)}
              </Dato>
              <Dato etiqueta="Borrador">{textoValor(metadatos.fecha_borrador)}</Dato>
              <Dato etiqueta="Congelamiento">
                {metadatos.fecha_congelamiento ? textoValor(metadatos.fecha_congelamiento) : 'Sin congelar'}
              </Dato>
              {metadatos.commit_congelamiento ? (
                <Dato etiqueta="Commit">
                  <span title={textoValor(metadatos.commit_congelamiento)}>
                    {textoValor(metadatos.commit_congelamiento).slice(0, 7)}
                  </span>
                </Dato>
              ) : null}
              {metadatos.sha256_conjunto_de_datos ? (
                <Dato etiqueta="Huella de los datos">
                  <span title={textoValor(metadatos.sha256_conjunto_de_datos)}>
                    {textoValor(metadatos.sha256_conjunto_de_datos).slice(0, 8)}
                  </span>
                </Dato>
              ) : null}
              <Dato etiqueta="Parámetros">{parametros.length}</Dato>
            </Datos>

            <ul className="mt-5 flex flex-wrap gap-2" aria-label="Parámetros por estado">
              {Object.entries(cuenta).map(([e, n]) => (
                <li key={e}>
                  <Chip tono={TONO_ESTADO[e as EstadoE5] ?? 'neutro'}>
                    {formatEtiqueta(e)}: <strong className="ml-1 tabular-nums">{n}</strong>
                  </Chip>
                </li>
              ))}
            </ul>

            <div className="mt-6">
              <Filtros>
                <Campo
                  etiqueta="Estado"
                  valor={estado}
                  onChange={setEstado}
                  opciones={[{ valor: TODOS, etiqueta: 'Todos' }, ...Object.keys(cuenta).map((e) => ({ valor: e, etiqueta: formatEtiqueta(e) }))]}
                />
                <label className="flex flex-col gap-1 text-sm">
                  <span className="text-xs font-medium text-muted">Buscar</span>
                  <input type="search" value={texto} onChange={(e) => setTexto(e.target.value)} className={`${CLASE_CONTROL} min-w-64`} />
                </label>
                <p role="status" className="ml-auto text-sm text-muted">
                  {visibles.length} de {parametros.length} parámetros
                </p>
              </Filtros>
            </div>

            <Tabla<ParametroPlano>
              titulo="Parámetros preespecificados"
              alturaMaxima="max-h-[36rem]"
              fuente="configs/preespecificacion.yaml"
              columnas={[
                { clave: 'seccion', etiqueta: 'Sección', render: (p) => formatEtiqueta(p.seccion), valor: (p) => p.seccion },
                { clave: 'ruta', etiqueta: 'Parámetro', render: (p) => <code className="font-mono text-xs">{p.ruta}</code>, valor: (p) => p.ruta },
                { clave: 'valor', etiqueta: 'Valor', render: (p) => <span className="break-words">{textoValor(p.valor)}</span> },
                {
                  clave: 'estado',
                  etiqueta: 'Estado',
                  render: (p) => <Chip tono={TONO_ESTADO[p.estado] ?? 'neutro'}>{formatEtiqueta(p.estado)}</Chip>,
                  valor: (p) => p.estado,
                },
                { clave: 'fuente', etiqueta: 'Fuente', render: (p) => <span className="text-xs text-muted">{p.fuente}</span> },
              ]}
              filas={visibles}
              idFila={(p) => `${p.seccion}.${p.ruta}`}
            />
          </>
        )}
      </EstadoConsulta>
    </>
  )
}
