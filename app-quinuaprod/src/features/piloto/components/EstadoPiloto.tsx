import { Chip } from '@/components/Chip'
import { Dato, Datos } from '@/components/Dato'
import { EstadoConsulta } from '@/components/EstadoConsulta'
import { useEstadoPiloto } from '@/lib/api/useEstadoPiloto'
import { formatEntero, formatEtiqueta } from '@/lib/format'

export function EstadoPiloto() {
  const query = useEstadoPiloto()

  return (
    <EstadoConsulta query={query}>
      {(data) => (
        <Datos etiqueta="Estado del piloto" columnas="lg:grid-cols-3">
          <Dato etiqueta="Modo">{formatEtiqueta(data.modo)}</Dato>
          <Dato etiqueta="Vinculante">
            <Chip tono={data.vinculante ? 'ok' : 'aviso'}>{data.vinculante ? 'Sí' : 'No'}</Chip>
          </Dato>
          <Dato etiqueta="Observaciones del piloto">{formatEntero(data.n_observaciones)}</Dato>
          <Dato etiqueta="Campañas permitidas">{data.campanas_permitidas.join(', ')}</Dato>
          <Dato etiqueta="Campañas de prueba" destacado>
            {data.campanas_de_prueba.join(', ')}
          </Dato>
          <Dato etiqueta="E5">
            {formatEtiqueta(data.estado_e5.toLowerCase())} v{data.version_e5}
          </Dato>
        </Datos>
      )}
    </EstadoConsulta>
  )
}
