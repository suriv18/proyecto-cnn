import { Chip } from '@/components/Chip'
import { useEstadoPiloto } from '@/lib/api/useEstadoPiloto'
import { formatEtiqueta } from '@/lib/format'

/** Franja permanente: qué datos se muestran, su estado y que las campañas externas siguen protegidas. */
export function AvisoPiloto() {
  const { data } = useEstadoPiloto()
  if (!data) return null

  return (
    <div
      role="note"
      aria-label="Aviso sobre los datos"
      className="flex flex-wrap items-center gap-x-3 gap-y-1.5 border-b border-line bg-sunken px-4 py-2 text-xs text-muted sm:px-8"
    >
      <Chip>Piloto {data.modo}</Chip>
      <Chip tono={data.vinculante ? 'ok' : 'aviso'}>{data.vinculante ? 'vinculante' : 'no vinculante'}</Chip>
      <Chip tono={data.fecha_congelamiento_e5 ? 'ok' : 'aviso'}>
        E5 {formatEtiqueta(data.estado_e5.toLowerCase())} v{data.version_e5}
      </Chip>
      <span>
        Se muestran las campañas de entrenamiento y prueba; las campañas externas permanecen protegidas hasta la evaluación confirmatoria.
      </span>
    </div>
  )
}
