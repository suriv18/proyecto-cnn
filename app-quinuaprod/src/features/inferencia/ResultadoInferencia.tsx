import { Chip } from '@/components/Chip'
import { Dato, Datos } from '@/components/Dato'
import type { Schemas } from '@/lib/api/client'
import { ApiError } from '@/lib/api/errors'
import { formatKgHa } from '@/lib/format'

const conSigno = (v: number) => `${v > 0 ? '+' : ''}${formatKgHa(v)}`

export function ErrorInferencia({ error }: { error: unknown }) {
  const e = error instanceof ApiError ? error : null
  return (
    <div role="alert" className="mt-4 rounded-lg border border-aviso/50 bg-aviso-soft p-4 text-sm text-aviso">
      <p className="font-semibold">{e ? e.message : 'No se pudo completar la inferencia.'}</p>
      {e?.detail && <p className="mt-1">{e.detail}</p>}
    </div>
  )
}

export function ResultadoInferencia({ r }: { r: Schemas['ResultadoInferencia'] }) {
  return (
    <section aria-label="Resultado de la inferencia" className="mt-6">
      <Datos etiqueta="Valores de la inferencia">
        <Dato etiqueta="CNN-1D (predicho)" destacado>
          {formatKgHa(r.y_pred)}
        </Dato>
        <Dato etiqueta="Línea base B2">{formatKgHa(r.y_b2)}</Dato>
        <Dato etiqueta="Anomalía (predicho − B2)">{conSigno(r.anomalia)}</Dato>
        {r.y_real != null && <Dato etiqueta="Observado">{formatKgHa(r.y_real)}</Dato>}
        {r.y_real != null && <Dato etiqueta="Error (predicho − observado)">{conSigno(r.y_pred - r.y_real)}</Dato>}
      </Datos>
      <p className="mt-3 flex flex-wrap items-center gap-2 text-sm text-muted">
        <Chip tono={r.vinculante ? 'ok' : 'aviso'}>{r.vinculante ? 'resultado vinculante' : 'resultado no vinculante'}</Chip>
        <span>Modelo {r.modelo_version}.</span>
        {!r.shap && <span>El SHAP por celda no está disponible para esta observación.</span>}
      </p>
    </section>
  )
}
