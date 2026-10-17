import { zodResolver } from '@hookform/resolvers/zod'
import { useState } from 'react'
import { useForm } from 'react-hook-form'
import { CLASE_BOTON } from '@/components/Boton'
import { CLASE_CONTROL, Campo } from '@/components/Campo'
import { useEstadoPiloto } from '@/lib/api/useEstadoPiloto'
import { colorFase, etiquetaFaseCorta } from '@/lib/fases'
import { formatEtiqueta } from '@/lib/format'
import { useInferir, useProvinciasDelAnalisis } from './api/queries'
import { ErrorInferencia, ResultadoInferencia } from './ResultadoInferencia'
import { tensorSchema, type TensorForm } from './tensorSchema'

const ENTRADA = `${CLASE_CONTROL} w-24 text-right tabular-nums aria-[invalid=true]:border-aviso`

export function FormTensor() {
  const piloto = useEstadoPiloto()
  const variables = piloto.data?.orden_variables ?? []
  const fases = piloto.data?.orden_fases ?? []
  const inferir = useInferir()
  // La línea base B2 depende de la provincia y de la campaña; el modelo predice la anomalía respecto de ella.
  const provincias = useProvinciasDelAnalisis().data ?? []
  const campanas = piloto.data?.campanas_permitidas ?? []
  const [provinciaElegida, setProvincia] = useState('')
  const [campanaElegida, setCampana] = useState('')
  const provincia = provinciaElegida || provincias[0]?.provincia_id || ''
  const campana = campanaElegida || String(campanas.at(-1) ?? '')
  // Sin valores iniciales a propósito: no se propone ningún dato que no sea del usuario.
  const { register, handleSubmit, formState: { errors } } = useForm<TensorForm>({ resolver: zodResolver(tensorSchema) })

  return (
    <form
      aria-label="Inferencia sobre un tensor editado"
      noValidate
      className="rounded-lg border border-line bg-paper p-5"
      onSubmit={handleSubmit((t) =>
        inferir.mutate({
          provincia_id: provincia,
          campana: Number(campana),
          tensor: { valores: t.valores, altitud_media_m: t.altitud_media_m, prop_quinua: t.prop_quinua },
        }),
      )}
    >
      <div className="mb-5 flex flex-wrap items-end gap-4">
        <Campo
          etiqueta="Provincia"
          valor={provincia}
          onChange={(v) => {
            setProvincia(v)
            inferir.reset()
          }}
          opciones={provincias.map((p) => ({ valor: p.provincia_id, etiqueta: p.provincia_id }))}
        />
        <Campo
          etiqueta="Campaña"
          valor={campana}
          onChange={(v) => {
            setCampana(v)
            inferir.reset()
          }}
          opciones={campanas.map((c) => ({ valor: String(c), etiqueta: String(c) }))}
        />
        <p className="max-w-md pb-2 text-xs text-muted">
          El modelo predice la anomalía respecto de la línea base B2 de esta provincia y campaña; el rendimiento es B2 más esa anomalía.
        </p>
      </div>
      <div className="relative overflow-auto" role="region" aria-label="Tensor de entrada" tabIndex={0}>
        <table className="text-sm">
          <caption className="sr-only">Valores por variable y fase</caption>
          <thead>
            <tr>
              <th scope="col" className="px-2 py-1 text-left text-xs font-medium text-muted">Variable</th>
              {fases.map((f) => (
                <th key={f} scope="col" className="px-1 pb-1 text-right text-xs font-medium">
                  <span className="mb-1 ml-auto block h-1.5 w-24 rounded-full" style={{ background: colorFase(f) }} aria-hidden="true" />
                  <span className="sr-only">{formatEtiqueta(f)}</span>
                  <span aria-hidden="true">{etiquetaFaseCorta(f)}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {variables.map((v, i) => (
              <tr key={v}>
                <th scope="row" className="px-2 py-1 text-left font-medium">{formatEtiqueta(v)}</th>
                {fases.map((f, j) => (
                  <td key={f} className="px-1 py-1">
                    <input
                      type="number"
                      step="any"
                      aria-label={`${formatEtiqueta(v)}, ${formatEtiqueta(f)}`}
                      aria-invalid={errors.valores?.[i]?.[j] ? true : undefined}
                      className={ENTRADA}
                      {...register(`valores.${i}.${j}` as const, { valueAsNumber: true })}
                    />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-5 flex flex-wrap gap-4">
        <label className="flex flex-col gap-1 text-sm">
          <span className="text-xs font-medium text-muted">Altitud media (m)</span>
          <input type="number" step="any" className={ENTRADA} aria-invalid={errors.altitud_media_m ? true : undefined} {...register('altitud_media_m', { valueAsNumber: true })} />
        </label>
        <label className="flex flex-col gap-1 text-sm">
          <span className="text-xs font-medium text-muted">Proporción de quinua</span>
          <input type="number" step="any" className={ENTRADA} aria-invalid={errors.prop_quinua ? true : undefined} {...register('prop_quinua', { valueAsNumber: true })} />
        </label>
      </div>

      {(errors.valores || errors.altitud_media_m || errors.prop_quinua) && (
        <p role="alert" className="mt-4 rounded-md border border-aviso/50 bg-aviso-soft p-3 text-sm text-aviso">
          Complete todos los valores con números válidos. Las covariables no pueden ser negativas.
        </p>
      )}

      <button type="submit" disabled={inferir.isPending || !provincia} className={`${CLASE_BOTON} mt-5`}>
        {inferir.isPending ? 'Calculando…' : 'Predecir con el tensor'}
      </button>
      {inferir.isError && <ErrorInferencia error={inferir.error} />}
      {inferir.data && <ResultadoInferencia r={inferir.data} />}
    </form>
  )
}
