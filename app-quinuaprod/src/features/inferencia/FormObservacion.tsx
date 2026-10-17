import { useState } from 'react'
import { CLASE_BOTON } from '@/components/Boton'
import { Campo } from '@/components/Campo'
import { useEstadoPiloto } from '@/lib/api/useEstadoPiloto'
import { useInferir, useObservacionesDisponibles } from './api/queries'
import { ErrorInferencia, ResultadoInferencia } from './ResultadoInferencia'

export function FormObservacion() {
  const piloto = useEstadoPiloto()
  const campanas = piloto.data?.campanas_de_prueba ?? []
  const [campanaElegida, setCampana] = useState('')
  const [provinciaElegida, setProvincia] = useState('')
  const campana = campanaElegida || String(campanas[0] ?? '')

  // Las provincias disponibles son las que tienen predicción del piloto en esa campaña.
  const provincias = useObservacionesDisponibles(campana ? Number(campana) : undefined).data ?? []
  const provincia = provinciaElegida || provincias[0] || ''
  const inferir = useInferir()

  return (
    <form
      aria-label="Inferencia sobre una observación del piloto"
      className="rounded-lg border border-line bg-paper p-5"
      onSubmit={(e) => {
        e.preventDefault()
        inferir.mutate({ provincia_id: provincia, campana: Number(campana) })
      }}
    >
      <div className="flex flex-wrap items-end gap-4">
        <Campo
          etiqueta="Campaña"
          valor={campana}
          onChange={(v) => {
            setCampana(v)
            setProvincia('')
            inferir.reset()
          }}
          opciones={campanas.map((c) => ({ valor: String(c), etiqueta: String(c) }))}
        />
        <Campo
          etiqueta="Provincia"
          valor={provincia}
          onChange={(v) => {
            setProvincia(v)
            inferir.reset()
          }}
          opciones={provincias.map((p) => ({ valor: p, etiqueta: p }))}
        />
        <button type="submit" disabled={!provincia || inferir.isPending} className={CLASE_BOTON}>
          {inferir.isPending ? 'Calculando…' : 'Predecir'}
        </button>
      </div>
      {inferir.isError && <ErrorInferencia error={inferir.error} />}
      {inferir.data && <ResultadoInferencia r={inferir.data} />}
    </form>
  )
}
