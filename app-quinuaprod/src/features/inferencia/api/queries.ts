import { useMutation, useQuery } from '@tanstack/react-query'
import { api, type Schemas } from '@/lib/api/client'
import { unwrap } from '@/lib/api/errors'

export const useInferir = () =>
  useMutation({
    mutationFn: async (body: Schemas['SolicitudInferencia']) => unwrap(await api.POST('/predicciones/inferir', { body })),
  })

/** Provincias con predicción del piloto en una campaña de prueba: las que se pueden consultar. */
export const useObservacionesDisponibles = (campana: number | undefined) =>
  useQuery({
    queryKey: ['inferencia', 'disponibles', campana],
    enabled: campana !== undefined,
    queryFn: async () => {
      const pagina = unwrap(
        await api.GET('/predicciones', { params: { query: { campana_prueba: campana, modelo: 'B2', semilla: 0, limit: 5000 } } }),
      )
      return pagina.items.map((p) => p.provincia_id).sort()
    },
  })

/** Provincias del ámbito del análisis: las de otros regímenes (selva, costa con riego) no tienen historia de entrenamiento. */
export const useProvinciasDelAnalisis = () =>
  useQuery({
    queryKey: ['provincias', 'analisis'],
    queryFn: async () => unwrap(await api.GET('/provincias')),
    select: (todas) => todas.filter((p) => p.clasificacion === 'altoandina' || p.clasificacion === 'transicion'),
  })
