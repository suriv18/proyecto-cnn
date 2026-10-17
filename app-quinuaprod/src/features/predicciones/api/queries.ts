import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api/client'
import type { paths } from '@/lib/api/schema'
import { unwrap } from '@/lib/api/errors'

type ConsultaPredicciones = NonNullable<paths['/predicciones']['get']['parameters']['query']>

export const usePredicciones = (query: ConsultaPredicciones = {}) =>
  useQuery({
    queryKey: ['predicciones', query],
    queryFn: async () => unwrap(await api.GET('/predicciones', { params: { query } })),
    placeholderData: keepPreviousData,
  })

