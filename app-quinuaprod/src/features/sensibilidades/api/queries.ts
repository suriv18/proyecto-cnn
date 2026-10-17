import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api/client'
import type { paths } from '@/lib/api/schema'
import { unwrap } from '@/lib/api/errors'

type ConsultaSensibilidades = NonNullable<paths['/sensibilidades']['get']['parameters']['query']>

export const useSensibilidades = (query: ConsultaSensibilidades = {}) =>
  useQuery({
    queryKey: ['sensibilidades', query],
    queryFn: async () => unwrap(await api.GET('/sensibilidades', { params: { query } })),
    placeholderData: keepPreviousData,
  })

export const useSensibilidadesEstratos = () =>
  useQuery({
    queryKey: ['sensibilidades', 'estratos'],
    queryFn: async () => unwrap(await api.GET('/sensibilidades/estratos')),
  })
