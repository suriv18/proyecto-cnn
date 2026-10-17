import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api/client'
import { unwrap } from '@/lib/api/errors'

export function useEstadoPiloto() {
  return useQuery({
    queryKey: ['meta'],
    queryFn: async () => unwrap(await api.GET('/meta')),
    staleTime: Infinity,
  })
}
