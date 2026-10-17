import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api/client'
import { unwrap } from '@/lib/api/errors'

export const usePreespecificacion = () =>
  useQuery({ queryKey: ['preespecificacion'], queryFn: async () => unwrap(await api.GET('/preespecificacion')) })
