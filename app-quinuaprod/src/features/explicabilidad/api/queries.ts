import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api/client'
import { unwrap } from '@/lib/api/errors'

export const useExplicabilidad = () =>
  useQuery({ queryKey: ['explicabilidad'], queryFn: async () => unwrap(await api.GET('/explicabilidad')) })

export const useImportanciaPredictores = () =>
  useQuery({ queryKey: ['indicadores', 'x1-x2'], queryFn: async () => unwrap(await api.GET('/indicadores/x1-x2')) })
