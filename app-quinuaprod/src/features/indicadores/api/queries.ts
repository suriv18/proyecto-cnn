import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api/client'
import { unwrap } from '@/lib/api/errors'

export const useIndicadoresY1 = () =>
  useQuery({ queryKey: ['indicadores', 'y1'], queryFn: async () => unwrap(await api.GET('/indicadores/y1')) })

export const useIndicadoresX1X2 = () =>
  useQuery({ queryKey: ['indicadores', 'x1-x2'], queryFn: async () => unwrap(await api.GET('/indicadores/x1-x2')) })

export const useVariabilidadSemillas = () =>
  useQuery({
    queryKey: ['indicadores', 'variabilidad-semillas'],
    queryFn: async () => unwrap(await api.GET('/indicadores/variabilidad-semillas')),
  })
