import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api/client'
import { unwrap } from '@/lib/api/errors'

export const useIndicadoresParaHipotesis = () =>
  useQuery({ queryKey: ['indicadores', 'y1'], queryFn: async () => unwrap(await api.GET('/indicadores/y1')) })

export const useExplicabilidadParaHipotesis = () =>
  useQuery({ queryKey: ['explicabilidad'], queryFn: async () => unwrap(await api.GET('/explicabilidad')) })

export const usePreespecificacionParaHipotesis = () =>
  useQuery({ queryKey: ['preespecificacion'], queryFn: async () => unwrap(await api.GET('/preespecificacion')) })
