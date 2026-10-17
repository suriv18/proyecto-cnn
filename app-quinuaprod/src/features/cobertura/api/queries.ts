import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api/client'
import type { paths } from '@/lib/api/schema'
import { unwrap } from '@/lib/api/errors'

type Q<P extends keyof paths> = paths[P] extends { get: { parameters: { query?: infer T } } } ? T : never

export const useEmbudo = () =>
  useQuery({ queryKey: ['cobertura', 'embudo'], queryFn: async () => unwrap(await api.GET('/cobertura/embudo')) })

export const useProvincias = (query?: Q<'/provincias'>) =>
  useQuery({
    queryKey: ['provincias', query ?? {}],
    queryFn: async () => unwrap(await api.GET('/provincias', { params: { query } })),
  })

export const useExclusiones = (query?: Q<'/cobertura/exclusiones'>) =>
  useQuery({
    queryKey: ['cobertura', 'exclusiones', query ?? {}],
    queryFn: async () => unwrap(await api.GET('/cobertura/exclusiones', { params: { query } })),
  })
