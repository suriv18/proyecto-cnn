import { createBrowserRouter, createMemoryRouter, type RouteObject } from 'react-router'
import { AppLayout } from './layout/AppLayout'
import { Cargando } from './Cargando'
import { NotFound } from './NotFound'
import { RouteError } from './RouteError'

// Cada pantalla se carga bajo demanda; el módulo exporta `default` como componente de la ruta.
const pagina = (cargar: () => Promise<{ default: React.ComponentType }>) => async () => ({
  Component: (await cargar()).default,
})

export const routes: RouteObject[] = [
  {
    path: '/',
    Component: AppLayout,
    ErrorBoundary: RouteError,
    HydrateFallback: Cargando,
    children: [
      { index: true, lazy: pagina(() => import('./ResumenPage')) },
      { path: 'cobertura', lazy: pagina(() => import('@/features/cobertura/CoberturaPage')) },
      { path: 'predicciones', lazy: pagina(() => import('@/features/predicciones/PrediccionesPage')) },
      { path: 'indicadores', lazy: pagina(() => import('@/features/indicadores/IndicadoresPage')) },
      { path: 'explicabilidad', lazy: pagina(() => import('@/features/explicabilidad/ExplicabilidadPage')) },
      { path: 'sensibilidades', lazy: pagina(() => import('@/features/sensibilidades/SensibilidadesPage')) },
      { path: 'preespecificacion', lazy: pagina(() => import('@/features/preespecificacion/PreespecificacionPage')) },
      { path: 'inferencia', lazy: pagina(() => import('@/features/inferencia/InferenciaPage')) },
      { path: '*', Component: NotFound },
    ],
  },
]

export const createAppRouter = () => createBrowserRouter(routes)
export const createTestRouter = (initialEntries: string[] = ['/']) => createMemoryRouter(routes, { initialEntries })
