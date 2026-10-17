import { render, screen } from '@testing-library/react'
import { RouterProvider } from 'react-router'
import { AppProviders } from '@/app/providers'
import { createTestRouter } from '@/app/routes'

/** Monta la app completa (proveedores + rutas reales) en una ruta inicial, con MSW activo. */
export async function renderApp(ruta = '/') {
  const router = createTestRouter([ruta])
  const vista = render(
    <AppProviders>
      <RouterProvider router={router} />
    </AppProviders>,
  )
  // Las rutas se cargan bajo demanda: se espera al layout antes de consultar.
  // La primera carga en frío de una pantalla con gráficos puede tardar más que el límite por defecto.
  await screen.findByRole('main', {}, { timeout: 10_000 })
  return { router, ...vista }
}
