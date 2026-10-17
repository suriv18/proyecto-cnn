import { isRouteErrorResponse, Link, useRouteError } from 'react-router'
import { Marca } from '@/components/Marca'
import { PageHeader } from '@/components/PageHeader'

export function RouteError() {
  const error = useRouteError()
  const noEncontrada = isRouteErrorResponse(error) && error.status === 404

  return (
    <main className="mx-auto max-w-2xl px-6 py-16">
      <Marca tamano={40} />
      <div className="mt-6">
        <PageHeader
          titulo={noEncontrada ? 'Página no encontrada' : 'Algo salió mal'}
          descripcion={
            noEncontrada
              ? 'La dirección no corresponde a ninguna pantalla.'
              : 'Ocurrió un error inesperado al mostrar esta pantalla. Vuelva al resumen e intente de nuevo.'
          }
        />
      </div>
      <Link to="/" className="font-medium text-accent underline underline-offset-4">
        Volver al resumen
      </Link>
    </main>
  )
}
