import { Link } from 'react-router'
import { PageHeader } from '@/components/PageHeader'

export function NotFound() {
  return (
    <>
      <PageHeader titulo="Página no encontrada" descripcion="La dirección no corresponde a ninguna pantalla." />
      <Link to="/" className="font-medium text-accent underline underline-offset-4">
        Volver al resumen
      </Link>
    </>
  )
}
