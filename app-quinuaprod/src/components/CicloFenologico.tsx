import { colorFase, etiquetaFaseCorta } from '@/lib/fases'
import { formatEtiqueta } from '@/lib/format'

/**
 * Estructura del tensor de entrada: variables (filas) × fases del ciclo (columnas).
 * El orden de las fases es el del cultivo; cada fase conserva su color en toda la app.
 */
export function CicloFenologico({ fases, variables }: { fases: string[]; variables: string[] }) {
  return (
    <figure className="rounded-lg border border-line bg-paper p-4">
      <div className="relative overflow-x-auto" role="region" aria-label="Tensor de entrada del modelo" tabIndex={0}>
        <table className="w-full min-w-[34rem] border-separate border-spacing-y-1 text-sm">
          <caption className="sr-only">Variables y fases del tensor de entrada</caption>
          <thead>
            <tr>
              <th scope="col" className="w-40 pb-1 text-left text-xs font-medium text-muted">
                Variable
              </th>
              {fases.map((f) => (
                <th key={f} scope="col" className="pb-1 text-left">
                  <span className="block h-1.5 rounded-full" style={{ background: colorFase(f) }} aria-hidden="true" />
                  <span className="mt-1.5 block text-xs font-medium">{etiquetaFaseCorta(f)}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {variables.map((v) => (
              <tr key={v}>
                <th scope="row" className="pr-3 text-left font-medium">
                  {formatEtiqueta(v)}
                </th>
                {fases.map((f) => (
                  <td key={f}>
                    <span className="block h-2.5 rounded-sm opacity-25" style={{ background: colorFase(f) }} aria-hidden="true" />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <figcaption className="mt-3 text-xs text-muted">
        Cada observación provincia-campaña es una matriz de {variables.length} variables por {fases.length} fases; el
        modelo lee el ciclo de izquierda a derecha.
      </figcaption>
    </figure>
  )
}
