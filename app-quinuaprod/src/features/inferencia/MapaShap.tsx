import { colorFase, etiquetaFaseCorta } from '@/lib/fases'
import { formatDecimal, formatEtiqueta } from '@/lib/format'

/** Intensidad máxima del fondo de una celda: el texto siempre conserva contraste suficiente. */
const OPACIDAD_MAXIMA = 40

/** Un valor que se redondea a 0,0 se escribe sin signo: «-0.0» no significa nada. */
const conSigno = (v: number) =>
  Math.abs(v) < 0.05 ? formatDecimal(0, 1) : `${v > 0 ? '+' : ''}${formatDecimal(v, 1)}`

function fondo(valor: number, maximo: number): string | undefined {
  if (maximo <= 0 || valor === 0) return undefined
  const color = valor > 0 ? 'var(--shap-positivo)' : 'var(--shap-negativo)'
  const pct = Math.round((Math.abs(valor) / maximo) * OPACIDAD_MAXIMA)
  if (pct === 0) return undefined
  return `color-mix(in srgb, ${color} ${pct}%, transparent)`
}

/**
 * Valores SHAP por variable (filas) y fase (columnas), en kg/ha de la anomalía. Cada celda escribe su valor y su
 * signo: el color refuerza, no es la única señal.
 */
export function MapaShap({ valores, variables, fases }: { valores: number[][]; variables: string[]; fases: string[] }) {
  const maximo = Math.max(0, ...valores.flat().map(Math.abs))
  return (
    <figure className="mt-6 rounded-lg border border-line bg-paper p-4">
      <figcaption className="mb-3 text-sm font-semibold">Contribución de cada variable en cada fase (SHAP)</figcaption>
      <div className="relative overflow-x-auto" role="region" aria-label="Mapa de valores SHAP" tabIndex={0}>
        <table className="w-full min-w-[32rem] border-separate border-spacing-1 text-sm">
          <caption className="sr-only">Valores SHAP en kg/ha por variable y fase</caption>
          <thead>
            <tr>
              <th scope="col" className="w-40 text-left text-xs font-medium text-muted">Variable</th>
              {fases.map((f) => (
                <th key={f} scope="col" className="text-right text-xs font-medium">
                  <span className="mb-1 block h-1.5 rounded-full" style={{ background: colorFase(f) }} aria-hidden="true" />
                  <span className="sr-only">{formatEtiqueta(f)}</span>
                  <span aria-hidden="true">{etiquetaFaseCorta(f)}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {variables.map((v, i) => (
              <tr key={v}>
                <th scope="row" className="pr-2 text-left font-medium">{formatEtiqueta(v)}</th>
                {fases.map((f, j) => (
                  <td
                    key={f}
                    className="rounded px-2 py-1.5 text-right tabular-nums"
                    style={{ background: fondo(valores[i][j], maximo) }}
                  >
                    {conSigno(valores[i][j])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <figcaption className="mt-3 text-xs text-muted">
        Valores en kg/ha. <span style={{ color: 'var(--shap-positivo)' }} className="font-semibold">Positivo</span> suma
        a la anomalía y <span style={{ color: 'var(--shap-negativo)' }} className="font-semibold">negativo</span> resta,
        respecto del promedio del modelo sobre un fondo de referencia del entrenamiento. Describe lo que aprendió el
        modelo, no una relación causal. Las covariables no se muestran.
      </figcaption>
    </figure>
  )
}
