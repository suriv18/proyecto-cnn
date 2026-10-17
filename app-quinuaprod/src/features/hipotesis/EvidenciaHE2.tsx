import { Chip } from '@/components/Chip'
import { Dato, Datos } from '@/components/Dato'
import { Tabla } from '@/components/Tabla'
import type { Schemas } from '@/lib/api/client'
import { colorFase } from '@/lib/fases'
import { formatDecimal, formatEtiqueta, formatPorcentaje } from '@/lib/format'
import { evidenciaHE2, type FilaHE2 } from '@/lib/hipotesis'

export function EvidenciaHE2({
  he2,
  criticas,
}: {
  he2: Schemas['Explicabilidad']['he2_tecnico']
  criticas: string[]
}) {
  const filas = evidenciaHE2(he2, criticas)
  return (
    <>
      <p className="flex flex-wrap items-center gap-2 text-sm text-muted">
        <Chip tono={he2.vinculante ? 'ok' : 'aviso'}>{he2.vinculante ? 'Vinculante' : 'Ejecución técnica · no vinculante'}</Chip>
        <span>Campañas de entrenamiento {he2.campanas_evaluadas.join(' y ')}. No decide HE2.</span>
      </p>
      <Tabla<FilaHE2>
        titulo="Evidencia descriptiva de HE2c en el piloto"
        fuente="reports/he2_tecnico/resultados.json"
        columnas={[
          { clave: 'campana', etiqueta: 'Campaña', render: (f) => f.campana },
          {
            clave: 'fase',
            etiqueta: 'Fase de máxima contribución',
            render: (f) => (
              <span className="inline-flex items-center gap-2">
                <span className="size-2.5 rounded-full" style={{ background: colorFase(f.fase) }} aria-hidden="true" />
                {formatEtiqueta(f.fase)}
              </span>
            ),
          },
          {
            clave: 'concuerda',
            etiqueta: `¿Es ${criticas.map(formatEtiqueta).join(' o ').toLowerCase()}?`,
            render: (f) => (
              <span className={f.concuerda ? 'text-ok' : 'text-aviso'}>
                <span aria-hidden="true">{f.concuerda ? '✓ ' : '✗ '}</span>
                {f.concuerda ? 'Sí' : 'No'}
              </span>
            ),
          },
          {
            clave: 'brecha',
            etiqueta: 'Ventaja sobre la 2.ª fase',
            derecha: true,
            render: (f) => formatPorcentaje(f.brecha),
            valor: (f) => f.brecha,
          },
          { clave: 'rango', etiqueta: 'Rango en la oclusión (1 a 5)', derecha: true, render: (f) => f.rango },
        ]}
        filas={filas}
        idFila={(f) => String(f.campana)}
      />
      <p className="text-sm text-muted">
        La ventaja sobre la segunda fase indica cuán frágil es la fase máxima: si es pequeña, un cambio mínimo en la
        semilla o en el fondo de SHAP puede cambiar el resultado de HE2c. El rango mide la fidelidad: 1 significa que
        ocluir esa fase es lo que más daña al modelo.
      </p>
      <Datos etiqueta="Estabilidad descriptiva de HE2" columnas="lg:grid-cols-3">
        <Dato etiqueta="W de Kendall">{formatDecimal(he2.w_kendall)}</Dato>
        <Dato etiqueta="Jaccard promedio">{formatDecimal(he2.jaccard_promedio)}</Dato>
        <Dato etiqueta="HE2 respaldada (técnico)">{he2.he2_respaldada ? 'Sí' : 'No'}</Dato>
      </Datos>
    </>
  )
}
