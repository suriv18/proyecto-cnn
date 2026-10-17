import { Tabla } from '@/components/Tabla'
import { evidenciaHE1, type FilaHE1 } from '@/lib/hipotesis'
import type { Schemas } from '@/lib/api/client'
import { formatDecimal, formatPorcentaje } from '@/lib/format'

const pct = (q: number) => `${q > 0 ? '+' : ''}${formatDecimal(q * 100, 1)} %`

/** q frente a una referencia, con marca explícita (no solo color) de si queda dentro del margen δ. */
function CeldaQ({ q, delta }: { q: number | undefined; delta: number }) {
  if (q === undefined) return <span>—</span>
  const dentro = q < delta
  return (
    <span className={dentro ? 'text-ok' : 'text-aviso'} title={dentro ? 'Dentro del margen' : 'Fuera del margen'}>
      <span aria-hidden="true">{dentro ? '✓ ' : '✗ '}</span>
      {pct(q)}
      <span className="sr-only">{dentro ? ', dentro del margen' : ', fuera del margen'}</span>
    </span>
  )
}

export function EvidenciaHE1({ y1, delta }: { y1: Schemas['IndicadorY1'][]; delta: number }) {
  const filas = evidenciaHE1(y1)
  const celdaSS = (v: number | undefined) => (v === undefined ? '—' : formatDecimal(v, 2))
  return (
    <>
      <p className="text-sm text-muted">
        q = RMSE(CNN-1D) / RMSE(referencia) − 1, con margen δ = {formatPorcentaje(delta)}: ✓ si q &lt; δ. B* todavía no
        está fijada (se elige con la validación interna), por eso se muestran B2 y B4 por separado. SS es la ganancia de
        habilidad exploratoria: positiva si el CNN-1D mejora a la referencia.
      </p>
      <Tabla<FilaHE1>
        titulo="Evidencia descriptiva de HE1 en el piloto"
        fuente="reports/piloto_oficial/p11_indicadores_y1.csv"
        columnas={[
          { clave: 'campana', etiqueta: 'Campaña', render: (f) => f.campana },
          { clave: 'xgb', etiqueta: 'q vs XGBoost', derecha: true, render: (f) => <CeldaQ q={f.qXgboost} delta={delta} /> },
          { clave: 'b2', etiqueta: 'q vs B2', derecha: true, render: (f) => <CeldaQ q={f.qB2} delta={delta} /> },
          { clave: 'b4', etiqueta: 'q vs B4', derecha: true, render: (f) => <CeldaQ q={f.qB4} delta={delta} /> },
          { clave: 'ssb2', etiqueta: 'SS vs B2', derecha: true, render: (f) => celdaSS(f.ssB2) },
          { clave: 'ssb4', etiqueta: 'SS vs B4', derecha: true, render: (f) => celdaSS(f.ssB4) },
        ]}
        filas={filas}
        idFila={(f) => String(f.campana)}
      />
    </>
  )
}
