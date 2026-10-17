import { useMemo, useState, type ReactNode } from 'react'
import { Chip } from '@/components/Chip'
import { EstadoConsulta } from '@/components/EstadoConsulta'
import { PageHeader } from '@/components/PageHeader'
import { Seccion } from '@/components/Seccion'
import { aplanar } from '@/lib/aplanar'
import { formatEtiqueta, formatPorcentaje } from '@/lib/format'
import {
  useExplicabilidadParaHipotesis,
  useIndicadoresParaHipotesis,
  usePreespecificacionParaHipotesis,
} from './api/queries'
import { DEFINICIONES, REDACCIONES, type RedaccionId } from './contenido'
import { EvidenciaHE1 } from './EvidenciaHE1'
import { EvidenciaHE2 } from './EvidenciaHE2'
import { MatrizConsistencia } from './MatrizConsistencia'

function Criterios({ filas }: { filas: { etiqueta: string; texto: ReactNode }[] }) {
  return (
    <dl className="grid gap-px overflow-hidden rounded-lg border border-line bg-line text-sm sm:grid-cols-[13rem_1fr]">
      {filas.map((f) => (
        <div key={f.etiqueta} className="contents">
          <dt className="bg-sunken px-4 py-2.5 font-medium">{f.etiqueta}</dt>
          <dd className="bg-paper px-4 py-2.5">{f.texto}</dd>
        </div>
      ))}
    </dl>
  )
}

function SelectorRedaccion({ valor, onChange }: { valor: RedaccionId; onChange: (v: RedaccionId) => void }) {
  return (
    <fieldset className="mb-6">
      <legend className="mb-2 text-xs font-medium text-muted">Redacción</legend>
      <div className="inline-flex rounded-md border border-line-strong bg-paper p-0.5">
        {Object.values(REDACCIONES).map((r) => (
          <label
            key={r.id}
            className="cursor-pointer rounded px-3 py-1.5 text-sm font-medium has-[:checked]:bg-accent has-[:checked]:text-accent-ink has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-accent"
          >
            <input
              type="radio"
              name="redaccion"
              value={r.id}
              checked={valor === r.id}
              onChange={() => onChange(r.id)}
              className="sr-only"
            />
            {r.etiqueta}
          </label>
        ))}
      </div>
      <p className="mt-2 text-sm text-muted">{REDACCIONES[valor].descripcion}</p>
    </fieldset>
  )
}

export default function HipotesisPage() {
  const [redaccion, setRedaccion] = useState<RedaccionId>('v3')
  const y1 = useIndicadoresParaHipotesis()
  const explicabilidad = useExplicabilidadParaHipotesis()
  const e5 = usePreespecificacionParaHipotesis()

  // El margen δ y las fases críticas se leen del registro E5, no de constantes de la interfaz.
  const parametros = useMemo(() => (e5.data ? aplanar(e5.data.secciones) : []), [e5.data])
  const valorE5 = (ruta: string) => parametros.find((p) => p.seccion === 'hipotesis' && p.ruta === ruta)?.valor
  const delta = typeof valorE5('margen_no_inferioridad_relativo') === 'number' ? (valorE5('margen_no_inferioridad_relativo') as number) : undefined
  const criticasE5 = valorE5('he2.fases_criticas_horizonte_principal')
  const criticas = Array.isArray(criticasE5) ? criticasE5.map(String) : undefined

  return (
    <>
      <PageHeader
        eyebrow="Referencia"
        titulo="Hipótesis y objetivos"
        descripcion="La cadena problema, objetivo e hipótesis de la tesis, cómo se decide cada hipótesis y qué evidencia descriptiva hay hasta ahora."
      />

      <p role="note" className="mb-8 rounded-r-lg border-l-4 border-accent bg-accent-soft p-4 text-sm">
        <strong className="font-semibold">Ninguna hipótesis está evaluada.</strong> Las campañas externas siguen
        protegidas hasta congelar E5. Lo que se muestra aquí es evidencia descriptiva del piloto, calculada sobre
        campañas de entrenamiento: sirve para entender el problema, no decide HE1, HE2 ni la hipótesis general.
      </p>

      <SelectorRedaccion valor={redaccion} onChange={setRedaccion} />

      <Seccion titulo="Matriz de consistencia" descripcion="Cada fila es un nivel; las columnas son la cadena de la tesis.">
        <MatrizConsistencia niveles={REDACCIONES[redaccion].niveles} />
      </Seccion>

      {redaccion === 'propuesta' && (
        <Seccion titulo="Definiciones" descripcion="La propuesta pide escribirlas una sola vez y reutilizarlas en toda la tesis.">
          <dl className="grid gap-px overflow-hidden rounded-lg border border-line bg-line text-sm sm:grid-cols-[13rem_1fr]">
            {DEFINICIONES.map((d) => (
              <div key={d.termino} className="contents">
                <dt className="bg-sunken px-4 py-2.5 font-medium">{d.termino}</dt>
                <dd className="bg-paper px-4 py-2.5">{d.texto}</dd>
              </div>
            ))}
          </dl>
        </Seccion>
      )}

      <Seccion
        titulo="HE1: exactitud predictiva"
        descripcion={<>Vincula la dimensión <strong>X1</strong> (modelado temporal con CNN-1D) con <strong>Y1</strong> (exactitud predictiva).</>}
      >
        <Criterios
          filas={[
            { etiqueta: 'Criterio primario (HE1a)', texto: 'q < δ en todas las campañas externas, frente a XGBoost y frente a B*, con q = RMSE(CNN-1D) / RMSE(referencia) − 1.' },
            { etiqueta: 'Margen δ (según E5)', texto: delta === undefined ? '—' : formatPorcentaje(delta) },
            { etiqueta: 'Regla de decisión', texto: 'Regla exacta de signos: nivel de significancia 0,5ⁿ (0,031 con cinco campañas). Se reportan la mediana de q y el rango por campaña, sin contrastes de apoyo.' },
            { etiqueta: 'Exploratorio (HE1b)', texto: 'Ganancia de habilidad SS = 1 − MSE(CNN-1D) / MSE(B*). No condiciona la decisión.' },
            { etiqueta: 'Indicadores (Y1)', texto: 'RMSE, MAE, rRMSE y R².' },
          ]}
        />
        <EstadoConsulta query={y1}>
          {(datos) => (delta === undefined ? null : <EvidenciaHE1 y1={datos} delta={delta} />)}
        </EstadoConsulta>
      </Seccion>

      <Seccion
        titulo="HE2: interpretabilidad"
        descripcion={<>Vincula la dimensión <strong>X2</strong> (explicabilidad con SHAP) con <strong>Y2</strong> (interpretabilidad de la predicción).</>}
      >
        <Criterios
          filas={[
            {
              etiqueta: 'Criterio primario (HE2c)',
              texto: `La fase de máxima contribución SHAP pertenece al conjunto preespecificado: ${criticas ? criticas.map(formatEtiqueta).join(' o ').toLowerCase() : '—'} (según E5).`,
            },
            { etiqueta: 'Regla de decisión', texto: 'Binomial exacta unilateral sobre las campañas externas, con probabilidad de azar 0,4: con cinco campañas solo 5 de 5 aciertos da p < 0,05.' },
            { etiqueta: 'Secundarios', texto: 'Estabilidad entre campañas (HE2a: W de Kendall y Jaccard corregido por azar) y fidelidad por oclusión (HE2b). Se reportan como cifras, sin valor p ni corrección, y no deciden HE2 ni compensan al criterio primario.' },
            { etiqueta: 'Indicadores (Y2)', texto: 'W de Kendall, estabilidad de las fases principales, aumento del RMSE por oclusión, rango de la fase máxima y coincidencia de la fase máxima.' },
          ]}
        />
        <EstadoConsulta query={explicabilidad}>
          {({ he2_tecnico }) => (criticas === undefined ? null : <EvidenciaHE2 he2={he2_tecnico} criticas={criticas} />)}
        </EstadoConsulta>
      </Seccion>

      <Seccion titulo="Hipótesis general" descripcion="Es conjuntiva: se respalda solo si se cumplen a la vez el criterio primario de HE1 y el de HE2.">
        <div className="flex flex-wrap items-center gap-3 rounded-lg border border-line bg-paper p-4 text-sm">
          <Chip>HE1a</Chip>
          <span aria-hidden="true">y</span>
          <Chip>HE2c</Chip>
          <span aria-hidden="true">→</span>
          <Chip tono="aviso">Sin evaluar</Chip>
          <span className="text-muted">Los criterios secundarios se reportan con su magnitud, pero no compensan un incumplimiento.</span>
        </div>
      </Seccion>
    </>
  )
}
