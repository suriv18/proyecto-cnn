import { Chip } from '@/components/Chip'
import { Dato, Datos } from '@/components/Dato'
import { EstadoConsulta } from '@/components/EstadoConsulta'
import { GraficoBarras, GraficoFases } from '@/components/Graficos'
import { PageHeader } from '@/components/PageHeader'
import { Seccion } from '@/components/Seccion'
import { Tabla } from '@/components/Tabla'
import { useEstadoPiloto } from '@/lib/api/useEstadoPiloto'
import { colorFase } from '@/lib/fases'
import { formatDecimal, formatEtiqueta } from '@/lib/format'
import { useExplicabilidad, useImportanciaPredictores } from './api/queries'

type PorCampana = Record<string, Record<string, number>>

/** Filas = fases (con su color), columnas = campañas. Las claves de campaña llegan como texto desde JSON. */
function TablaFaseCampana({ titulo, datos, fases }: { titulo: string; datos: PorCampana; fases: string[] }) {
  const campanas = Object.keys(datos).sort()
  return (
    <Tabla
      titulo={titulo}
      columnas={[
        {
          clave: 'fase',
          etiqueta: 'Fase',
          render: (f) => (
            <span className="inline-flex items-center gap-2">
              <span className="size-2.5 rounded-full" style={{ background: colorFase(f) }} aria-hidden="true" />
              {formatEtiqueta(f)}
            </span>
          ),
        },
        ...campanas.map((c) => ({
          clave: c,
          etiqueta: c,
          derecha: true,
          render: (f: string) => formatDecimal(datos[c][f], 3),
          valor: (f: string) => datos[c][f],
        })),
      ]}
      filas={fases}
      idFila={(f) => f}
    />
  )
}

export default function ExplicabilidadPage() {
  const explicabilidad = useExplicabilidad()
  const x1x2 = useImportanciaPredictores()
  const piloto = useEstadoPiloto()
  const fases = piloto.data?.orden_fases ?? []

  return (
    <>
      <PageHeader
        eyebrow="Interpretación"
        titulo="Explicabilidad"
        descripcion="Cuánto aportó cada fase del cultivo y cada variable a las predicciones del CNN-1D, medido con SHAP."
      />
      <p role="note" className="mb-8 rounded-r-lg border-l-4 border-accent bg-accent-soft p-4 text-sm">
        <strong className="font-semibold">Cómo leer esta pantalla.</strong> SHAP describe la función que aprendió el
        modelo, no relaciones causales. Los resultados exportados solo traen agregados por fase y por predictor; no hay
        valores SHAP por provincia ni por celda variable × fase.
      </p>

      <EstadoConsulta query={explicabilidad}>
        {({ piloto_oficial: po, he2_tecnico: he2 }) => {
          const barras = fases.map((f) => ({ fase: f, valor: po.contribucion_por_fase[f] }))
          return (
            <>
              <Seccion
                titulo="Contribución por fase (piloto oficial)"
                descripcion={
                  <>
                    SHAP absoluto medio sobre {po.n_explicadas} predicciones de las campañas {po.campanas.join(' y ')}. Fase
                    de mayor contribución: <strong>{formatEtiqueta(po.fase_maxima)}</strong>.
                  </>
                }
              >
                <GraficoFases
                  titulo="SHAP absoluto medio por fase"
                  descripcion={`Contribución por fase: ${barras.map((b) => `${formatEtiqueta(b.fase)} ${formatDecimal(b.valor, 2)}`).join('; ')}.`}
                  datos={barras}
                  formato={(v) => formatDecimal(v, 2)}
                  etiquetaEje="SHAP absoluto medio"
                />
                <Tabla
                  titulo="Contribución por fase"
                  fuente="reports/piloto_oficial/resultados.json (p8)"
                  columnas={[
                    {
                      clave: 'fase',
                      etiqueta: 'Fase',
                      render: (f) => (
                        <span className="inline-flex items-center gap-2">
                          <span className="size-2.5 rounded-full" style={{ background: colorFase(f) }} aria-hidden="true" />
                          {formatEtiqueta(f)}
                        </span>
                      ),
                    },
                    {
                      clave: 'v',
                      etiqueta: 'SHAP absoluto medio',
                      derecha: true,
                      render: (f) => formatDecimal(po.contribucion_por_fase[f], 3),
                      valor: (f) => po.contribucion_por_fase[f],
                    },
                  ]}
                  filas={fases}
                  idFila={(f) => f}
                />
              </Seccion>

              <Seccion
                titulo="Ejecución técnica de HE2"
                descripcion={
                  <>
                    <Chip tono={he2.vinculante ? 'ok' : 'aviso'}>{he2.vinculante ? 'Vinculante' : 'No vinculante'}</Chip>{' '}
                    <span>
                      {he2.vinculante ? 'Resultado vinculante.' : 'Ejecución técnica, no vinculante.'} Campañas evaluadas:{' '}
                      {he2.campanas_evaluadas.join(', ')}.
                    </span>
                  </>
                }
              >
                <Datos etiqueta="Estabilidad de HE2" columnas="lg:grid-cols-3">
                  <Dato etiqueta="W de Kendall">{formatDecimal(he2.w_kendall)}</Dato>
                  <Dato etiqueta="Jaccard promedio">{formatDecimal(he2.jaccard_promedio)}</Dato>
                  <Dato etiqueta="HE2 respaldada">{he2.he2_respaldada ? 'Sí' : 'No'}</Dato>
                </Datos>
                <h3 className="mt-6 text-base font-semibold">Contribución por fase y campaña</h3>
                <TablaFaseCampana titulo="Contribución SHAP por fase y campaña" datos={he2.contribuciones as PorCampana} fases={fases} />
                <h3 className="mt-6 text-base font-semibold">Aumento del RMSE al ocluir cada fase</h3>
                <TablaFaseCampana titulo="Aumento del RMSE por oclusión de fase" datos={he2.aumentos_rmse as PorCampana} fases={fases} />
                <p className="font-mono text-[0.6875rem] text-muted">Fuente: reports/he2_tecnico/resultados.json</p>
              </Seccion>
            </>
          )
        }}
      </EstadoConsulta>

      <Seccion titulo="Importancia por predictor" descripcion="De la variable que más pesa en las predicciones a la que menos.">
        <EstadoConsulta query={x1x2}>
          {({ x2 }) => {
            const orden = x2.orden_de_predictores
            const barras = orden.map((p) => ({ etiqueta: formatEtiqueta(p), valor: x2.importancia_por_predictor[p] }))
            return (
              <>
                <GraficoBarras
                  titulo="SHAP absoluto medio por predictor"
                  descripcion={`Importancia por predictor, de mayor a menor: ${barras.map((b) => `${b.etiqueta} ${formatDecimal(b.valor, 2)}`).join('; ')}.`}
                  datos={barras}
                  formato={(v) => formatDecimal(v, 2)}
                />
                <Tabla
                  titulo="Importancia por predictor"
                  fuente="reports/piloto_oficial/resultados.json (p11, X2)"
                  columnas={[
                    { clave: 'rango', etiqueta: '#', render: (p) => orden.indexOf(p) + 1 },
                    { clave: 'p', etiqueta: 'Predictor', render: (p) => formatEtiqueta(p) },
                    { clave: 'v', etiqueta: 'SHAP absoluto medio', derecha: true, render: (p) => formatDecimal(x2.importancia_por_predictor[p], 3) },
                  ]}
                  filas={orden}
                  idFila={(p) => p}
                />
              </>
            )
          }}
        </EstadoConsulta>
      </Seccion>
    </>
  )
}
