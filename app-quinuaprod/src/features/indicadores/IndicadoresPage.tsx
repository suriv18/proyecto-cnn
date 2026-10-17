import { Dato, Datos } from '@/components/Dato'
import { EstadoConsulta } from '@/components/EstadoConsulta'
import { PageHeader } from '@/components/PageHeader'
import { Seccion } from '@/components/Seccion'
import { Tabla } from '@/components/Tabla'
import type { Schemas } from '@/lib/api/client'
import { formatDecimal, formatEntero, formatEtiqueta, formatPorcentaje } from '@/lib/format'
import { esReferencia, etiquetaModelo, ORDEN_MODELOS } from '@/lib/modelos'
import { useIndicadoresX1X2, useIndicadoresY1, useVariabilidadSemillas } from './api/queries'

const porCampanaYModelo = <T extends { modelo: Schemas['Modelo']; campana_prueba: number }>(filas: T[]) =>
  [...filas].sort(
    (a, b) => a.campana_prueba - b.campana_prueba || ORDEN_MODELOS.indexOf(a.modelo) - ORDEN_MODELOS.indexOf(b.modelo),
  )


export default function IndicadoresPage() {
  const y1 = useIndicadoresY1()
  const x1x2 = useIndicadoresX1X2()
  const semillas = useVariabilidadSemillas()

  return (
    <>
      <PageHeader
        eyebrow="Resultados"
        titulo="Indicadores"
        descripcion="Exactitud (Y1), modelado (X1) y explicabilidad (X2) del Anexo 01, medidos sobre las campañas de prueba del piloto."
      />

      <Seccion
        titulo="Exactitud (Y1)"
        descripcion="Error de predicción por modelo y campaña de prueba. Las filas B1-B4 son líneas base, no modelos entrenados; el modelo en estudio (CNN-1D) va resaltado."
      >
        <EstadoConsulta query={y1}>
          {(filas) => (
            <Tabla
              titulo="Indicadores de exactitud Y1"
              fuente="reports/piloto_oficial/p11_indicadores_y1.csv"
              claseFila={(f) => (f.modelo === 'cnn1d' ? 'bg-accent-soft/50 font-medium' : '')}
              columnas={[
                { clave: 'campana', etiqueta: 'Campaña', render: (f) => f.campana_prueba, valor: (f) => f.campana_prueba },
                {
                  clave: 'modelo',
                  etiqueta: 'Modelo',
                  render: (f) => (esReferencia(f.modelo) ? `${etiquetaModelo(f.modelo)} (línea base)` : etiquetaModelo(f.modelo)),
                  valor: (f) => ORDEN_MODELOS.indexOf(f.modelo),
                },
                { clave: 'rmse', etiqueta: 'RMSE', unidad: 'kg/ha', derecha: true, render: (f) => formatDecimal(f.rmse, 1), valor: (f) => f.rmse },
                { clave: 'mae', etiqueta: 'MAE', unidad: 'kg/ha', derecha: true, render: (f) => formatDecimal(f.mae, 1), valor: (f) => f.mae },
                { clave: 'rrmse', etiqueta: 'rRMSE', unidad: '%', derecha: true, render: (f) => formatDecimal(f.rrmse, 1), valor: (f) => f.rrmse },
                { clave: 'r2', etiqueta: 'R²', derecha: true, render: (f) => formatDecimal(f.r2), valor: (f) => f.r2 },
              ]}
              filas={porCampanaYModelo(filas)}
              idFila={(f) => `${f.modelo}-${f.campana_prueba}`}
            />
          )}
        </EstadoConsulta>
      </Seccion>

      <EstadoConsulta query={x1x2}>
        {({ x1, x2 }) => (
          <>
            <Seccion titulo="Modelado (X1)" descripcion="Propiedades del CNN-1D y verificación de que sus ejecuciones son válidas y reproducibles.">
              <Datos etiqueta="Indicadores X1">
                <Dato etiqueta="Campo receptivo">{x1.campo_receptivo_pasos} pasos</Dato>
                <Dato etiqueta="Predicciones válidas">{formatPorcentaje(x1.predicciones_validas)}</Dato>
                <Dato etiqueta="Ejecuciones reproducibles">{formatPorcentaje(x1.ejecuciones_reproducibles)}</Dato>
                <Dato etiqueta="Semillas verificadas">{x1.semillas_verificadas.join(', ')}</Dato>
              </Datos>
            </Seccion>

            <Seccion
              titulo="Explicabilidad (X2)"
              descripcion="Qué parte de las predicciones se explicó con SHAP. El detalle por predictor y por fase está en Explicabilidad."
            >
              <Datos etiqueta="Indicadores X2" columnas="lg:grid-cols-3">
                <Dato etiqueta="Predicciones explicadas" destacado>
                  {formatPorcentaje(x2.fraccion_explicada)}
                </Dato>
                <Dato etiqueta="Explicadas / total">
                  {formatEntero(x2.n_explicadas)} de {formatEntero(x2.n_predicciones)}
                </Dato>
                <Dato etiqueta="Predictor principal">{formatEtiqueta(x2.orden_de_predictores[0])}</Dato>
              </Datos>
            </Seccion>
          </>
        )}
      </EstadoConsulta>

      <Seccion titulo="Variabilidad entre semillas" descripcion="RMSE de los modelos estocásticos repetidos con distintas semillas.">
        <EstadoConsulta query={semillas}>
          {(filas) => (
            <Tabla
              titulo="Variabilidad del RMSE entre semillas"
              fuente="reports/piloto_oficial/p6_variabilidad_por_semilla.csv"
              columnas={[
                { clave: 'campana', etiqueta: 'Campaña', render: (f) => f.campana_prueba, valor: (f) => f.campana_prueba },
                { clave: 'modelo', etiqueta: 'Modelo', render: (f) => etiquetaModelo(f.modelo) },
                { clave: 'n', etiqueta: 'Semillas', derecha: true, render: (f) => f.n_semillas },
                { clave: 'mediana', etiqueta: 'Mediana', unidad: 'kg/ha', derecha: true, render: (f) => formatDecimal(f.mediana, 1), valor: (f) => f.mediana },
                { clave: 'rango', etiqueta: 'Mín. – Máx.', unidad: 'kg/ha', derecha: true, render: (f) => `${formatDecimal(f.minimo, 1)} – ${formatDecimal(f.maximo, 1)}` },
                { clave: 'ee', etiqueta: 'Error estándar', derecha: true, render: (f) => formatDecimal(f.error_estandar_mediana) },
              ]}
              filas={porCampanaYModelo(filas)}
              idFila={(f) => `${f.modelo}-${f.campana_prueba}`}
            />
          )}
        </EstadoConsulta>
      </Seccion>
    </>
  )
}
