import { CicloFenologico } from '@/components/CicloFenologico'
import { EstadoConsulta } from '@/components/EstadoConsulta'
import { GraficoBarras } from '@/components/Graficos'
import { PageHeader } from '@/components/PageHeader'
import { Seccion } from '@/components/Seccion'
import { Tabla } from '@/components/Tabla'
import { useEmbudo } from '@/features/cobertura/api/queries'
import { EstadoPiloto } from '@/features/piloto/components/EstadoPiloto'
import { useEstadoPiloto } from '@/lib/api/useEstadoPiloto'
import { formatEntero } from '@/lib/format'

export default function ResumenPage() {
  const embudo = useEmbudo()
  const piloto = useEstadoPiloto()

  return (
    <>
      <PageHeader
        eyebrow="Resultados"
        titulo="Resumen"
        descripcion="Qué datos entran al modelo CNN-1D que predice el rendimiento de quinua por provincia y campaña, y en qué estado está el piloto."
      />

      <Seccion titulo="Estado del piloto">
        <EstadoPiloto />
      </Seccion>

      <Seccion
        titulo="Estructura de entrada"
        descripcion="Cada provincia-campaña se describe con las variables climáticas y de vegetación de las cinco fases del cultivo, más la altitud y la proporción de quinua."
      >
        <EstadoConsulta query={piloto}>
          {(meta) => <CicloFenologico fases={meta.orden_fases} variables={meta.orden_variables} />}
        </EstadoConsulta>
      </Seccion>

      <Seccion
        titulo="Embudo de depuración"
        descripcion="Observaciones provincia-campaña que quedan tras cada nivel de depuración (N0 a N5)."
      >
        <EstadoConsulta query={embudo}>
          {(niveles) => {
            const calculados = niveles.filter((n) => n.n !== null)
            return (
              <>
                <GraficoBarras
                  titulo="Observaciones por nivel"
                  descripcion={`Observaciones por nivel de depuración: ${calculados.map((n) => `${n.nivel} ${n.n}`).join(', ')}.`}
                  datos={calculados.map((n) => ({ etiqueta: n.nivel, valor: n.n as number }))}
                  formato={formatEntero}
                />
                <Tabla
                  titulo="Embudo de depuración"
                  fuente="reports/E1_matriz_cobertura_enmendado_2016_2025.json"
                  columnas={[
                    { clave: 'nivel', etiqueta: 'Nivel', render: (n) => n.nivel },
                    {
                      clave: 'n',
                      etiqueta: 'Observaciones',
                      derecha: true,
                      render: (n) => (n.n === null ? 'Sin calcular' : formatEntero(n.n)),
                    },
                  ]}
                  filas={niveles}
                  idFila={(n) => n.nivel}
                />
              </>
            )
          }}
        </EstadoConsulta>
      </Seccion>
    </>
  )
}
