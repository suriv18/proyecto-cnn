import { PageHeader } from '@/components/PageHeader'
import { Seccion } from '@/components/Seccion'
import { FormObservacion } from './FormObservacion'
import { FormTensor } from './FormTensor'

export default function InferenciaPage() {
  return (
    <>
      <PageHeader
        eyebrow="Herramienta"
        titulo="Predicción interactiva"
        descripcion="Consulte la predicción del CNN-1D para una provincia y campaña. La escala es provincia, nunca parcela."
      />

      <Seccion
        titulo="Observación del piloto"
        descripcion="Recupera la predicción que el piloto calculó para una provincia de una campaña de prueba."
      >
        <FormObservacion />
      </Seccion>

      <Seccion
        titulo="Tensor editado"
        descripcion={
          <span role="note">
            Predice con la CNN-1D del piloto sobre un tensor que usted escribe. El modelo se entrenó solo con campañas de
            entrenamiento y su estado se indica en el resultado. Si el servidor no tiene el modelo serializado, la
            respuesta es «No implementado».
          </span>
        }
      >
        <FormTensor />
      </Seccion>
    </>
  )
}
