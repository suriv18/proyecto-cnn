import { z } from 'zod'

const NUMERO = 'Ingrese un número'
const celda = z.number({ error: NUMERO })

/** Tensor 6 variables × 5 fases más las dos covariables, como lo espera POST /predicciones/inferir. */
export const tensorSchema = z.object({
  valores: z.array(z.array(celda).length(5)).length(6),
  altitud_media_m: z.number({ error: NUMERO }).min(0, 'No puede ser negativa'),
  prop_quinua: z.number({ error: NUMERO }).min(0, 'No puede ser negativa'),
})

export type TensorForm = z.infer<typeof tensorSchema>
