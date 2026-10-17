/**
 * Nombres para mostrar de los identificadores del contrato (variables, fases, clasificaciones...).
 * Los ids vienen sin tildes ni siglas ("ndvi_max"); aquí se escriben como se leen.
 */
const ETIQUETAS: Record<string, string> = {
  // Variables del tensor
  precipitacion: 'Precipitación',
  temperatura_maxima: 'Temperatura máxima',
  temperatura_minima: 'Temperatura mínima',
  radiacion_solar: 'Radiación solar',
  ndvi_mean: 'NDVI medio',
  ndvi_max: 'NDVI máximo',
  // Fases fenológicas
  emergencia: 'Emergencia',
  desarrollo_vegetativo: 'Desarrollo vegetativo',
  floracion: 'Floración',
  llenado_grano: 'Llenado de grano',
  madurez: 'Madurez',
  // Clasificación territorial
  altoandina: 'Altoandina',
  transicion: 'Transición',
  costa_riego: 'Costa con riego',
  selva: 'Selva',
  pendiente_verificacion: 'Pendiente de verificación',
  // Variantes de sensibilidad
  base: 'Base',
  sin_transicion: 'Sin transición',
  tensor_completo: 'Tensor completo',
  horizonte_temprano: 'Horizonte temprano',
  sin_arrastre_ndvi: 'Sin arrastre de NDVI',
  con_exclusion_por_altitud: 'Con exclusión por altitud',
  // Motivos de exclusión
  fuera_del_ambito_altoandino: 'Fuera del ámbito altoandino',
  ciclo_fuera_de_ventana_de_extraccion: 'Ciclo fuera de la ventana de extracción',
  tensor_incompleto: 'Tensor incompleto',
  // Modo del piloto y estados de E5
  oficial: 'Oficial',
  tecnico: 'Técnico',
  borrador: 'Borrador',
  fijado_por_tesis: 'Fijado por tesis',
  fijado: 'Fijado',
  propuesto: 'Propuesto',
  por_decidir_piloto: 'Por decidir (piloto)',
  por_decidir_asesor: 'Por decidir (asesor)',
}

export function etiquetaDe(id: string): string | undefined {
  return ETIQUETAS[id]
}
