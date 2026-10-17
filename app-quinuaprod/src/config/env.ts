// Un valor vacío cuenta como «sin definir»: un build de Docker con `ARG VITE_API_URL=` deja la variable en ''
// y la interfaz debe seguir llamando a /api/v1 en su propio origen.
export const API_URL = import.meta.env.VITE_API_URL || '/api/v1'
export const USE_MOCKS = import.meta.env.VITE_USE_MOCKS === 'true'
