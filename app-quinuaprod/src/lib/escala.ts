/** Marcas «redondas» (1, 2, 5 × 10^k) para un eje que parte de cero, con unas 4-5 divisiones. */
export function escalaAgradable(maximo: number, divisiones = 4): { max: number; marcas: number[] } {
  if (!Number.isFinite(maximo) || maximo <= 0) return { max: 1, marcas: [0, 1] }
  const crudo = maximo / divisiones
  const potencia = 10 ** Math.floor(Math.log10(crudo))
  const fraccion = crudo / potencia
  const paso = (fraccion <= 1 ? 1 : fraccion <= 2 ? 2 : fraccion <= 5 ? 5 : 10) * potencia
  const n = Math.ceil(maximo / paso)
  return { max: n * paso, marcas: Array.from({ length: n + 1 }, (_, i) => i * paso) }
}
