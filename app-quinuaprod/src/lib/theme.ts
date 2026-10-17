export type Tema = 'claro' | 'oscuro' | 'sistema'

export const CLAVE_TEMA = 'quinuaprod-tema'

export function leerTema(): Tema {
  try {
    const v = localStorage.getItem(CLAVE_TEMA)
    if (v === 'claro' || v === 'oscuro' || v === 'sistema') return v
  } catch {
    // localStorage puede no estar disponible (modo privado, datos bloqueados).
  }
  return 'sistema'
}

export function guardarTema(tema: Tema): void {
  try {
    localStorage.setItem(CLAVE_TEMA, tema)
  } catch {
    // La preferencia es solo una comodidad; la app funciona sin ella.
  }
}

export function esOscuro(tema: Tema, prefiereOscuro: boolean): boolean {
  return tema === 'oscuro' || (tema === 'sistema' && prefiereOscuro)
}

export function aplicarTema(tema: Tema): void {
  const prefiereOscuro = window.matchMedia('(prefers-color-scheme: dark)').matches
  document.documentElement.classList.toggle('dark', esOscuro(tema, prefiereOscuro))
}
