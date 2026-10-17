import { useEffect, useState } from 'react'
import { NavLink, Outlet, ScrollRestoration, useLocation } from 'react-router'
import { Logotipo } from '@/components/Marca'
import { GRUPOS, NAVEGACION } from '@/config/navegacion'
import { AvisoPiloto } from './AvisoPiloto'
import { ThemeToggle } from './ThemeToggle'

function Navegacion({ onNavegar }: { onNavegar?: () => void }) {
  return (
    <nav aria-label="Principal" className="space-y-6">
      {GRUPOS.map((grupo) => (
        <div key={grupo}>
          <p className="mb-1.5 px-3 text-[0.6875rem] font-semibold uppercase tracking-[0.12em] text-muted">{grupo}</p>
          <ul className="space-y-0.5">
            {NAVEGACION.filter((i) => i.grupo === grupo).map((item) => (
              <li key={item.ruta}>
                <NavLink
                  to={item.ruta}
                  end={item.ruta === '/'}
                  onClick={onNavegar}
                  className={({ isActive }) =>
                    `block rounded-md border-l-[3px] px-3 py-2 text-sm transition-colors ${
                      isActive
                        ? 'border-accent bg-accent-soft font-semibold text-ink'
                        : 'border-transparent font-medium text-muted hover:bg-sunken hover:text-ink'
                    }`
                  }
                >
                  {item.etiqueta}
                </NavLink>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </nav>
  )
}

export function AppLayout() {
  const { pathname } = useLocation()
  // El menú móvil queda ligado a la ruta en que se abrió: al navegar se cierra sin efectos.
  const [abiertoEn, setAbiertoEn] = useState<string | null>(null)
  const menuAbierto = abiertoEn === pathname
  const setMenuAbierto = (abrir: boolean) => setAbiertoEn(abrir ? pathname : null)

  useEffect(() => {
    if (!menuAbierto) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setAbiertoEn(null)
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [menuAbierto])

  return (
    <div className="min-h-dvh lg:grid lg:grid-cols-[17rem_minmax(0,1fr)]">
      <a
        href="#contenido"
        className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:rounded-md focus:bg-paper focus:px-3 focus:py-2 focus:shadow-lg"
      >
        Saltar al contenido
      </a>

      <aside className="hidden border-r border-line bg-paper lg:block">
        <div className="sticky top-0 flex max-h-dvh flex-col gap-8 overflow-y-auto p-5">
          <Logotipo />
          <Navegacion />
        </div>
      </aside>

      <div className="flex min-w-0 flex-col">
        <header className="flex items-center justify-between gap-3 border-b border-line bg-paper px-4 py-2.5 sm:px-6">
          <div className="flex items-center gap-3 lg:hidden">
            <button
              type="button"
              aria-expanded={menuAbierto}
              aria-controls="menu-movil"
              onClick={() => setMenuAbierto(!menuAbierto)}
              className="rounded-md border border-line-strong px-3 py-1.5 text-sm font-medium hover:bg-sunken"
            >
              Menú
            </button>
            <Logotipo compacto />
          </div>
          <div className="ml-auto">
            <ThemeToggle />
          </div>
        </header>

        {menuAbierto && (
          <div id="menu-movil" className="border-b border-line bg-paper p-4 lg:hidden">
            <Navegacion onNavegar={() => setMenuAbierto(false)} />
          </div>
        )}

        <AvisoPiloto />

        <main id="contenido" className="mx-auto w-full max-w-6xl flex-1 px-4 py-8 sm:px-8 sm:py-10">
          <Outlet />
        </main>

        <footer className="border-t border-line px-4 py-4 text-xs text-muted sm:px-8">
          Tesis de maestría: modelo CNN-1D interpretable con SHAP para predecir el rendimiento de quinua. SHAP describe
          la función aprendida, no causalidad; la escala es provincia, nunca parcela.
        </footer>
      </div>
      <ScrollRestoration />
    </div>
  )
}
