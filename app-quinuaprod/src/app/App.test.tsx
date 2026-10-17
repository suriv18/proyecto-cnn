import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it } from 'vitest'
import { NAVEGACION } from '@/config/navegacion'
import { CLAVE_TEMA } from '@/lib/theme'
import { renderApp } from '@/testing/render'

afterEach(() => {
  localStorage.clear()
  document.documentElement.classList.remove('dark')
})

describe('shell de la aplicación', () => {
  it('el resumen muestra el estado real del piloto obtenido de /meta', async () => {
    await renderApp('/')
    const estado = await screen.findByLabelText('Estado del piloto')
    expect(estado).toHaveTextContent('2019, 2020')
    expect(estado).not.toHaveTextContent('2021')
  })

  it('la franja de aviso describe los datos sin nombrar campañas externas', async () => {
    await renderApp('/')
    const aviso = await screen.findByRole('note', { name: 'Aviso sobre los datos' })
    expect(aviso).toHaveTextContent(/campañas externas permanecen protegidas/)
    expect(aviso.textContent).not.toMatch(/202[1-5]/)
  })

  it.each(NAVEGACION)('la ruta $ruta se renderiza con su título', async ({ ruta, etiqueta }) => {
    await renderApp(ruta)
    // El título de la pantalla es el único h1; la etiqueta de navegación puede diferir ("Resumen" vs. "Registro E5").
    const h1 = await screen.findByRole('heading', { level: 1 })
    expect(h1).toBeInTheDocument()
    const nav = screen.getAllByRole('navigation', { name: 'Principal' })[0]
    expect(within(nav).getByRole('link', { name: etiqueta })).toHaveAttribute('aria-current', 'page')
  })

  it('una ruta inexistente muestra la página no encontrada', async () => {
    await renderApp('/no-existe')
    expect(await screen.findByRole('heading', { name: 'Página no encontrada' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Volver al resumen' })).toHaveAttribute('href', '/')
  })

  it('navega entre pantallas con los enlaces del menú', async () => {
    const user = userEvent.setup()
    await renderApp('/')
    const nav = screen.getAllByRole('navigation', { name: 'Principal' })[0]
    await user.click(within(nav).getByRole('link', { name: 'Cobertura' }))
    expect(await screen.findByRole('heading', { name: 'Cobertura', level: 1 })).toBeInTheDocument()
  })

  it('el menú móvil se abre, navega y se cierra', async () => {
    const user = userEvent.setup()
    await renderApp('/')
    const boton = screen.getByRole('button', { name: 'Menú' })
    expect(boton).toHaveAttribute('aria-expanded', 'false')
    await user.click(boton)
    expect(boton).toHaveAttribute('aria-expanded', 'true')
    const menu = document.getElementById('menu-movil')!
    await user.click(within(menu).getByRole('link', { name: 'Indicadores' }))
    expect(await screen.findByRole('heading', { name: 'Indicadores', level: 1 })).toBeInTheDocument()
    expect(boton).toHaveAttribute('aria-expanded', 'false')
  })

  it('Escape cierra el menú móvil', async () => {
    const user = userEvent.setup()
    await renderApp('/')
    const boton = screen.getByRole('button', { name: 'Menú' })
    await user.click(boton)
    await user.keyboard('{Escape}')
    expect(boton).toHaveAttribute('aria-expanded', 'false')
  })
})

describe('tema', () => {
  it('cambia a oscuro, aplica la clase y recuerda la preferencia', async () => {
    const user = userEvent.setup()
    await renderApp('/')
    await user.selectOptions(screen.getByLabelText('Tema'), 'oscuro')
    expect(document.documentElement).toHaveClass('dark')
    expect(localStorage.getItem(CLAVE_TEMA)).toBe('oscuro')
    await user.selectOptions(screen.getByLabelText('Tema'), 'claro')
    expect(document.documentElement).not.toHaveClass('dark')
  })
})
