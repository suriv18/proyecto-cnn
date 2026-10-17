import { expect, test, type Page } from '@playwright/test'

const PANTALLAS = [
  { ruta: '/', enlace: 'Resumen', titulo: 'Resumen' },
  { ruta: '/cobertura', enlace: 'Cobertura', titulo: 'Cobertura' },
  { ruta: '/predicciones', enlace: 'Predicción vs observado', titulo: 'Predicción vs observado' },
  { ruta: '/indicadores', enlace: 'Indicadores', titulo: 'Indicadores' },
  { ruta: '/explicabilidad', enlace: 'Explicabilidad', titulo: 'Explicabilidad' },
  { ruta: '/sensibilidades', enlace: 'Sensibilidades', titulo: 'Sensibilidades' },
  { ruta: '/preespecificacion', enlace: 'Registro E5', titulo: 'Registro E5' },
  { ruta: '/inferencia', enlace: 'Predicción interactiva', titulo: 'Predicción interactiva' },
]

// El menú lateral fijo aparece desde 1024 px (lg); por debajo (tablet y móvil) hay un botón "Menú".
const esEscritorio = (proyecto: string) => proyecto === 'desktop'

async function abrirMenu(page: Page, proyecto: string) {
  if (!esEscritorio(proyecto)) await page.getByRole('button', { name: 'Menú' }).click()
  return page.getByRole('navigation', { name: 'Principal' })
}

const sinScrollHorizontal = (page: Page) =>
  page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)

test.describe('navegación responsive', () => {
  test('el menú es lateral en escritorio y desplegable en tablet y móvil', async ({ page }, info) => {
    await page.goto('/')
    const boton = page.getByRole('button', { name: 'Menú' })
    if (esEscritorio(info.project.name)) {
      await expect(page.getByRole('navigation', { name: 'Principal' })).toBeVisible()
      await expect(boton).toBeHidden()
    } else {
      await expect(page.getByRole('navigation', { name: 'Principal' })).toBeHidden()
      await expect(boton).toBeVisible()
      await boton.click()
      await expect(page.getByRole('navigation', { name: 'Principal' })).toBeVisible()
      await page.keyboard.press('Escape')
      await expect(page.getByRole('navigation', { name: 'Principal' })).toBeHidden()
    }
  })

  test('se llega a cada pantalla desde el menú y ninguna desborda horizontalmente', async ({ page }, info) => {
    await page.goto('/')
    for (const p of PANTALLAS) {
      const nav = await abrirMenu(page, info.project.name)
      await nav.getByRole('link', { name: p.enlace, exact: true }).click()
      await expect(page).toHaveURL(new RegExp(`${p.ruta === '/' ? '/$' : p.ruta + '$'}`))
      await expect(page.getByRole('heading', { name: p.titulo, level: 1 })).toBeVisible()
      expect(await sinScrollHorizontal(page), `desborda en ${p.ruta}`).toBe(true)
    }
  })

  test('un enlace directo a una pantalla carga sin pasar por el resumen', async ({ page }) => {
    await page.goto('/sensibilidades')
    await expect(page.getByRole('heading', { name: 'Sensibilidades', level: 1 })).toBeVisible()
  })

  test('una ruta inexistente muestra no encontrada y permite volver', async ({ page }) => {
    await page.goto('/no-existe')
    await expect(page.getByRole('heading', { name: 'Página no encontrada' })).toBeVisible()
    await page.getByRole('link', { name: 'Volver al resumen' }).click()
    await expect(page.getByRole('heading', { name: 'Resumen', level: 1 })).toBeVisible()
  })

  test('el enlace "Saltar al contenido" recibe el foco con el teclado', async ({ page }) => {
    await page.goto('/')
    await expect(page.getByRole('heading', { name: 'Resumen', level: 1 })).toBeVisible()
    await page.keyboard.press('Tab')
    await expect(page.getByRole('link', { name: 'Saltar al contenido' })).toBeFocused()
  })
})

test.describe('tema', () => {
  test('el tema oscuro se aplica y se recuerda al recargar', async ({ page }) => {
    await page.goto('/')
    await page.getByLabel('Tema').selectOption('oscuro')
    await expect(page.locator('html')).toHaveClass(/dark/)
    await page.reload()
    await expect(page.locator('html')).toHaveClass(/dark/)
    await page.getByLabel('Tema').selectOption('claro')
    await expect(page.locator('html')).not.toHaveClass(/dark/)
  })

  test('con tema "sistema" sigue la preferencia del navegador', async ({ page }) => {
    await page.emulateMedia({ colorScheme: 'dark' })
    await page.goto('/')
    await expect(page.locator('html')).toHaveClass(/dark/)
  })
})
