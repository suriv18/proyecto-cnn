import AxeBuilder from '@axe-core/playwright'
import { expect, test, type Page } from '@playwright/test'

const RUTAS = ['/', '/cobertura', '/predicciones', '/indicadores', '/explicabilidad', '/sensibilidades', '/hipotesis', '/preespecificacion', '/inferencia']
const ETIQUETAS = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']

async function auditar(page: Page) {
  await expect(page.getByRole('heading', { level: 1 })).toBeVisible()
  await expect(page.getByText('Cargando…')).toHaveCount(0)
  const { violations } = await new AxeBuilder({ page }).withTags(ETIQUETAS).analyze()
  return violations.map((v) => `${v.id} (${v.impact}): ${v.help} → ${v.nodes.map((n) => n.target.join(' ')).slice(0, 3).join(' | ')}`)
}

for (const tema of ['claro', 'oscuro'] as const) {
  test.describe(`accesibilidad WCAG 2.1 AA, tema ${tema}`, () => {
    test.beforeEach(async ({ page }) => {
      await page.addInitScript((t) => localStorage.setItem('quinuaprod-tema', t), tema)
    })

    for (const ruta of RUTAS) {
      test(`sin violaciones en ${ruta}`, async ({ page }) => {
        await page.goto(ruta)
        if (tema === 'oscuro') await expect(page.locator('html')).toHaveClass(/dark/)
        expect(await auditar(page)).toEqual([])
      })
    }
  })
}

test('sin violaciones con el menú móvil abierto', async ({ page }, info) => {
  test.skip(info.project.name === 'desktop', 'El menú desplegable solo existe en tablet y móvil')
  await page.goto('/')
  await page.getByRole('button', { name: 'Menú' }).click()
  expect(await auditar(page)).toEqual([])
})
