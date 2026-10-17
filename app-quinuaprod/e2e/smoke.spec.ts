import { expect, test } from '@playwright/test'

test.describe('humo: la app arranca con los mocks de datos reales', () => {
  test('muestra el título y el estado del piloto', async ({ page }) => {
    await page.goto('/')
    await expect(page.getByRole('heading', { name: 'Resumen', level: 1 })).toBeVisible()
    const estado = page.getByLabel('Estado del piloto')
    await expect(estado).toBeVisible()
    await expect(estado).toContainText('2019, 2020')
  })

  test('no expone campañas externas 2021-2025', async ({ page }) => {
    await page.goto('/')
    await expect(page.getByLabel('Estado del piloto')).toBeVisible()
    const texto = await page.locator('body').innerText()
    // Con límite de palabra: el nombre de un archivo de origen (p. ej. ..._2016_2025.json) no es una campaña mostrada.
    expect(texto).not.toMatch(/\b202[1-5]\b/)
  })

  test('el contrato de la API responde desde el navegador', async ({ page }) => {
    await page.goto('/')
    await expect(page.getByLabel('Estado del piloto')).toBeVisible()
    const r = await page.evaluate(async () => {
      const res = await fetch('/api/v1/predicciones?campana_prueba=2021')
      return { status: res.status, tipo: res.headers.get('content-type') }
    })
    expect(r.status).toBe(422)
    expect(r.tipo).toContain('application/problem+json')
  })
})
