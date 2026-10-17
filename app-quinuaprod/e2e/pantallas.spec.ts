import { expect, test, type Page } from '@playwright/test'

const RUTAS = ['/', '/cobertura', '/predicciones', '/indicadores', '/explicabilidad', '/sensibilidades', '/hipotesis', '/preespecificacion', '/inferencia']

const filas = (page: Page, region: string | RegExp) =>
  page.getByRole('region', { name: region }).getByRole('row')

test.describe('transversal', () => {
  test('ninguna pantalla produce errores de consola ni de página', async ({ page }) => {
    const errores: string[] = []
    page.on('console', (m) => m.type() === 'error' && errores.push(m.text()))
    page.on('pageerror', (e) => errores.push(e.message))
    for (const ruta of RUTAS) {
      await page.goto(ruta)
      await expect(page.getByRole('heading', { level: 1 })).toBeVisible()
      await expect(page.getByText('Cargando…')).toHaveCount(0)
    }
    expect(errores).toEqual([])
  })

  test('ninguna pantalla de datos muestra campañas externas (2021-2025)', async ({ page }) => {
    // El registro E5 se excluye: es texto de la preespecificación, no resultados.
    for (const ruta of RUTAS.filter((r) => r !== '/preespecificacion')) {
      await page.goto(ruta)
      await expect(page.getByText('Cargando…')).toHaveCount(0)
      const texto = await page.locator('body').innerText()
      expect(texto, `campaña externa visible en ${ruta}`).not.toMatch(/\b202[1-5]\b/)
    }
  })

  test('las pantallas no desbordan horizontalmente la página', async ({ page }) => {
    for (const ruta of RUTAS) {
      await page.goto(ruta)
      await expect(page.getByText('Cargando…')).toHaveCount(0)
      const ok = await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)
      expect(ok, `desborda en ${ruta}`).toBe(true)
    }
  })
})

test.describe('Resumen', () => {
  test('muestra el embudo y dibuja el gráfico', async ({ page }) => {
    await page.goto('/')
    await expect(filas(page, 'Embudo de depuración')).toHaveCount(7) // cabecera + N0..N5
    await expect(page.getByRole('img', { name: /Observaciones por nivel/ }).locator('svg')).toBeVisible()
  })
})

test.describe('Cobertura', () => {
  test('filtra la matriz por clasificación', async ({ page }) => {
    await page.goto('/cobertura')
    await expect(filas(page, 'Matriz provincia por campaña')).toHaveCount(81)
    await page.getByLabel('Clasificación').selectOption('selva')
    await expect(filas(page, 'Matriz provincia por campaña')).toHaveCount(4) // 3 provincias + cabecera
    await expect(page.getByRole('status').filter({ hasText: 'de 80 provincias' })).toContainText('3 de 80')
  })
})

test.describe('Predicción vs observado', () => {
  test('cambia de campaña y ordena por error', async ({ page }) => {
    await page.goto('/predicciones')
    await expect(page.getByRole('region', { name: /campaña 2019/ })).toBeVisible()
    await expect(page.getByRole('img', { name: /Dispersión/ }).locator('svg')).toBeVisible()
    await page.getByLabel('Campaña de prueba').selectOption('2020')
    await expect(page.getByRole('region', { name: /campaña 2020/ })).toBeVisible()
    await page.getByRole('button', { name: /Error abs\. CNN-1D/ }).click()
    await expect(page.getByRole('columnheader', { name: /Error abs\. CNN-1D/ })).toHaveAttribute('aria-sort', 'ascending')
  })
})

test.describe('Indicadores y explicabilidad', () => {
  test('Indicadores muestra Y1, X1 y X2', async ({ page }) => {
    await page.goto('/indicadores')
    await expect(page.getByRole('region', { name: 'Indicadores de exactitud Y1' })).toBeVisible()
    await expect(page.getByLabel('Indicadores X1')).toContainText('5 pasos')
    await expect(page.getByLabel('Indicadores X2')).toContainText('30 de 58')
  })

  test('Explicabilidad advierte que SHAP no es causal y dibuja los gráficos', async ({ page }) => {
    await page.goto('/explicabilidad')
    await expect(page.getByRole('note').filter({ hasText: 'no relaciones causales' })).toBeVisible()
    await expect(page.getByRole('img', { name: /Contribución por fase/ }).locator('svg')).toBeVisible()
    await expect(page.getByRole('img', { name: /Importancia por predictor/ }).locator('svg')).toBeVisible()
    await expect(page.getByText(/Ejecución técnica, no vinculante/)).toBeVisible()
  })
})

test.describe('Sensibilidades y registro E5', () => {
  test('Sensibilidades filtra por variante', async ({ page }) => {
    await page.goto('/sensibilidades')
    await expect(filas(page, 'RMSE por variante, modelo y campaña')).toHaveCount(13)
    await page.getByRole('combobox', { name: 'Variante' }).selectOption('tensor_completo')
    await expect(filas(page, 'RMSE por variante, modelo y campaña')).toHaveCount(3)
  })

  test('Registro E5 busca por texto', async ({ page }) => {
    await page.goto('/preespecificacion')
    await expect.poll(() => filas(page, 'Parámetros preespecificados').count()).toBeGreaterThan(50)
    const antes = await filas(page, 'Parámetros preespecificados').count()
    await page.getByLabel('Buscar').fill('bootstrap')
    await expect.poll(() => filas(page, 'Parámetros preespecificados').count()).toBeLessThan(antes)
    await expect(page.getByLabel('Metadatos de E5')).toContainText('2026-10-17')
  })
})

test.describe('Hipótesis y objetivos', () => {
  test('muestra la matriz y permite comparar la redacción de la tesis con la propuesta', async ({ page }) => {
    await page.goto('/hipotesis')
    await expect(page.getByRole('article', { name: 'Nivel general' })).toContainText('mejor línea base provincial')
    await page.getByRole('radio', { name: 'Propuesta de coherencia' }).check({ force: true })
    await expect(page.getByRole('article', { name: 'Nivel específico 1' })).toContainText('Evaluar si la exactitud predictiva')
    await expect(page.getByRole('heading', { name: 'Definiciones' })).toBeVisible()
  })

  test('muestra la evidencia descriptiva y avisa que no decide ninguna hipótesis', async ({ page }) => {
    await page.goto('/hipotesis')
    await expect(page.getByRole('note').filter({ hasText: 'Ninguna hipótesis está evaluada' })).toBeVisible()
    await expect(page.getByRole('region', { name: 'Evidencia descriptiva de HE1 en el piloto' })).toContainText('fuera del margen')
    await expect(page.getByRole('region', { name: 'Evidencia descriptiva de HE2c en el piloto' })).toContainText('Desarrollo vegetativo')
  })
})

test.describe('Predicción interactiva', () => {
  test('recupera la predicción del piloto para una provincia', async ({ page }) => {
    await page.goto('/inferencia')
    const form = page.getByRole('form', { name: 'Inferencia sobre una observación del piloto' })
    await expect(form.getByLabel('Provincia')).toHaveValue('ABANCAY')
    await form.getByRole('button', { name: 'Predecir' }).click()
    const resultado = page.getByRole('region', { name: 'Resultado de la inferencia' })
    await expect(resultado).toContainText('CNN-1D (predicho)')
    await expect(resultado).toContainText('SHAP por celda no está disponible')
  })

  test('el tensor editado valida y luego informa que no está implementado', async ({ page }) => {
    await page.goto('/inferencia')
    const form = page.getByRole('form', { name: 'Inferencia sobre un tensor editado' })
    await form.getByRole('button', { name: 'Predecir con el tensor' }).click()
    await expect(form.getByRole('alert')).toContainText('Complete todos los valores')
    const campos = form.getByRole('spinbutton')
    await expect(campos).toHaveCount(32)
    for (let i = 0; i < 32; i++) await campos.nth(i).fill('1')
    await form.getByRole('button', { name: 'Predecir con el tensor' }).click()
    await expect(form.getByRole('alert')).toContainText('No implementado')
  })
})
