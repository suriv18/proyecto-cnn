import { fireEvent, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import provincias from '@/testing/mocks/fixtures/provincias.json'
import { renderApp } from '@/testing/render'

const filasDe = (tabla: HTMLElement) => within(tabla).getAllByRole('row').slice(1)

describe('Resumen', () => {
  it('muestra el embudo real con N0 = 800 y N5 sin calcular', async () => {
    await renderApp('/')
    const tabla = await screen.findByRole('region', { name: 'Embudo de depuración' })
    const filas = filasDe(tabla)
    expect(filas).toHaveLength(6)
    expect(filas[0]).toHaveTextContent('N0')
    expect(filas[0]).toHaveTextContent('800')
    expect(filas[5]).toHaveTextContent('Sin calcular')
  })
})

describe('Cobertura', () => {
  it('lista las 80 provincias y filtra por clasificación', async () => {
    const user = userEvent.setup()
    await renderApp('/cobertura')
    const tabla = await screen.findByRole('region', { name: 'Matriz provincia por campaña' })
    expect(filasDe(tabla)).toHaveLength(provincias.length)
    await user.selectOptions(screen.getByLabelText('Clasificación'), 'altoandina')
    const altoandinas = provincias.filter((p) => p.clasificacion === 'altoandina').length
    expect(filasDe(tabla)).toHaveLength(altoandinas)
    expect(screen.getByRole('status')).toHaveTextContent(`${altoandinas} de ${provincias.length}`)
  })

  it('marca las exclusiones con su motivo y solo muestra campañas permitidas', async () => {
    await renderApp('/cobertura')
    const tabla = await screen.findByRole('region', { name: 'Matriz provincia por campaña' })
    expect(within(tabla).getAllByLabelText(/^Excluida:/).length).toBeGreaterThan(0)
    const cabeceras = within(tabla).getAllByRole('columnheader').map((c) => c.textContent)
    expect(cabeceras).toEqual(expect.arrayContaining(['2016', '2020']))
    expect(cabeceras.join(' ')).not.toMatch(/202[1-5]/)
  })

  it('ordena por provincia al pulsar la cabecera', async () => {
    const user = userEvent.setup()
    await renderApp('/cobertura')
    const tabla = await screen.findByRole('region', { name: 'Matriz provincia por campaña' })
    await user.click(within(tabla).getByRole('button', { name: /Provincia/ }))
    const primera = filasDe(tabla)[0]
    const esperada = [...provincias.map((p) => p.provincia_id)].sort((a, b) => a.localeCompare(b))[0]
    expect(primera).toHaveTextContent(esperada)
    expect(within(tabla).getByRole('columnheader', { name: /Provincia/ })).toHaveAttribute('aria-sort', 'ascending')
  })
})

describe('Predicción vs observado', () => {
  it('muestra la campaña 2019 por defecto con observado y modelos por provincia', async () => {
    await renderApp('/predicciones')
    const tabla = await screen.findByRole('region', { name: /Predicción por provincia, campaña 2019/ })
    const abancay = filasDe(tabla).find((f) => f.textContent?.includes('ABANCAY'))
    expect(abancay).toBeDefined()
    expect(abancay).toHaveTextContent(/2\D?169/)
    expect(within(tabla).getByRole('columnheader', { name: /^CNN-1D/ })).toBeInTheDocument()
    expect(within(tabla).getByRole('columnheader', { name: /Error abs\. CNN-1D/ })).toBeInTheDocument()
  })

  it('cambia de campaña y de modelo del gráfico', async () => {
    const user = userEvent.setup()
    await renderApp('/predicciones')
    await screen.findByRole('region', { name: /campaña 2019/ })
    await user.selectOptions(screen.getByLabelText('Campaña de prueba'), '2020')
    expect(await screen.findByRole('region', { name: /campaña 2020/ })).toBeInTheDocument()
    await user.selectOptions(screen.getByLabelText('Modelo del gráfico'), 'B2')
    expect(await screen.findByRole('columnheader', { name: /Error abs\. B2/ })).toBeInTheDocument()
    expect(screen.getByText(/B1-B4 son líneas base/)).toBeInTheDocument()
  })
})

describe('Indicadores', () => {
  it('muestra Y1 por modelo, X1 y X2 con valores reales', async () => {
    await renderApp('/indicadores')
    const y1 = await screen.findByRole('region', { name: 'Indicadores de exactitud Y1' })
    expect(within(y1).getAllByText('CNN-1D').length).toBe(2)
    expect(within(y1).getAllByText(/línea base/).length).toBeGreaterThan(0)
    const x1 = await screen.findByLabelText('Indicadores X1')
    expect(x1).toHaveTextContent('5 pasos')
    expect(x1).toHaveTextContent(/100[.,]0 %/)
    const x2 = await screen.findByLabelText('Indicadores X2')
    expect(x2).toHaveTextContent('30 de 58')
    expect(x2).toHaveTextContent('NDVI máximo')
  })
})

describe('Explicabilidad', () => {
  it('advierte que SHAP no es causal y muestra la fase de mayor contribución', async () => {
    await renderApp('/explicabilidad')
    expect(await screen.findByRole('note')).toHaveTextContent(/no relaciones causales/)
    expect(await screen.findByText('Llenado de grano', { selector: 'strong' })).toBeInTheDocument()
    const fases = await screen.findByRole('region', { name: 'Contribución por fase' })
    expect(filasDe(fases)).toHaveLength(5)
  })

  it('muestra HE2 técnico como no vinculante con estabilidad y tablas por campaña', async () => {
    await renderApp('/explicabilidad')
    expect(await screen.findByText(/Ejecución técnica, no vinculante/)).toBeInTheDocument()
    expect(await screen.findByLabelText('Estabilidad de HE2')).toHaveTextContent('W de Kendall')
    const aumento = await screen.findByRole('region', { name: 'Aumento del RMSE por oclusión de fase' })
    expect(within(aumento).getAllByRole('columnheader').map((c) => c.textContent?.replace(/[▲▼]/g, ''))).toEqual(['Fase', '2018', '2019'])
  })

  it('ordena los predictores de mayor a menor importancia', async () => {
    await renderApp('/explicabilidad')
    const tabla = await screen.findByRole('region', { name: 'Importancia por predictor' })
    expect(filasDe(tabla)[0]).toHaveTextContent('NDVI máximo')
    expect(filasDe(tabla)).toHaveLength(6)
  })
})

describe('Sensibilidades', () => {
  it('filtra por variante y por modelo', async () => {
    const user = userEvent.setup()
    await renderApp('/sensibilidades')
    const tabla = await screen.findByRole('region', { name: 'RMSE por variante, modelo y campaña' })
    await waitFor(() => expect(filasDe(tabla).length).toBeGreaterThan(0))
    // Por defecto: modelo CNN-1D en las 6 variantes × 2 campañas.
    expect(filasDe(tabla)).toHaveLength(12)
    await user.selectOptions(screen.getByLabelText('Variante'), 'sin_transicion')
    await waitFor(() => expect(filasDe(tabla)).toHaveLength(2))
    expect(filasDe(tabla)[0]).toHaveTextContent('Sin transición')
  })

  it('muestra el error por estrato', async () => {
    await renderApp('/sensibilidades')
    const tabla = await screen.findByRole('region', { name: 'Error por estrato de proporción de quinua' })
    expect(within(tabla).getAllByText('Bajo').length).toBeGreaterThan(0)
    expect(within(tabla).getAllByText('Alto').length).toBeGreaterThan(0)
  })
})

describe('Registro E5', () => {
  it('muestra los parámetros y la cuenta por estado, sin pendientes', async () => {
    await renderApp('/preespecificacion')
    const estados = await screen.findByRole('list', { name: 'Parámetros por estado' })
    expect(estados).toHaveTextContent('Fijado por tesis')
    expect(estados).not.toHaveTextContent('Propuesto')
    const meta = await screen.findByLabelText('Metadatos de E5')
    expect(meta).toHaveTextContent('Congelado v1.0')
    expect(meta).toHaveTextContent('2026-10-17')
    expect(meta).toHaveTextContent('47c43a7')
    expect(meta).toHaveTextContent('3a44df55')
  })

  it('busca por texto y filtra por estado', async () => {
    const user = userEvent.setup()
    await renderApp('/preespecificacion')
    const tabla = await screen.findByRole('region', { name: 'Parámetros preespecificados' })
    const total = filasDe(tabla).length
    await user.type(screen.getByLabelText('Buscar'), 'bootstrap')
    await waitFor(() => expect(filasDe(tabla).length).toBeLessThan(total))
    expect(filasDe(tabla).length).toBeGreaterThan(0)
    await user.clear(screen.getByLabelText('Buscar'))
    await user.selectOptions(screen.getByLabelText('Estado'), 'fijado')
    await waitFor(() => expect(filasDe(tabla).every((f) => /Fijado(?! por)/.test(f.textContent ?? ''))).toBe(true))
  })
})

describe('Predicción interactiva', () => {
  it('devuelve la predicción real del piloto para una provincia', async () => {
    const user = userEvent.setup()
    await renderApp('/inferencia')
    const form = await screen.findByRole('form', { name: 'Inferencia sobre una observación del piloto' })
    await waitFor(() => expect(within(form).getByLabelText('Provincia')).toHaveValue('ABANCAY'))
    await user.click(within(form).getByRole('button', { name: 'Predecir' }))
    const resultado = await screen.findByRole('region', { name: 'Resultado de la inferencia' })
    expect(resultado).toHaveTextContent('CNN-1D (predicho)')
    expect(resultado).toHaveTextContent(/2\D?169 kg\/ha/)
    expect(resultado).toHaveTextContent(/no vinculante|vinculante/)
    expect(resultado).toHaveTextContent('SHAP por celda no está disponible')
  })

  it('el tensor vacío no se envía y pide completar los valores', async () => {
    const user = userEvent.setup()
    await renderApp('/inferencia')
    const form = await screen.findByRole('form', { name: 'Inferencia sobre un tensor editado' })
    await waitFor(() => expect(within(form).getAllByRole('spinbutton').length).toBe(32))
    await user.click(within(form).getByRole('button', { name: 'Predecir con el tensor' }))
    expect(await within(form).findByRole('alert')).toHaveTextContent(/Complete todos los valores/)
    expect(within(form).queryByRole('region', { name: 'Resultado de la inferencia' })).not.toBeInTheDocument()
  })

  it('un tensor completo, con provincia y campaña, recibe «No implementado» de los datos de demostración', async () => {
    const user = userEvent.setup()
    await renderApp('/inferencia')
    const form = await screen.findByRole('form', { name: 'Inferencia sobre un tensor editado' })
    await waitFor(() => expect(within(form).getAllByRole('spinbutton').length).toBe(32))
    await waitFor(() => expect(within(form).getByLabelText('Provincia')).not.toHaveValue(''))
    expect(within(form).getByLabelText('Campaña')).toHaveValue('2020')
    for (const campo of within(form).getAllByRole('spinbutton')) fireEvent.change(campo, { target: { value: '1' } })
    await user.click(within(form).getByRole('button', { name: 'Predecir con el tensor' }))
    const alerta = await within(form).findByRole('alert')
    expect(alerta).toHaveTextContent('No implementado')
    expect(alerta).toHaveTextContent(/modelo CNN-1D/)
  })
})

describe('Hipótesis y objetivos', () => {
  it('declara que ninguna hipótesis está evaluada', async () => {
    await renderApp('/hipotesis')
    const aviso = (await screen.findAllByRole('note')).find((n) => n.textContent?.includes('Ninguna hipótesis está evaluada'))
    expect(aviso).toHaveTextContent(/campañas externas siguen protegidas/)
  })

  it('muestra por defecto la redacción de la tesis v3 en tres niveles', async () => {
    await renderApp('/hipotesis')
    expect(await screen.findByRole('article', { name: 'Nivel general' })).toHaveTextContent('mejor línea base provincial')
    expect(screen.getByRole('article', { name: 'Nivel específico 1' })).toHaveTextContent('¿En qué medida el modelado temporal')
    expect(screen.getByRole('article', { name: 'Nivel específico 2' })).toBeInTheDocument()
    expect(screen.queryByText('Definiciones')).not.toBeInTheDocument()
  })

  it('cambia a la propuesta de coherencia y muestra las definiciones', async () => {
    const user = userEvent.setup()
    await renderApp('/hipotesis')
    await screen.findByRole('article', { name: 'Nivel general' })
    await user.click(screen.getByRole('radio', { name: 'Propuesta de coherencia' }))
    expect(screen.getByRole('article', { name: 'Nivel específico 1' })).toHaveTextContent('Evaluar si la exactitud predictiva')
    expect(screen.getByRole('article', { name: 'Nivel general' })).toHaveTextContent('B*')
    expect(screen.getByRole('heading', { name: 'Definiciones' })).toBeInTheDocument()
  })

  it('HE1: toma δ de E5 y marca el margen con símbolo y texto, no solo con color', async () => {
    await renderApp('/hipotesis')
    expect(await screen.findByText(/Margen δ \(según E5\)/)).toBeInTheDocument()
    const tabla = await screen.findByRole('region', { name: 'Evidencia descriptiva de HE1 en el piloto' })
    const filas = filasDe(tabla)
    expect(filas).toHaveLength(2)
    expect(filas[1]).toHaveTextContent('2020')
    expect(filas[1]).toHaveTextContent(/\+14[.,]3 %, fuera del margen/)
    expect(filas[0]).toHaveTextContent(/-5[.,]7 %, dentro del margen/)
  })

  it('HE2: la fase máxima y su concordancia salen de los datos y del conjunto crítico de E5', async () => {
    await renderApp('/hipotesis')
    const tabla = await screen.findByRole('region', { name: 'Evidencia descriptiva de HE2c en el piloto' })
    const filas = filasDe(tabla)
    expect(filas).toHaveLength(2)
    expect(filas[0]).toHaveTextContent('2018')
    expect(filas[0]).toHaveTextContent('Desarrollo vegetativo')
    expect(filas[0]).toHaveTextContent('No')
    expect(filas[1]).toHaveTextContent('Floración')
    expect(filas[1]).toHaveTextContent('Sí')
    expect(within(tabla).getByRole('columnheader', { name: /floración o llenado de grano/i })).toBeInTheDocument()
  })

  it('la hipótesis general figura como conjuntiva y sin evaluar', async () => {
    await renderApp('/hipotesis')
    expect(await screen.findByText('Sin evaluar')).toBeInTheDocument()
    expect(screen.getByText('HE1a')).toBeInTheDocument()
    expect(screen.getByText('HE2c')).toBeInTheDocument()
  })
})
