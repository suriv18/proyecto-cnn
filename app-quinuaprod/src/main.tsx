import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '@/app/styles/index.css'
import App from '@/app/App'

async function iniciar() {
  // La comparación es literal a propósito: Vite la resuelve al compilar y elimina los mocks del build real.
  if (import.meta.env.VITE_USE_MOCKS === 'true') {
    const { worker } = await import('@/testing/mocks/browser')
    await worker.start({ onUnhandledFrame: 'bypass' })
  }
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <App />
    </StrictMode>,
  )
}

void iniciar()
