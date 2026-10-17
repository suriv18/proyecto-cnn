/// <reference types="vitest/config" />
import { rmSync } from 'node:fs'
import { join } from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig, type Plugin } from 'vite'

/**
 * Vite copia todo `public/` al build. `mockServiceWorker.js` solo sirve con los mocks (`--mode mocks`); en
 * cualquier otro build se elimina para no publicar un archivo que nunca se registra.
 */
function sinServiceWorkerDeMocks(mode: string): Plugin {
  let salida = 'dist'
  return {
    name: 'sin-service-worker-de-mocks',
    apply: 'build',
    configResolved(config) {
      salida = join(config.root, config.build.outDir)
    },
    closeBundle() {
      if (mode !== 'mocks') rmSync(join(salida, 'mockServiceWorker.js'), { force: true })
    },
  }
}

export default defineConfig(({ mode }) => ({
  plugins: [react(), tailwindcss(), sinServiceWorkerDeMocks(mode)],
  resolve: { alias: { '@': '/src' } },
  test: {
    environment: 'jsdom',
    env: { VITE_API_URL: 'http://localhost:3000/api/v1' },
    include: ['src/**/*.test.{ts,tsx}'],
    globals: true,
    testTimeout: 20_000,
    setupFiles: ['./src/testing/setup.ts'],
  },
}))
