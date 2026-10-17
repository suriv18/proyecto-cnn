/// <reference types="vitest/config" />
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { '@': '/src' } },
  test: {
    environment: 'jsdom',
    env: { VITE_API_URL: 'http://localhost:3000/api/v1' },
    include: ['src/**/*.test.{ts,tsx}'],
    globals: true,
    testTimeout: 20_000,
    setupFiles: ['./src/testing/setup.ts'],
  },
})
