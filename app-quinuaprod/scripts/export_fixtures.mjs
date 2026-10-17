// Ejecuta scripts/export_fixtures.py con el Python del entorno de service-cnn, en Windows y en Linux/macOS.
import { spawnSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { join, resolve } from 'node:path'

const raiz = resolve(import.meta.dirname, '..', '..', 'service-cnn')
const python = [join(raiz, '.venv', 'Scripts', 'python.exe'), join(raiz, '.venv', 'bin', 'python')].find(existsSync)

if (!python) {
  console.error(`No se encontró el entorno de Python en ${join(raiz, '.venv')}. Créalo con «uv venv» e instálalo (ver docs/integracion-backend.md).`)
  process.exit(1)
}

const resultado = spawnSync(python, [join(import.meta.dirname, 'export_fixtures.py')], {
  stdio: 'inherit',
  env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
})
process.exit(resultado.status ?? 1)
