# app-quinuaprod

Visor React + TypeScript + Vite de los resultados de la tesis CNN-1D + SHAP para rendimiento de quinua.

## Desarrollo

Desde esta carpeta:

```sh
npm ci
npm run dev:mocks     # sin backend: MSW con las instantáneas reales del piloto
npm run dev           # con el backend real en http://localhost:8000 (proxy de /api)
```

`npm run dev:mocks` activa MSW y no necesita backend. `npm run dev` reenvía `/api` a `http://localhost:8000`; si el backend corre en otra dirección, define `QUINUAPROD_API_PROXY` antes de arrancar. Para llamar a una API en otro origen, configura `.env.local` a partir de `.env.example` y ajusta `VITE_API_URL` (la API debe permitir ese origen por CORS). Las variables `VITE_*` son públicas y no deben contener secretos.

## Estructura

```text
src/
  app/                         # Composición: rutas, layout, proveedores, tema
    App.tsx, routes.tsx        # Router; pantallas cargadas bajo demanda
    App.test.tsx               # Pruebas del shell (rutas, menú, tema)
    ResumenPage.tsx            # Compone piloto + cobertura: las funcionalidades no se importan entre sí
    layout/                    # AppLayout, menú responsive, aviso del piloto, selector de tema
    providers.tsx              # TanStack Query con caché por montaje
    styles/index.css           # Tailwind v4 y tema claro/oscuro
  components/                  # Compartidos: Tabla ordenable, Graficos, Campo, EstadoConsulta, PageHeader
  config/                      # env.ts y navegacion.ts
  features/                    # Una carpeta por pantalla, con su api/ de consultas
    cobertura/  predicciones/  indicadores/  explicabilidad/
    sensibilidades/  preespecificacion/  inferencia/  piloto/
    pantallas.test.tsx         # Pruebas de componentes de las pantallas
  lib/
    api/                       # client.ts, errors.ts (ApiError/unwrap), useEstadoPiloto.ts, schema.d.ts (generado)
    format.ts, modelos.ts, predicciones.ts, estadistica.ts, theme.ts
  testing/                     # setup.ts, render.tsx y mocks/ (MSW + fixtures reales de service-cnn)
  main.tsx                     # Inicio de mocks y montaje de React
api/openapi.yaml               # Contrato fuente de la API
scripts/export_fixtures.py     # Exportación de datos reales
public/                        # Archivos servidos sin transformación
e2e/                           # Playwright: smoke, navegacion, pantallas, accesibilidad
docs/                          # Integración con el backend y enmienda de alcance
```

### Reglas para ampliar el frontend

- `app` compone funcionalidades y configura proveedores; las consultas del dominio pertenecen a `features/<dominio>/api`.
- Cada funcionalidad reúne sus componentes, consultas y pruebas. Crea subcarpetas solo cuando tengan contenido.
- `lib` y `config` no importan desde `features` ni desde `app`. Las funcionalidades tampoco importan `app` ni otras funcionalidades; compónlas en `app`.
- Usa `@/` para imports entre áreas y rutas relativas dentro de un módulo. Importa archivos concretos, sin índices que reexporten todo.
- Conserva las pruebas junto al código que verifican. `testing` contiene infraestructura compartida, y `e2e` los flujos de navegador.
- El único acceso de arranque a `testing` es la importación dinámica del worker cuando `VITE_USE_MOCKS=true`. El servidor Node queda reservado a los tests.
- Cuando existan varias pantallas, incorpora rutas en `app`; cuando un componente sea compartido de verdad, extráelo a `src/components`. No se crean rutas, stores ni carpetas vacías por anticipado.
- No edites `schema.d.ts` ni `public/mockServiceWorker.js` manualmente. Mantén las fixtures como datos exportados, sin inventar resultados o campañas externas.

Estas reglas son convenciones de arquitectura; el linter actual no impone fronteras entre módulos.

## Referencias y criterio

Revisión web realizada el 17 de octubre de 2026:

- [Bulletproof React: estructura del proyecto](https://github.com/alan2207/bulletproof-react/blob/master/docs/project-structure.md), aplicación de referencia: agrupa por funcionalidades, separa la composición y proveedores, coloca infraestructura de pruebas en `testing` y recomienda imports directos y dependencias unidireccionales.
- [Grafana: arquitectura frontend](https://github.com/grafana/grafana/blob/main/AGENTS.md#frontend-publicapp), producto real con React/TypeScript: separa dominios en `features` de servicios y componentes compartidos en `core`.

La adaptación local conserva `lib/api` como infraestructura compartida y crea únicamente el dominio ya implementado, `piloto`. El tamaño actual no requiere reproducir el monorepo, los plugins ni los stores de Grafana. El contrato OpenAPI y sus mocks agrupan más endpoints que la interfaz actual; se mantienen centralizados hasta que existan funcionalidades que los consuman.

## Calidad

- **Unitarias y de componentes (Vitest):** lógica pura, contrato de los mocks y las pantallas con datos reales.
- **E2E (Playwright):** desktop (1440 px), tablet (820 px) y móvil (Pixel 7): navegación, pantallas y errores de consola.
- **Accesibilidad (axe, WCAG 2.1 AA):** todas las pantallas en tema claro y oscuro, y el menú móvil abierto.
- **Datos:** ninguna pantalla muestra campañas externas 2021-2025; una prueba e2e lo verifica.

Integración con el backend: [docs/integracion-backend.md](docs/integracion-backend.md). Despliegue: [docs/despliegue.md](docs/despliegue.md). Cambio de alcance respecto de la tesis: [docs/enmienda-alcance-ui.md](docs/enmienda-alcance-ui.md).

## Validación y generación

```sh
npm run lint
npm run typecheck
npm test
npm run build
npm run test:e2e
npm run gen:api
npm run export:fixtures
```

Playwright construye en modo `mocks` y verifica escritorio, tablet y móvil. Requiere Chromium instalado mediante `npx playwright install chromium`. `export:fixtures` utiliza el entorno Python de `service-cnn` y escribe en `src/testing/mocks/fixtures`. `gen:api` regenera los tipos en `src/lib/api/schema.d.ts`.

En PowerShell, si la política de ejecución bloquea `npm.ps1`, usa `npm.cmd` con los mismos argumentos.
