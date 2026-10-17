# Integración con el backend

La UI consume `api/openapi.yaml` (OpenAPI 3.1), que es la fuente de verdad. Hoy responde MSW con una instantánea real de `service-cnn/`; el backend aún no existe. Este documento describe qué falta y cómo hacer el cambio.

## Endpoints y su origen

| Endpoint | Origen en `service-cnn/` | Estado |
|---|---|---|
| `GET /meta` | `reports/piloto_oficial/resultados.json` (p0, p4, p7), `data/processed/metadata.json`, `configs/preespecificacion.yaml` | Por implementar |
| `GET /provincias` | `configs/provincias.csv` | Por implementar |
| `GET /cobertura/embudo` | `reports/E1_matriz_cobertura_enmendado_2016_2025.json` | Por implementar |
| `GET /cobertura/exclusiones` | `data/processed/exclusiones.csv` | Por implementar |
| `GET /predicciones` | `reports/piloto_oficial/predicciones.csv` | Por implementar |
| `GET /indicadores/y1` | `reports/piloto_oficial/p11_indicadores_y1.csv` | Por implementar |
| `GET /indicadores/x1-x2` | `reports/piloto_oficial/resultados.json` (p11) | Por implementar |
| `GET /indicadores/variabilidad-semillas` | `reports/piloto_oficial/p6_variabilidad_por_semilla.csv` | Por implementar |
| `GET /explicabilidad` | `resultados.json` (p8) y `reports/he2_tecnico/resultados.json` | Por implementar |
| `GET /sensibilidades` y `/sensibilidades/estratos` | `reports/sensibilidades/*.csv` | Por implementar |
| `GET /preespecificacion` | `configs/preespecificacion.yaml` | Por implementar |
| `POST /predicciones/inferir` | Predicciones del piloto (por observación) o el CNN-1D serializado (tensor editado) | Parcial: ver abajo |

El mock de `POST /predicciones/inferir` devuelve la mediana entre semillas del CNN-1D y B2 ya calculadas por el piloto, sin SHAP (`shap: null`). Con `tensor` responde `501`.

## Reglas que el backend debe cumplir

1. **Campañas externas.** Los endpoints solo exponen `Meta.campanas_permitidas`. Una campaña fuera de la lista debe responder `422` con `application/problem+json`, como hace el mock. No basta con filtrar en la UI.
2. **Errores.** Siempre RFC 9457 (`type`, `title`, `status`, `detail`). La UI muestra `title` y `detail` tal cual.
3. **`vinculante`.** `Meta.vinculante` y `ResultadoInferencia.vinculante` deben reflejar el estado real; la UI los muestra sin reinterpretarlos.
4. **Valores no calculados.** Usar `null` (por ejemplo `NivelEmbudo.n` de N5, `shap`), nunca `0` ni cadenas vacías.
5. **Paginación.** `GET /predicciones` acepta `limit` (máximo 5000) y `offset`; la UI pide 5000 por campaña de prueba (2160 filas).

## Trabajo pendiente en `service-cnn/`

- Crear `service-cnn/api/` (FastAPI) de solo lectura que valide sus respuestas contra el OpenAPI. Los artefactos de `reports/` están ignorados por git y se regeneran con DVC; el servicio debe leer de ahí o de una instantánea versionada.
- **Declarar `pyarrow`** en `pyproject.toml`: el pipeline lee y escribe parquet y el entorno actual no lo tiene, por lo que `pd.read_parquet` falla.
- **Serializar el CNN-1D** (`torch.save` o ONNX) y versionarlo con DVC. Hoy el pipeline entrena y evalúa al vuelo; sin modelo serializado el tensor editado seguirá respondiendo `501`.
- **SHAP por observación** (matriz 6×5): hoy solo se exportan agregados. Si se quiere mostrar, el pipeline debe guardarlos y el contrato `ResultadoInferencia.shap` ya lo admite.
- CORS para el origen de la UI.

## Cambiar de mocks al backend real

1. Copiar `.env.example` a `.env.local` y poner `VITE_USE_MOCKS=false` y `VITE_API_URL=<url del backend>`.
2. `npm run dev` (sin `--mode mocks`).
3. Si el contrato cambia: editar `api/openapi.yaml`, ejecutar `npm run gen:api` y corregir lo que `npm run typecheck` señale.
4. Si el pipeline produce nuevos resultados: `npm run export:fixtures` y revisar los conteos de las pruebas que dependen de los datos (por ejemplo, 13 filas de Sensibilidades).
5. Para contrastar el backend con la UI, ejecutar contra él los mismos casos que `src/testing/mocks/handlers.test.ts` (filtros, 422 en campañas externas, 501 sin modelo).

El build de producción no incluye MSW ni las fixtures; solo `--mode mocks` los empaqueta.
