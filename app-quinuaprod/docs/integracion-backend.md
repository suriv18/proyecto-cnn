# Integración con el backend

La UI consume `api/openapi.yaml` (OpenAPI 3.1), que es la fuente de verdad. El backend de solo lectura ya existe en `service-cnn/src/api` (FastAPI) y cumple ese contrato; la interfaz sigue usando por defecto los mocks de MSW con una instantánea de los mismos artefactos. Este documento describe cómo ejecutar el backend, qué falta y cómo cambiar de mocks al servicio real.

## Endpoints y su origen

| Endpoint | Origen en `service-cnn/` | Estado |
|---|---|---|
| `GET /meta` | `reports/piloto_oficial/resultados.json` (p0, p4, p7), `data/processed/metadata.json`, `configs/preespecificacion.yaml` | Implementado |
| `GET /provincias` | `configs/provincias.csv` | Implementado |
| `GET /cobertura/embudo` | `reports/E1_matriz_cobertura_enmendado_2016_2025.json` | Implementado |
| `GET /cobertura/exclusiones` | `data/processed/exclusiones.csv` | Implementado |
| `GET /predicciones` | `reports/piloto_oficial/predicciones.csv` | Implementado |
| `GET /indicadores/y1` | `reports/piloto_oficial/p11_indicadores_y1.csv` | Implementado |
| `GET /indicadores/x1-x2` | `reports/piloto_oficial/resultados.json` (p11) | Implementado |
| `GET /indicadores/variabilidad-semillas` | `reports/piloto_oficial/p6_variabilidad_por_semilla.csv` | Implementado |
| `GET /explicabilidad` | `resultados.json` (p8) y `reports/he2_tecnico/resultados.json` | Implementado |
| `GET /sensibilidades` y `/sensibilidades/estratos` | `reports/sensibilidades/*.csv` | Implementado |
| `GET /preespecificacion` | `configs/preespecificacion.yaml` | Implementado |
| `POST /predicciones/inferir` | Predicciones del piloto (por observación) o el CNN-1D serializado (tensor editado) | Implementado. Con tensor necesita el modelo serializado (ver «Modelo serializado»); sin él responde 501 |

El mock de `POST /predicciones/inferir` devuelve la mediana entre semillas del CNN-1D y B2 ya calculadas por el piloto, sin SHAP (`shap: null`). Con `tensor` valida igual que el backend (provincia y campaña obligatorias, campaña permitida) y responde `501`, porque los mocks no ejecutan el modelo y no inventan predicciones.

## Reglas que el backend debe cumplir

1. **Campañas externas.** Los endpoints solo exponen `Meta.campanas_permitidas`. Una campaña fuera de la lista debe responder `422` con `application/problem+json`, como hace el mock. No basta con filtrar en la UI.
2. **Errores.** Siempre RFC 9457 (`type`, `title`, `status`, `detail`). La UI muestra `title` y `detail` tal cual.
3. **`vinculante`.** `Meta.vinculante` y `ResultadoInferencia.vinculante` deben reflejar el estado real; la UI los muestra sin reinterpretarlos.
4. **Valores no calculados.** Usar `null` (por ejemplo `NivelEmbudo.n` de N5, `shap`), nunca `0` ni cadenas vacías.
5. **Paginación.** `GET /predicciones` acepta `limit` (máximo 5000) y `offset`; la UI pide 5000 por campaña de prueba (2160 filas).

## Ejecutar el backend

Desde `service-cnn/`, con las dependencias instaladas (`uv pip install -e ".[api,dev]"`):

```sh
.venv/Scripts/python.exe -m uvicorn src.api.main:app --port 8000
```

- Lee los artefactos de `reports/`, `data/processed/` y `configs/`. Si faltan, responde `503` con instrucciones.
- `/docs` muestra el contrato de la interfaz (`openapi.yaml`) cuando el monorepo está completo.
- `QUINUAPROD_RAIZ` cambia el directorio de datos y `QUINUAPROD_CORS_ORIGINS` los orígenes permitidos (por defecto, los de Vite: 5173 y 4173).
- `GET /salud` responde sin necesitar datos (útil para el despliegue).
- Pruebas: `.venv/Scripts/python.exe -m pytest tests/api`. Validan cada respuesta contra el YAML, la regla de campañas externas y la paridad con los fixtures de la interfaz.

## Modelo serializado

El tensor editado se predice con la CNN-1D del piloto, guardada en `service-cnn/models/cnn1d_piloto/` (no se sube a git):

```sh
.venv/Scripts/python.exe scripts/export_modelo.py
```

También es la etapa `serializar_modelo` de `dvc.yaml`. Se ejecuta **de forma aislada** para que DVC no relance la ingesta contra Earth Engine: `dvc repro -s serializar_modelo` (con el `.venv` en el `PATH`). `dvc status serializar_modelo` indica si el modelo está al día respecto de los datos, el código y los resultados del piloto.

- **Qué es:** el mismo entrenamiento del último pliego del piloto: campañas 2016-2019, representación en anomalía, hiperparámetros de P7 y semillas 0 a 9. Los toma de `reports/piloto_oficial/resultados.json`.
- **Se verifica solo:** al terminar compara sus predicciones con `predicciones.csv` del piloto y falla si difieren en más de 0,01 kg/ha (diferencia actual: 0,000035).
- **Formato:** `pesos.npz`, `preprocesador.npz`, `fondo.npz` (referencia de SHAP), `lineas_base.json` y `modelo.json` (tarjeta con arquitectura, campañas, huella de los pesos y advertencias). Sin `pickle`; el servicio los lee con NumPy y **no necesita PyTorch**.
- **Cómo predice:** el modelo da la anomalía respecto de la línea base B2. El rendimiento es B2(provincia, campaña) más la mediana de las anomalías de las 10 semillas, como el piloto. Por eso con un tensor la solicitud exige `provincia_id` y `campana`.
- **SHAP:** con un tensor, `shap` trae la matriz de variables por fases en kg/ha de la anomalía, promediada entre las 10 semillas. Se calcula en NumPy (DeepLIFT con regla de reescalado, el algoritmo de `shap.DeepExplainer`) con un fondo de 100 filas del entrenamiento guardado en `fondo.npz`, el mismo que usa el piloto. Reproduce al `shap` de la biblioteca y las contribuciones por fase que el piloto publicó en P8 (diferencia de ~1e-9), y cumple la precisión local. Describe lo que aprendió el modelo, no causalidad; las covariables no están en la matriz. En modo observación `shap` sigue en `null`.
- **Límites:** `y_real` va en `null`; una provincia sin historia de entrenamiento responde `404`; las campañas externas se rechazan con `422`. El resultado declara `vinculante` y `modelo_version` (la huella de los pesos).

## Trabajo pendiente

- **Modelo definitivo:** el modelo serializado es el del piloto, no el de la evaluación confirmatoria. Tras congelar E5 y entrenar con la ventana definitiva hay que volver a exportarlo.
- **Almacenamiento remoto del modelo:** la etapa de DVC ya existe, pero no hay un remoto configurado (`dvc push`); `models/` no se sube a git. Para desplegar, el modelo se genera en el servidor con el script o se copia la carpeta.
- **SHAP en modo observación:** solo se calcula con un tensor. Para mostrarlo al consultar una observación del piloto haría falta leer su tensor de `data/processed` (parquet) desde el servicio.
- **Despliegue:** la API no tiene autenticación ni límites de uso porque solo sirve resultados de investigación sin datos personales. Si se publica, hay que decidir si se restringe el acceso y configurar `QUINUAPROD_CORS_ORIGINS` con el dominio real.
- **`matplotlib`:** `scripts/build_e3_report.py` lo importa y no está declarado; su prueba falla al recolectarse en un entorno limpio.

## Cambiar de mocks al backend real

1. Copiar `.env.example` a `.env.local` y poner `VITE_USE_MOCKS=false` y `VITE_API_URL=<url del backend>`.
2. `npm run dev` (sin `--mode mocks`).
3. Si el contrato cambia: editar `api/openapi.yaml`, ejecutar `npm run gen:api` y corregir lo que `npm run typecheck` señale.
4. Si el pipeline produce nuevos resultados: `npm run export:fixtures` y revisar los conteos de las pruebas que dependen de los datos (por ejemplo, 13 filas de Sensibilidades).
5. Para contrastar el backend con la UI, ejecutar contra él los mismos casos que `src/testing/mocks/handlers.test.ts` (filtros, 422 en campañas externas, 501 sin modelo).

El build de producción no incluye MSW ni las fixtures; solo `--mode mocks` los empaqueta.
