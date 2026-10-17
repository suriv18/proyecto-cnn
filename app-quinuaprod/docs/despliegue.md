# Despliegue

app-quinuaprod son dos piezas independientes:

- **Interfaz:** archivos estáticos (`app-quinuaprod/dist`). Se sirve desde cualquier hosting de estáticos.
- **Backend:** un servicio FastAPI de solo lectura (`service-cnn/src/api`), empaquetado en una imagen de Docker.

Ninguna pieza escribe datos, y no hay autenticación ni datos personales (RNF5). Sirven resultados de investigación del piloto, que se declaran como vinculantes o no en cada respuesta.

## Estado de esta guía

| Qué | Estado |
|---|---|
| Código del backend, pruebas y CI | Probado |
| Lista de archivos de la imagen (`COPY` y `.dockerignore`) | Probada: se replicó en un directorio y se ejecutó solo con `pandas`, `pyyaml`, `fastapi` y `uvicorn` (sin `torch` ni `shap`); todos los endpoints respondieron 200 |
| **Construcción de la imagen con `docker build`** | **No probada**: en el equipo de desarrollo el servicio de Docker estaba apagado |
| Hosting de la interfaz y proxy inverso | No probados: solo se describen aquí |

Antes de publicar, construye la imagen y ejecuta el contenedor (ver abajo) y comprueba `GET /salud`.

## Backend

### Requisitos

Los artefactos no están en git: se generan con el pipeline y deben existir **antes** de construir la imagen.

| Artefacto | Cómo se genera |
|---|---|
| `service-cnn/reports/piloto_oficial/`, `he2_tecnico/`, `sensibilidades/` y `E1_matriz_cobertura_enmendado_2016_2025.json` | `scripts/run_pilot.py`, `run_he2_tecnico.py`, `run_sensibilidades.py` |
| `service-cnn/data/processed/exclusiones.csv` y `metadata.json` | `scripts/run_preprocessing.py` |
| `service-cnn/models/cnn1d_piloto/` | `scripts/export_modelo.py` o `dvc repro -s serializar_modelo` |

### Construir y ejecutar

Desde la **raíz del repositorio** (el contexto necesita el contrato de la interfaz):

```sh
docker build -f service-cnn/Dockerfile -t quinuaprod-api .
docker run --rm -p 8000:8000 -e QUINUAPROD_CORS_ORIGINS=https://mi-dominio.example quinuaprod-api
curl http://localhost:8000/salud
```

- La imagen no instala `torch`, `shap` ni `xgboost`: la inferencia y SHAP corren en NumPy.
- Corre como usuario sin privilegios y tiene un `HEALTHCHECK` sobre `GET /salud`.
- `.dockerignore` permite solo los 20 archivos que copia el `Dockerfile`; así no se envían al daemon el `.venv`, `node_modules` ni los datos crudos.
- Si falta un artefacto, el `docker build` falla en el `COPY` correspondiente. Si un dato falta en ejecución, la API responde `503` con instrucciones.

### Variables de entorno

| Variable | Para qué | Por defecto |
|---|---|---|
| `QUINUAPROD_CORS_ORIGINS` | Orígenes que pueden llamar a la API, separados por comas. **Configúrala con el dominio real de la interfaz.** | `http://localhost:5173,http://localhost:4173` |
| `QUINUAPROD_RAIZ` | Directorio con `reports/`, `data/` y `configs/` | La raíz de `service-cnn` dentro de la imagen |

No hay secretos: la API no usa claves ni credenciales.

## Interfaz

```sh
cd app-quinuaprod
npm ci
VITE_API_URL=https://api.mi-dominio.example/api/v1 npm run build   # en PowerShell: $env:VITE_API_URL="..."; npm run build
```

- El resultado queda en `dist/`. **No incluye los mocks** (MSW ni las fixtures): solo los empaqueta `--mode mocks`.
- `VITE_API_URL` se fija al construir. Si omites la variable, la interfaz llama a `/api/v1` en su propio origen: sirve entonces la interfaz y la API bajo el mismo dominio, con el proxy inverso enviando `/api/v1` al backend. Así no hace falta CORS.
- Las variables `VITE_*` son públicas y quedan dentro del código; no pongas secretos.

El servidor de estáticos debe:

1. **Devolver `index.html` para cualquier ruta que no sea un archivo** (las pantallas usan rutas como `/cobertura`; un enlace directo daría 404 sin esta regla).
2. Servir `/assets/*` con caché larga: los nombres llevan huella y cambian en cada versión. `index.html` no debe cachearse.
3. Opcional pero recomendable: `X-Content-Type-Options: nosniff` y `Referrer-Policy: same-origin`.

## Qué queda expuesto

- Resultados del piloto sobre campañas de entrenamiento y prueba. **No** se sirven las campañas externas: el servidor las descarta con la lista del piloto y una prueba lo verifica.
- `GET /docs` y `GET /openapi.json` publican el contrato. Si no quieres exponerlos, no copies `app-quinuaprod/api/openapi.yaml` en la imagen (FastAPI usará el esquema generado a partir del código).
- La API no limita el uso. Si se publica en Internet abierta, conviene un límite de tasa en el proxy inverso.

## Actualizar los resultados

1. Ejecuta el pipeline y vuelve a exportar el modelo.
2. Reexporta los fixtures si cambiaron los resultados (`npm run export:fixtures`) y revisa las pruebas que dependen de conteos exactos.
3. Reconstruye la imagen y la interfaz.

Al congelar E5 y evaluar las campañas externas, esta guía deja de ser suficiente: hay que decidir qué se publica de esos resultados, con qué rotulado y desde cuándo (ver la enmienda de alcance).
