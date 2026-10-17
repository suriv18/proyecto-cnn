# Despliegue

app-quinuaprod son dos piezas independientes:

- **Interfaz:** archivos estáticos (`app-quinuaprod/dist`). Se sirve desde cualquier hosting de estáticos.
- **Backend:** un servicio FastAPI de solo lectura (`service-cnn/src/api`), empaquetado en una imagen de Docker.

Ninguna pieza escribe datos y no hay datos personales (RNF5). Por defecto la API es pública, con un límite de consultas por cliente; una clave de API es opcional (ver «Control de acceso»). Sirven resultados de investigación del piloto, que se declaran como vinculantes o no en cada respuesta.

## Estado de esta guía

| Qué | Estado |
|---|---|
| Código del backend, pruebas y CI | Probado |
| Límite de uso y clave de API (`src/api/acceso.py`) | Probado: 18 pruebas, incluidas las cabeceras CORS en los errores 401 y 429 |
| Lista de archivos de la imagen del backend (`COPY` y `.dockerignore`) | Probada: se replicó en un directorio y se ejecutó solo con `pandas`, `pyyaml`, `fastapi` y `uvicorn` (sin `torch` ni `shap`); todos los endpoints respondieron 200 |
| El build de la interfaz no publica los mocks | Probado: sin `--mode mocks` el `dist` no contiene `mockServiceWorker.js` ni el código de MSW |
| **`docker build` del backend y de la interfaz, y `docker-compose.yml`** | **No probados**: en el equipo de desarrollo el servicio de Docker estaba apagado |
| **`nginx.conf.template`** | **No probada**: no se pudo ejecutar nginx al escribirla |

Antes de publicar, construye las imágenes, levanta los contenedores (ver abajo) y comprueba `GET /salud` y que la interfaz cargue y navegue.

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
| `QUINUAPROD_LIMITE_POR_MINUTO` | Consultas por minuto y por cliente; al superarlas, `429` con `Retry-After`. `0` lo desactiva | `300` |
| `QUINUAPROD_CONFIAR_PROXY` | `true` si hay un proxy inverso **de confianza**: el cliente se toma del primer valor de `X-Forwarded-For` | `false` |
| `QUINUAPROD_API_KEY` | Si se define, todas las rutas salvo `/salud` exigen la cabecera `X-API-Key` | Sin definir (API pública) |

Solo `QUINUAPROD_API_KEY` es un secreto: pásalo con el mecanismo de secretos de tu plataforma, no lo escribas en un archivo del repositorio.

### Control de acceso

- **Límite de uso (activo por defecto, 300 por minuto y cliente).** Cada pantalla de la interfaz hace pocas consultas (en el recorrido de las 9 pantallas fueron 29 en total), así que el límite no estorba al uso normal. Se cuenta en memoria y por proceso: con varias réplicas, cada una cuenta por separado. `/salud` no cuenta.
- **Detrás de un proxy.** Sin `QUINUAPROD_CONFIAR_PROXY=true`, todos los clientes llegan con la dirección del proxy y comparten un solo cupo. Con `true`, **el proxy debe fijar** `X-Forwarded-For` con el cliente real y no añadirlo a lo que traiga la solicitud; si no, cualquiera puede falsearla y evadir el límite. La plantilla de nginx de la interfaz lo hace así. No actives `true` si la API es accesible sin pasar por ese proxy.
- **Clave de API (opcional).** Cuando está activa, los intentos con clave errónea también consumen el límite (frena la fuerza bruta) y los errores `401` y `429` llevan cabeceras CORS para que el navegador pueda leerlos.
- **Una clave en la interfaz web no es un secreto.** Cualquiera puede leerla en el código que descarga el navegador. La clave sirve para llamadas entre servidores o para cuando un proxy con autenticación propia (por ejemplo, con inicio de sesión) agrega la cabecera por el usuario. Para una interfaz pública, la defensa real es el límite de uso y CORS.

## Interfaz

```sh
cd app-quinuaprod
npm ci
VITE_API_URL=https://api.mi-dominio.example/api/v1 npm run build   # en PowerShell: $env:VITE_API_URL="..."; npm run build
```

- El resultado queda en `dist/`. **No incluye los mocks**: ni el código de MSW, ni las fixtures, ni `mockServiceWorker.js`; solo los empaqueta `--mode mocks`.
- `VITE_API_URL` se fija al construir. Si omites la variable, la interfaz llama a `/api/v1` en su propio origen: sirve entonces la interfaz y la API bajo el mismo dominio, con el proxy inverso enviando `/api/v1` al backend. Así no hace falta CORS.
- Las variables `VITE_*` son públicas y quedan dentro del código; no pongas secretos.

El servidor de estáticos debe:

1. **Devolver `index.html` para cualquier ruta que no sea un archivo** (las pantallas usan rutas como `/cobertura`; un enlace directo daría 404 sin esta regla).
2. Servir `/assets/*` con caché larga: los nombres llevan huella y cambian en cada versión. `index.html` no debe cachearse.
3. Opcional pero recomendable: `X-Content-Type-Options: nosniff` y `Referrer-Policy: same-origin`.

### Con Docker (no probado)

`app-quinuaprod/Dockerfile` compila la interfaz y la sirve con nginx sin privilegios; `nginx.conf.template` ya cumple las tres reglas anteriores y reenvía `/api/` al backend (`API_UPSTREAM`), de modo que la interfaz y la API comparten origen y no hace falta CORS:

```sh
docker build -t quinuaprod-web app-quinuaprod
docker run --rm -p 8080:8080 -e API_UPSTREAM=http://host.docker.internal:8000 quinuaprod-web
```

O las dos piezas juntas, desde la raíz del repositorio (necesitas los artefactos del backend, ver arriba):

```sh
docker compose up --build        # interfaz en http://localhost:8080
```

Si publicas la interfaz con otra API, pasa `--build-arg VITE_API_URL=https://api.mi-dominio.example/api/v1`. Un valor vacío cuenta como «mismo origen».

## Qué queda expuesto

- Resultados del piloto sobre campañas de entrenamiento y prueba. **No** se sirven las campañas externas: el servidor las descarta con la lista del piloto y una prueba lo verifica.
- `GET /docs` y `GET /openapi.json` publican el contrato. Si no quieres exponerlos, no copies `app-quinuaprod/api/openapi.yaml` en la imagen (FastAPI usará el esquema generado a partir del código).
- La API limita el uso por cliente (300 consultas por minuto por defecto). Si la publicas en Internet abierta, conviene además un límite en el proxy inverso, que protege también a la interfaz.

## Actualizar los resultados

1. Ejecuta el pipeline y vuelve a exportar el modelo.
2. Reexporta los fixtures si cambiaron los resultados (`npm run export:fixtures`) y revisa las pruebas que dependen de conteos exactos.
3. Reconstruye la imagen y la interfaz.

Al congelar E5 y evaluar las campañas externas, esta guía deja de ser suficiente: hay que decidir qué se publica de esos resultados, con qué rotulado y desde cuándo (ver la enmienda de alcance).
