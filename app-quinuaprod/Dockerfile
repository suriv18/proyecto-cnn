# Interfaz de app-quinuaprod: se compila con Node y se sirve con nginx sin privilegios.
#
# Se construye desde la carpeta de la interfaz:
#     docker build -t quinuaprod-web app-quinuaprod
#     docker run --rm -p 8080:8080 -e API_UPSTREAM=http://host.docker.internal:8000 quinuaprod-web
#
# Por defecto la interfaz llama a /api/v1 en su propio origen y nginx lo reenvía al backend (API_UPSTREAM),
# así no hace falta CORS. Para llamar a otra URL, fíjala al construir:
#     docker build --build-arg VITE_API_URL=https://api.ejemplo.org/api/v1 -t quinuaprod-web app-quinuaprod
#
# Probado con `docker compose up --build`: la imagen construye y corre como usuario sin privilegios.
FROM node:24-alpine AS compilar
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# Vacío significa «el mismo origen» (src/config/env.ts trata '' como sin definir).
ARG VITE_API_URL=
ENV VITE_API_URL=$VITE_API_URL
# Sin mocks: solo los empaqueta `--mode mocks`.
RUN npm run build

FROM nginxinc/nginx-unprivileged:1.27-alpine
# La imagen sustituye ${API_UPSTREAM} al arrancar (plantillas en /etc/nginx/templates).
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY nginx-seguridad.conf /etc/nginx/snippets/seguridad.conf
COPY --from=compilar /app/dist /usr/share/nginx/html
ENV API_UPSTREAM=http://api:8000
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -q --spider http://127.0.0.1:8080/ || exit 1
