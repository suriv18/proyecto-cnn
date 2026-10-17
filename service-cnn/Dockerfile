# Backend de solo lectura de app-quinuaprod (service-cnn/src/api).
#
# Se construye desde la RAÍZ del repositorio, porque necesita el contrato de la interfaz:
#     docker build -f service-cnn/Dockerfile -t quinuaprod-api .
#     docker run -p 8000:8000 -e QUINUAPROD_CORS_ORIGINS=https://mi-dominio.example quinuaprod-api
#
# Requisitos antes de construir (los artefactos no están en git, se generan con el pipeline):
#   - reports/piloto_oficial, reports/he2_tecnico y reports/sensibilidades (scripts/run_pilot.py, run_he2_tecnico.py,
#     run_sensibilidades.py)
#   - data/processed/exclusiones.csv y metadata.json (scripts/run_preprocessing.py)
#   - models/cnn1d_piloto (scripts/export_modelo.py o `dvc repro -s serializar_modelo`)
#
# La imagen no instala torch, shap ni xgboost: la inferencia y SHAP corren en NumPy.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Mantener alineado con pyproject.toml (dependencias de src/api y src/models/inferencia.py).
RUN pip install "pandas>=2.2" "pyyaml>=6" "fastapi>=0.115" "uvicorn>=0.30"

# Sin privilegios: el servicio solo lee.
RUN useradd --system --no-create-home --uid 10001 quinuaprod

# La raíz de datos del servicio es /app/service-cnn; el contrato queda en /app/app-quinuaprod/api (para /docs).
WORKDIR /app/service-cnn

# Código: solo lo que importa el servicio.
COPY service-cnn/src/__init__.py src/__init__.py
COPY service-cnn/src/api/ src/api/
COPY service-cnn/src/models/__init__.py src/models/__init__.py
COPY service-cnn/src/models/inferencia.py src/models/inferencia.py
COPY service-cnn/src/explainability/__init__.py src/explainability/__init__.py
COPY service-cnn/src/explainability/shap_numpy.py src/explainability/shap_numpy.py

# Datos: un archivo por línea, para no copiar nada de más (p. ej. el resto de reports/).
COPY service-cnn/configs/provincias.csv configs/provincias.csv
COPY service-cnn/configs/preespecificacion.yaml configs/preespecificacion.yaml
COPY service-cnn/data/processed/exclusiones.csv data/processed/exclusiones.csv
COPY service-cnn/data/processed/metadata.json data/processed/metadata.json
COPY service-cnn/reports/E1_matriz_cobertura_enmendado_2016_2025.json reports/E1_matriz_cobertura_enmendado_2016_2025.json
COPY service-cnn/reports/piloto_oficial/resultados.json reports/piloto_oficial/resultados.json
COPY service-cnn/reports/piloto_oficial/predicciones.csv reports/piloto_oficial/predicciones.csv
COPY service-cnn/reports/piloto_oficial/p11_indicadores_y1.csv reports/piloto_oficial/p11_indicadores_y1.csv
COPY service-cnn/reports/piloto_oficial/p6_variabilidad_por_semilla.csv reports/piloto_oficial/p6_variabilidad_por_semilla.csv
COPY service-cnn/reports/he2_tecnico/resultados.json reports/he2_tecnico/resultados.json
COPY service-cnn/reports/sensibilidades/sensibilidades_rmse.csv reports/sensibilidades/sensibilidades_rmse.csv
COPY service-cnn/reports/sensibilidades/error_por_estrato_de_quinua.csv reports/sensibilidades/error_por_estrato_de_quinua.csv
COPY service-cnn/models/cnn1d_piloto/ models/cnn1d_piloto/
COPY app-quinuaprod/api/openapi.yaml ../app-quinuaprod/api/openapi.yaml

USER quinuaprod
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/salud', timeout=2)" || exit 1

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
