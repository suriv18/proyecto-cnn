# Configuración pendiente de datos reales

- **`provincias.csv`** — placeholder vacío. Reemplazar con la clasificación territorial oficial (Tabla 4, sección 4.5.3 de la tesis): las 80 provincias de los 8 departamentos delimitados (Puno, Ayacucho, Apurímac, Arequipa, Junín, Cusco, La Libertad, Huancavelica), columnas `provincia_id,departamento,clasificacion` con `clasificacion` en `{altoandina, transicion, selva, costa_riego}`.
- **`produccion_documentada.csv`** — placeholder vacío. Reemplazar con el resultado de verificar producción documentada por provincia (Tabla 3), columnas `provincia_id,produccion_documentada` (booleano), confirmado contra MIDAGRI/SIEA durante la auditoría H1.
- **`column_mapping.yaml`** — contiene nombres de columna *placeholder* para el archivo real de MIDAGRI/SIEA (ver comentario dentro del archivo). `scripts/run_ingestion.py` falla explícitamente mientras no se reemplacen.
- **`hyperparams.yaml`** — espacios de búsqueda de Optuna por modelo. Se congela como parte del registro de preespecificación (H2) antes de la evaluación definitiva.
