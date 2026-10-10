# F5 · Despliegue (adaptado: empaquetado reproducible)

- **DSRM**: A6 (comunicación). F6 (monitoreo) queda **fuera de alcance**.
- **Cronograma**: Act. 11 (abr→jun 2027). Comunicación y cierre: Act. 10 redacción (ene→abr 2027) y Act. 12 revisión, similitud y sustentación (jun→ago 2027).
- **Entregables**: E8 (artículo enviado y repositorio documentado), E7 (manuscrito), H3 (versión final y sustentación).
- **Secciones de la tesis**: 4.17, 4.18, Capítulo 6.
- **Estado de la fase**: Parcial. Fase futura; se registra para no perder lo ya construido.

## Requisitos y métricas de calidad

| ID | Requisito (plan 4.18) | Estado | Evidencia |
|---|---|---|---|
| Q-F5-1 | Repositorio versionado con scripts de descarga, construcción, entrenamiento, evaluación, explicaciones y figuras, con licencia | Parcial | Git con historial; falta licencia y script de figuras |
| Q-F5-2 | Manifiesto de datos (fuente, versión, URL, fecha, hash, licencia, cobertura, transformación) | Parcial | `data/manifest/*.yaml`; verificar campos de hash y licencia |
| Q-F5-3 | Entorno de ejecución (versiones, dependencias, contenedor, hardware) | Pendiente | `pyproject.toml`; no verificado contenedor ni lockfile |
| Q-F5-4 | Preespecificación fechada | Pendiente | Depende de E5 (F3) |
| Q-F5-5 | Resultados intermedios por campaña, modelo y semilla | Pendiente | Depende de F4 |
| Q-F5-6 | Scripts de reconstrucción donde la licencia impida redistribuir datos | Pendiente | Revisar términos MIDAGRI, INEI, Copernicus, CHIRPS, MODIS, SENAMHI |
| Q-F5-7 | Pipeline DVC reproducible de extremo a extremo | Parcial | `dvc.yaml` con 5 stages; `preprocess`, `train_eval` y `explain` apuntan a placeholders |
| Q-F5-8 | Control de similitud, verificación de referencias en fuente original, declaración de uso de IA generativa | Pendiente | — |

## Riesgos y controles

| ID | Riesgo | Control |
|---|---|---|
| R-F5-1 | Datos no redistribuibles por licencia | Publicar código + manifiesto + reglas de reconstrucción |
| R-F5-2 | Dependencia de credenciales GEE (`gee_project_id`) no portables | Documentar autenticación y alternativa de exportación |
| R-F5-3 | `dvc.lock` desfasado respecto de `dvc.yaml` | `dvc repro` limpio como requisito del gate |

## Criterio de gate F5

Un tercero puede reconstruir resultados desde cero con el repositorio, el entorno y el manifiesto; E8 entregado; E7 y H3 según cronograma.
