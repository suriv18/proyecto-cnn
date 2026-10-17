# Decisión: ¿usar PostgreSQL 18?

**Estado:** evaluación del 2026-10-17. **Recomendación: no usar base de datos por ahora.** Se revisa si ocurre alguno de los disparadores de la última sección.

## Cómo está hoy el sistema

- **Solo lectura.** La API (`service-cnn/src/api`) sirve resultados del piloto; no recibe ni guarda datos.
- **Datos pequeños.** El mayor es `predicciones.csv` (4 320 filas); todos los artefactos del servicio suman menos de 1 MB. Se leen una vez y quedan en memoria.
- **Sin estado compartido ni usuarios.** No hay autenticación, sesiones ni datos personales (RNF5).
- **Reproducibilidad basada en archivos.** El plan de §4.18 de la tesis y el pipeline (DVC) producen y versionan archivos; `docs/02` descarta explícitamente un feature store o servicio de datos.

## Qué aportaría PostgreSQL y qué costaría

| Posible uso | ¿Lo necesita el sistema hoy? | Comentario |
|---|---|---|
| Servir los resultados del piloto | No | Con 4 320 filas y lectura única, un archivo en memoria responde más rápido y sin operar nada |
| Guardar consultas o inferencias | No | No se requiere trazabilidad por usuario; la inferencia no persiste nada |
| Cuentas de usuario y permisos | No | La API es pública y de solo lectura |
| Consultas analíticas sobre las series crudas (GEE, SISAGRI) | Podría, pero Postgres no es lo ideal | Son parquet de uso por lotes; DuckDB o los propios parquet encajan mejor |
| Varias instancias de la API compartiendo estado | No | No hay estado que compartir |

**Costos de añadirlo:** un componente más que operar (respaldos, parches, seguridad, migraciones); un servicio extra en CI y en el despliegue; y una segunda fuente de verdad que puede quedar distinta de los archivos del pipeline, lo que choca con el plan de reproducibilidad.

## Sobre la versión 18

- Es la versión estable vigente. La 19 estaba planificada para el 29 de octubre de 2026 (RC1 el 15 de octubre), según [el anuncio del equipo de lanzamiento](https://www.postgresql.org/message-id/915517c4-d0ee-49f9-bbeb-c683ef2b9d3f%40postgresql.org) y [el anuncio de RC1](https://www.postgresql.org/about/news/postgresql-19-rc-1-released-3409/); un proyecto que empiece ahora puede usar la 18 y planificar la actualización.
- Según la política que conozco, la comunidad da cinco años de soporte por versión mayor; es coherente con que la 14 deje de recibir correcciones el 12 de noviembre de 2026 (lo indica [postgresql.org](https://www.postgresql.org/) y [esta guía de actualización](https://wz-it.com/en/blog/postgresql-14-end-of-life-upgrade-guide/)). Por esa regla, la 18 llegaría a finales de 2030, pero **no consulté la página oficial de la política de versiones ni confirmé esa fecha**.
- Ninguna novedad de la 18 resuelve algo que este sistema necesite hoy: la decisión no depende de la versión sino de si hay una base de datos que justifique.

## Alternativas si se quiere consultar sin operar un servidor

1. **Dejarlo como está** (archivos + memoria). Recomendado.
2. **SQLite o DuckDB embebido**, si en algún momento se necesitan consultas SQL sobre los resultados. Un archivo, sin servidor, reproducible con DVC.

## Disparadores para reconsiderar

Se justificaría PostgreSQL (18 o la vigente entonces) si el alcance cambia a alguno de estos:

1. **Sistema vivo:** ingesta periódica de campañas nuevas y publicación continua (hoy la tesis lo deja fuera: F6 monitoreo y producción continua).
2. **Cuentas o permisos:** acceso restringido por usuario o registro de quién consultó qué.
3. **Escritura desde la aplicación:** por ejemplo, guardar escenarios o predicciones solicitadas por los usuarios.
4. **Varias instancias** del servicio que deban compartir datos que cambian.

Si ocurre alguno, la migración es acotada: `src/api/repositorio.py` es el único lugar que lee datos, y los contratos de la API no cambian.
