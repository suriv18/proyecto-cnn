# F1 · Comprensión del negocio y de los datos

- **DSRM**: A1–A2 (problema, objetivos de la solución) + ciclo de rigor.
- **Cronograma**: Act. 1 (jul→ago 2026) auditoría de cobertura y elegibilidad; Act. 2 (jul 2026→mar 2027) revisión de literatura.
- **Entregables**: H1 (hito), E1 (matriz provincia × campaña e informe de viabilidad, actualizado con protocolo y matriz de antecedentes).
- **Secciones de la tesis**: 1.1–1.4, 2.1–2.2, 4.5–4.8, Tablas 4–6.
- **Estado de la fase**: Parcial. **Gate H1: abierto.**

## Requisitos y métricas de calidad

| ID | Requisito / métrica | Estado | Evidencia |
|---|---|---|---|
| Q-F1-1 | Disponibilidad de rendimiento, producción y superficie a escala provincial verificada con fuente real (sec. 4.5.2) | Hecho | `data/manifest/midagri_sisagri.yaml`; commit "Ejecuta la auditoría H1 con datos reales de MIDAGRI" |
| Q-F1-2 | Cascada N0–N5 (Tabla 5) sin tasas supuestas | Parcial | N0=800, N1=770, N2=670, N3=620, N4=604 (enmendado) / 730 (`_real`); **N5 = null** |
| Q-F1-3 | Un único E1 canónico y versionado | Pendiente | Hay 5 JSON en `reports/E1_matriz_cobertura*.json` (original_2006_2024, 2016_2025, enmendado_2016_2025, real, base); N4 difiere (604 vs 730) sin explicación documentada. Dato nuevo (2026-10-09): 730 celdas incluyen campañas fuera del periodo (2015 y 2026); en 2016–2025 hay **674** con producción y superficie cosechada > 0, que tampoco coincide con 604 |
| Q-F1-4 | Estabilidad de la demarcación provincial verificada (sec. 4.5.4) | Pendiente | Sin evidencia en repo |
| Q-F1-5 | Clasificación de celdas: cero estructural / vacío por intermitencia / dato faltante (4.5.5) | Parcial | `scripts/compute_n4_n5.py`; sin informe de resultados |
| Q-F1-6 | Protocolo de revisión de literatura y matriz de antecedentes conservados (Act. 2, sec. 2.1) | Pendiente | Sin registro de búsquedas, duplicados ni exclusiones en el repo |
| Q-F1-7 | Rango plausible de rendimiento y tolerancia de redondeo definidos antes de H1 (4.7) | Pendiente | Sin evidencia |

## Riesgos

| ID | Riesgo | Condición de activación (Tabla 6) | Estado |
|---|---|---|---|
| R-F1-1 | **Divergencia documento–código sobre el periodo**: docx CRISP = 2005/06–2023/24 (19 camp.); código/datos = 2016–2025 (10 camp.) | Enmienda no aprobada o no reflejada en el docx | **Activo** |
| R-F1-2 | Cobertura provincial más corta que el periodo propuesto | Serie insuficiente para ventana inicial + varias campañas externas | **Materializado**: MIDAGRI no cubre 2006–2014; con 10 campañas quedan pocas campañas externas |
| R-F1-3 | Ampliación de N3 (62 → 76 provincias con producción observada) decidida unilateralmente | — | Pendiente de decisión del asesor |
| R-F1-4 | Distritos SISAGRI sumados a provincia sin confirmar | — | Se aplica en código; marcado "pendiente confirmar" en manifiesto |

## Tareas de control

| Control | Mitiga | Acción | Resultado esperado |
|---|---|---|---|
| C-F1-1 | R-F1-1 | Reaplicar la enmienda (2016–2025, N0=800) al docx CRISP-MLQ **tras** aprobación del asesor; registrar versión | Docx, manifiesto y código coherentes |
| C-F1-2 | R-F1-2 | Calcular y documentar el número de campañas externas posibles con ventana inicial mínima; advertir que la potencia de HE1/HE2 se reduce | Sección de viabilidad en E1 |
| C-F1-3 | Q-F1-3 | Elegir un E1 canónico, mover los demás a `reports/archive/` y explicar la diferencia N4 604 vs 730 | Un solo `E1_matriz_cobertura.json` + nota |
| C-F1-4 | Q-F1-6 | Crear registro de búsquedas (fuente, cadena, fecha, resultados, exclusiones) junto con la matriz de extracción | Anexo de revisión |
| C-F1-5 | R-F1-3, R-F1-4 | Llevar al asesor las decisiones de "Bloqueos" | Acta o correo archivado en `docs/` |

## Bloqueos (decisiones del asesor, de `docs/PENDIENTE.md` §1)

1. Aprobación formal de la enmienda de periodo 2006–2024 → 2016–2025.
2. Ampliar o no N3 a 76 provincias (+14 sobre las 62 preespecificadas).
3. Confirmar suma distrito → provincia en SISAGRI.
4. Confirmar la ventana de extracción amplia (año calendario jul(t-1)–jun(t)).
5. Fuente alterna para 2006–2015 (SENAMHI/INEI); probablemente no existe.
6. (Transversal a F2) Validar el respaldo departamental exploratorio.

## Criterio de gate H1

Se cierra cuando: Q-F1-1…Q-F1-5 y Q-F1-7 cumplidos, E1 canónico publicado con N5 calculado, enmienda aprobada por el asesor y reflejada en el docx, y registro de literatura conservado. **Condición de continuidad**: si la cobertura no alcanza, enmienda metodológica antes de entrenar (Tabla 6).
