# Fichas de fase CRISP-ML(Q)

Marco de trabajo del proyecto desde 2026-10-09. Documento base: `docs/Proyecto_Tesis_Maestria_UNMSM_WMSR_CRISP-MLQ (1).docx` (Tabla 2, sección 4.2) y `docs/Cronograma_Proyecto_Tesis_UNMSM_WMSAR_CRISP-MLQ (1).xlsx`.

Referencias: Peffers et al. (2007) para la DSRM; Studer et al. (2021) para CRISP-ML(Q).

## Regla de trabajo

Cada fase tiene una ficha con: requisitos y métricas de calidad, riesgos, tareas de control, estado con evidencia y **criterio de gate**. No se avanza a la fase siguiente (no se escribe código nuevo de esa fase) sin el gate de la anterior cerrado, salvo decisión explícita registrada en la ficha.

## Tablero (corte 2026-10-09)

| Ficha | Fase | Actividades del cronograma | Ventana | Entregables / hitos | Estado | Gate |
|---|---|---|---|---|---|---|
| [F1](F1-comprension-negocio-datos.md) | Comprensión del negocio y de los datos | 1, 2 | jul-2026 → ago-2026 (act. 2 hasta mar-2027) | H1, E1 | Parcial | **Abierto** |
| [F2](F2-preparacion-datos.md) | Preparación de datos | 3, 4, 5 | ago-2026 → oct-2026 | E2 (exploratorio), **E3 entregado** | ~90 % (código y datos completos; falta la aprobación del asesor) | **Abierto** |
| [F3](F3-modelado.md) | Modelado | 6, 7 | oct-2026 → nov-2026 | E4, H2, E5 | Código con datos sintéticos | Bloqueado por F2 |
| [F4](F4-evaluacion.md) | Evaluación | 8, 9 | nov-2026 → dic-2026 | E6 | Código con datos sintéticos | Bloqueado por F3 |
| [F5](F5-empaquetado-reproducible.md) | Despliegue (empaquetado reproducible) | 11 | abr-2027 → jun-2027 | E8 | Parcial (DVC, manifiestos) | Futuro |
| — | Comunicación y cierre (DSRM A6) | 10, 12 | ene-2027 → ago-2027 | E7, H3 | No iniciada | Futuro |
| — | F6 Monitoreo y mantenimiento | — | — | — | **Fuera de alcance** (sec. 1.5-1.6): el artefacto no se desplegará en operación | — |

## Informes entregados

- [E3 · Informe de depuración y análisis exploratorio](E3-informe-depuracion-eda.md) — generado por `scripts/build_e3_report.py`.

## Desviaciones conocidas respecto del plan

1. **H1 previsto para ago-2026 y sigue abierto** (≈2 meses de retraso). F2 y F3 se solapan en la práctica porque se codificó F3/F4 antes de cerrar F1/F2.
2. **El docx CRISP-MLQ conserva el periodo original 2005/2006–2023/2024 (19 campañas, N0 = 1 520)**; el código, los manifiestos y los datos reales ya usan 2016–2025 (10 campañas, N0 = 800) por la enmienda H1, todavía en estado `PROPUESTA`. Hasta que el asesor la apruebe y se reaplique al docx, documento y código se contradicen. Se rastrea en F1 (R-F1-1).
3. Seis decisiones dependen del asesor (ver F1, sección "Bloqueos").

## Convenciones de las fichas

- **Requisito/Métrica**: condición medible que debe cumplirse (tarea de calidad de CRISP-ML(Q)).
- **Riesgo**: lo que puede invalidar el requisito; el ID se cita desde las tareas de control.
- **Control**: acción concreta que mitiga el riesgo y deja evidencia.
- **Estado**: `Hecho` (con evidencia verificable en repo), `Parcial`, `Pendiente`.
- Las cifras de tests provienen de `docs/PENDIENTE.md` (299/299 al último cierre); **no se re-ejecutaron** al crear estas fichas.
- Actualizar la ficha cuando cambie el estado de un requisito, no al final de la fase.
