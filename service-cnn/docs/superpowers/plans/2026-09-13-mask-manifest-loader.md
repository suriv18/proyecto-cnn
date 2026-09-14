# Cargador de Umbrales de la Máscara Agrícola Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Conectar `data/manifest/mascara_agricola.yaml` (umbrales ya fundamentados con fuentes citables) con código real: una función que construya `MaskThresholds` desde el manifiesto, fallando explícitamente si `amplitud_ndvi_min` aún no está calibrado, y advirtiendo si algún umbral sigue marcado `PROVISIONAL` (sin aprobación del asesor).

**Architecture:** Un módulo nuevo, `src/ingestion/mask_manifest.py`, con una sola función pública `load_mask_thresholds`. Sigue el patrón ya establecido por `midagri_loader.load_column_mapping` (leer YAML con `yaml.safe_load`, construir un dataclass ya existente). No modifica `src/preprocessing/mask.py` (`MaskThresholds` ya existe y no cambia) ni el manifiesto YAML (ya escrito, corregido y comiteado).

**Tech Stack:** Python 3.14, PyYAML, pytest, `pathlib`.

## Global Constraints

- `MaskThresholds` (`src/preprocessing/mask.py`) no se modifica — sus 4 campos (`altitud_min_m`, `altitud_max_m`, `pendiente_max_grados`, `amplitud_ndvi_min`) ya existen y no cambian de tipo ni de nombre.
- El manifiesto real ya existe en `data/manifest/mascara_agricola.yaml`, con esta estructura VERIFICADA (vía `yaml.safe_load` directo contra el archivo real, no asumida — no volver a verificar esto, ya está confirmado):
  ```yaml
  umbrales:
    altitud_min_m: 2500
    altitud_max_m: 4000
    pendiente_max_porcentaje: 25
    pendiente_max_grados: 14.04
    amplitud_ndvi_min: null
    detalle:
      altitud:
        fuente: "..."
        estado: "PROVISIONAL — pendiente de aprobación del asesor"
      pendiente:
        fuente: "..."
        estado: "PROVISIONAL — pendiente de aprobación del asesor"
      amplitud_ndvi:
        fuente: "..."
        estado: "PENDIENTE — requiere ejecutar la calibración contra datos reales de entrenamiento"
  ```
  Los 4 valores numéricos están planos bajo `umbrales`; `fuente`/`estado` de cada uno están anidados bajo `umbrales.detalle.<nombre>` (`altitud` cubre tanto `altitud_min_m` como `altitud_max_m`; `pendiente` cubre `pendiente_max_grados`; `amplitud_ndvi` cubre `amplitud_ndvi_min`) — así `fuente`/`estado` no colisionan entre los 3 bloques (en YAML, claves repetidas al mismo nivel se sobrescriben silenciosamente; esta estructura anidada es la corrección de un bug real ya encontrado y arreglado en el manifiesto, commit `4c3870a`). No modificar el manifiesto como parte de este plan — los tests usan sus propios fixtures YAML en `tmp_path`, replicando esta misma estructura anidada.
- Si `umbrales.amplitud_ndvi_min` es `None`/`null`, se lanza `AmplitudNdviNoCalibradaError` (subclase de `ValueError`, mismo estilo que `HorizonteExcedidoError` en `src/ingestion/gee_series_builder.py:25-27`) — nunca se construye un `MaskThresholds` con ese campo faltante o inventado.
- Si `umbrales.detalle.<nombre>.estado` empieza con `"PROVISIONAL"`, se emite `warnings.warn(..., UserWarning)` por cada bloque de `detalle` cuyo `estado` cumpla esa condición — no bloquea la carga. `"PENDIENTE"` (el estado de `amplitud_ndvi` hoy) NO cuenta como provisional para esta advertencia — ese caso ya se maneja aparte con `AmplitudNdviNoCalibradaError`, que es un error, no un warning.
- La función devuelve `MaskThresholds` directamente (sin wrapper) — no cambia la forma de lo que consume el resto del pipeline.
- `ee`/red no aplica a este módulo — es lectura pura de YAML, sin GEE.

---

## Task 1: `load_mask_thresholds` — carga básica, error de amplitud faltante, advertencia de umbrales PROVISIONAL

**Files:**
- Create: `service-cnn/src/ingestion/mask_manifest.py`
- Create: `service-cnn/tests/ingestion/test_mask_manifest.py`

**Interfaces:**
- Consumes: `src.preprocessing.mask.MaskThresholds` (ya existe, solo se importa).
- Produces: `load_mask_thresholds(ruta_yaml: str | Path) -> MaskThresholds`. `AmplitudNdviNoCalibradaError(ValueError)`, exportada desde el mismo módulo.

- [ ] **Step 1: Escribir los tests en rojo**

Crear `service-cnn/tests/ingestion/test_mask_manifest.py`:

```python
"""Pruebas del cargador de umbrales de la máscara agrícola desde manifiesto.

Referencia: data/manifest/mascara_agricola.yaml, sección 4.10.1 de la tesis
("el nombre, versión, resolución y fecha de cada capa, así como los
umbrales de altitud, pendiente y amplitud, se fijarán y registrarán antes
de la evaluación definitiva").
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.ingestion.mask_manifest import (
    AmplitudNdviNoCalibradaError,
    load_mask_thresholds,
)
from src.preprocessing.mask import MaskThresholds


def _escribir_manifiesto(
    tmp_path: Path,
    altitud_min_m: float,
    altitud_max_m: float,
    pendiente_max_grados: float,
    amplitud_ndvi_min: float | None,
    estado_altitud: str | None = None,
    estado_pendiente: str | None = None,
    estado_amplitud: str | None = None,
) -> Path:
    """Construye un manifiesto YAML fixture con la MISMA estructura anidada
    del manifiesto real (umbrales.detalle.<nombre>.estado) — ver Global
    Constraints del plan para la estructura completa verificada."""
    detalle = {}
    if estado_altitud is not None:
        detalle["altitud"] = {"fuente": "fixture de prueba", "estado": estado_altitud}
    if estado_pendiente is not None:
        detalle["pendiente"] = {"fuente": "fixture de prueba", "estado": estado_pendiente}
    if estado_amplitud is not None:
        detalle["amplitud_ndvi"] = {"fuente": "fixture de prueba", "estado": estado_amplitud}

    umbrales = {
        "altitud_min_m": altitud_min_m,
        "altitud_max_m": altitud_max_m,
        "pendiente_max_grados": pendiente_max_grados,
        "amplitud_ndvi_min": amplitud_ndvi_min,
    }
    if detalle:
        umbrales["detalle"] = detalle

    ruta = tmp_path / "mascara_agricola.yaml"
    ruta.write_text(
        yaml.safe_dump({"umbrales": umbrales}, allow_unicode=True),
        encoding="utf-8",
    )
    return ruta


class TestLoadMaskThresholds:
    def test_carga_los_4_umbrales_correctamente(self, tmp_path):
        ruta = _escribir_manifiesto(
            tmp_path,
            altitud_min_m=2500,
            altitud_max_m=4000,
            pendiente_max_grados=14.04,
            amplitud_ndvi_min=0.15,
        )

        resultado = load_mask_thresholds(ruta)

        assert resultado == MaskThresholds(
            altitud_min_m=2500,
            altitud_max_m=4000,
            pendiente_max_grados=14.04,
            amplitud_ndvi_min=0.15,
        )

    def test_acepta_ruta_como_string(self, tmp_path):
        ruta = _escribir_manifiesto(
            tmp_path,
            altitud_min_m=2500,
            altitud_max_m=4000,
            pendiente_max_grados=14.04,
            amplitud_ndvi_min=0.15,
        )

        resultado = load_mask_thresholds(str(ruta))

        assert resultado.amplitud_ndvi_min == 0.15

    def test_amplitud_ndvi_min_nulo_lanza_error_explicito(self, tmp_path):
        ruta = _escribir_manifiesto(
            tmp_path,
            altitud_min_m=2500,
            altitud_max_m=4000,
            pendiente_max_grados=14.04,
            amplitud_ndvi_min=None,
        )

        with pytest.raises(AmplitudNdviNoCalibradaError, match="sample_cells_for_calibration"):
            load_mask_thresholds(ruta)

    def test_advierte_si_altitud_esta_marcada_provisional(self, tmp_path):
        ruta = _escribir_manifiesto(
            tmp_path,
            altitud_min_m=2500,
            altitud_max_m=4000,
            pendiente_max_grados=14.04,
            amplitud_ndvi_min=0.15,
            estado_altitud="PROVISIONAL — pendiente de aprobación del asesor",
        )

        with pytest.warns(UserWarning, match="PROVISIONAL"):
            load_mask_thresholds(ruta)

    def test_advierte_si_pendiente_esta_marcada_provisional(self, tmp_path):
        ruta = _escribir_manifiesto(
            tmp_path,
            altitud_min_m=2500,
            altitud_max_m=4000,
            pendiente_max_grados=14.04,
            amplitud_ndvi_min=0.15,
            estado_pendiente="PROVISIONAL — pendiente de aprobación del asesor",
        )

        with pytest.warns(UserWarning, match="PROVISIONAL"):
            load_mask_thresholds(ruta)

    def test_no_advierte_si_no_hay_seccion_detalle(self, tmp_path, recwarn):
        ruta = _escribir_manifiesto(
            tmp_path,
            altitud_min_m=2500,
            altitud_max_m=4000,
            pendiente_max_grados=14.04,
            amplitud_ndvi_min=0.15,
        )

        load_mask_thresholds(ruta)

        assert len(recwarn) == 0

    def test_estado_pendiente_de_amplitud_no_genera_warning_extra(self, tmp_path, recwarn):
        """amplitud_ndvi_min ya vale 0.15 (calibrado) en este fixture, pero
        su bloque `detalle.amplitud_ndvi.estado` todavía dice "PENDIENTE"
        (posible desfase entre el valor numérico y la nota de estado, ej.
        si alguien actualiza el valor pero olvida actualizar el estado) —
        "PENDIENTE" no empieza con "PROVISIONAL", así que no debe generar
        warning; solo se advierte por estados que literalmente empiezan
        con "PROVISIONAL". Usa la fixture `recwarn` (no `pytest.warns(None)`,
        removido en pytest moderno) para verificar ausencia de warnings."""
        ruta = _escribir_manifiesto(
            tmp_path,
            altitud_min_m=2500,
            altitud_max_m=4000,
            pendiente_max_grados=14.04,
            amplitud_ndvi_min=0.15,
            estado_amplitud="PENDIENTE — requiere ejecutar la calibración contra datos reales de entrenamiento",
        )

        load_mask_thresholds(ruta)

        assert len(recwarn) == 0

    def test_carga_correctamente_el_manifiesto_real_del_proyecto(self):
        """Verifica que el manifiesto real (ya comiteado en el repo) es
        cargable con esta función — no solo los fixtures sintéticos de
        arriba. Hoy el manifiesto real tiene amplitud_ndvi_min: null y
        altitud/pendiente marcados PROVISIONAL, así que se espera
        AmplitudNdviNoCalibradaError (no una carga exitosa) — este test
        solo confirma que no hay un error de INTEGRACIÓN (ruta rota,
        estructura YAML no parseable, KeyError por campo faltante) antes
        de llegar a ese punto esperado."""
        ruta_real = Path("data/manifest/mascara_agricola.yaml")
        with pytest.raises(AmplitudNdviNoCalibradaError):
            load_mask_thresholds(ruta_real)
```

- [ ] **Step 2: Ejecutar los tests para verificar que fallan**

Run: `cd service-cnn && python -m pytest tests/ingestion/test_mask_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ingestion.mask_manifest'`.

- [ ] **Step 3: Implementar el módulo**

Crear `service-cnn/src/ingestion/mask_manifest.py`:

```python
"""Cargador de umbrales de la máscara agrícola desde el manifiesto reproducible.

Referencia: data/manifest/mascara_agricola.yaml; sección 4.10.1 de la tesis
("el nombre, versión, resolución y fecha de cada capa, así como los
umbrales de altitud, pendiente y amplitud, se fijarán y registrarán antes
de la evaluación definitiva").

Este módulo NO importa `ee`/`geemap` ni toca la red: es lectura pura de un
archivo YAML ya existente, construyendo el `MaskThresholds` que consume
`src.preprocessing.mask.apply_agricultural_mask` y
`src.ingestion.gee_mask_pipeline.build_agricultural_mask_image`.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import yaml

from src.preprocessing.mask import MaskThresholds


class AmplitudNdviNoCalibradaError(ValueError):
    """`amplitud_ndvi_min` aún no se calibró contra datos reales de entrenamiento.

    A diferencia de `altitud_min_m`/`altitud_max_m`/`pendiente_max_grados`
    (criterios agronómicos externos, ya fijados con fuente citada en el
    manifiesto), la tesis (sección 4.10.1) exige que este umbral se calibre
    estadísticamente — no se puede inventar ni dejar en blanco.
    """


def load_mask_thresholds(ruta_yaml: str | Path) -> MaskThresholds:
    """Construye `MaskThresholds` desde el manifiesto de la máscara agrícola.

    Args:
        ruta_yaml: ruta al manifiesto (ej. `data/manifest/mascara_agricola.yaml`).

    Returns:
        `MaskThresholds` con los 4 valores leídos de la sección `umbrales`
        del manifiesto.

    Raises:
        AmplitudNdviNoCalibradaError: si `umbrales.amplitud_ndvi_min` es
            `null` en el manifiesto — indica que aún falta ejecutar
            `gee_mask_pipeline.sample_cells_for_calibration` +
            `mask.calibrate_ndvi_amplitude_threshold` sobre campañas de
            entrenamiento antes de poder usar la máscara en producción.

    Nota: si `umbrales.detalle.<nombre>.estado` empieza con "PROVISIONAL"
    para algún umbral, se emite un `UserWarning` (no bloquea la carga) —
    ver `_advertir_umbrales_provisionales`.
    """
    contenido = yaml.safe_load(Path(ruta_yaml).read_text(encoding="utf-8"))
    umbrales = contenido["umbrales"]

    if umbrales.get("amplitud_ndvi_min") is None:
        raise AmplitudNdviNoCalibradaError(
            "amplitud_ndvi_min no está calibrado en el manifiesto "
            f"({ruta_yaml}). Ejecute gee_mask_pipeline."
            "sample_cells_for_calibration + mask.calibrate_ndvi_amplitude_"
            "threshold sobre campañas de entrenamiento reales, y registre "
            "el resultado en el manifiesto antes de construir MaskThresholds."
        )

    _advertir_umbrales_provisionales(umbrales)

    return MaskThresholds(
        altitud_min_m=umbrales["altitud_min_m"],
        altitud_max_m=umbrales["altitud_max_m"],
        pendiente_max_grados=umbrales["pendiente_max_grados"],
        amplitud_ndvi_min=umbrales["amplitud_ndvi_min"],
    )


def _advertir_umbrales_provisionales(umbrales: dict) -> None:
    detalle = umbrales.get("detalle", {})
    for nombre_umbral, info in detalle.items():
        estado = info.get("estado", "") if isinstance(info, dict) else ""
        if isinstance(estado, str) and estado.startswith("PROVISIONAL"):
            warnings.warn(
                f"Umbral '{nombre_umbral}' es PROVISIONAL: {estado}",
                UserWarning,
                stacklevel=2,
            )
```

- [ ] **Step 4: Ejecutar los tests para verificar que pasan**

Run: `cd service-cnn && python -m pytest tests/ingestion/test_mask_manifest.py -v`
Expected: PASS — 8 tests.

- [ ] **Step 5: Ejecutar toda la suite de tests de `service-cnn` para verificar que nada más se rompió**

Run: `cd service-cnn && python -m pytest tests/ -q --ignore=tests/features/test_dataset_builder.py`
Expected: PASS — 299 (base) + 8 (esta tarea) = 307, o el número exacto que reporte la salida real.

- [ ] **Step 6: Commit**

```bash
cd service-cnn
git add src/ingestion/mask_manifest.py tests/ingestion/test_mask_manifest.py
git commit -m "$(cat <<'EOF'
Agregar load_mask_thresholds: cargar MaskThresholds desde el manifiesto

Lee data/manifest/mascara_agricola.yaml y construye un MaskThresholds
real (src.preprocessing.mask), siguiendo el mismo patron que
midagri_loader.load_column_mapping. Si amplitud_ndvi_min es null en el
manifiesto (aun no calibrado estadisticamente, a diferencia de los
otros 3 umbrales que son criterio agronomico externo), lanza
AmplitudNdviNoCalibradaError en vez de construir un MaskThresholds
incompleto. Advierte (UserWarning, no bloquea) cuando un umbral esta
marcado PROVISIONAL en umbrales.detalle.<nombre>.estado -- hoy
altitud y pendiente, pendientes de aprobacion del asesor.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016mkokyMQFzBE7QpPxKea4U
EOF
)"
```

---

## Task 2: Actualizar `PENDIENTE.md`

**Files:**
- Modify: `service-cnn/docs/PENDIENTE.md`

**Interfaces:** ninguna — solo documentación.

- [ ] **Step 1: Marcar el paso de conexión al código como resuelto**

En `service-cnn/docs/PENDIENTE.md`, sección 2.1, localizar el párrafo que empieza con "No hay ningún código de producción que instancie `MaskThresholds` todavía..." (añadido en el commit `caf25a4`) y reemplazarlo por:

```markdown
`src.ingestion.mask_manifest.load_mask_thresholds` ya conecta el manifiesto
con código real: construye `MaskThresholds` leyendo
`data/manifest/mascara_agricola.yaml`, lanzando
`AmplitudNdviNoCalibradaError` si `amplitud_ndvi_min` sigue sin calibrar
(hoy el caso) y advirtiendo (sin bloquear) que `altitud_min_m`/
`altitud_max_m`/`pendiente_max_grados` siguen `PROVISIONAL` sin aprobación
del asesor.
```

- [ ] **Step 2: Actualizar el conteo de tests**

Correr `cd service-cnn && python -m pytest tests/ -q --ignore=tests/features/test_dataset_builder.py`, tomar el número real reportado, y actualizar la línea de conteo de tests de la sección 2.1 al valor real post-Task 1.

- [ ] **Step 3: Commit**

```bash
cd service-cnn
git add docs/PENDIENTE.md
git commit -m "$(cat <<'EOF'
Documentar que load_mask_thresholds ya conecta el manifiesto al codigo

src.ingestion.mask_manifest.load_mask_thresholds ya construye
MaskThresholds real desde data/manifest/mascara_agricola.yaml. Sigue
bloqueado el uso en produccion por: amplitud_ndvi_min sin calibrar
(requiere GEE real) y aprobacion pendiente del asesor para
altitud/pendiente.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016mkokyMQFzBE7QpPxKea4U
EOF
)"
```
