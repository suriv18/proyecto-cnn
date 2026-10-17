"""Endpoints de solo lectura. Cada uno implementa una operación de `app-quinuaprod/api/openapi.yaml`."""
from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from .problemas import Problema
from .repositorio import Repositorio

Modelo = Literal["cnn1d", "elastic_net", "random_forest", "xgboost", "svr", "B1", "B2", "B3", "B4"]
Clasificacion = Literal["altoandina", "transicion", "selva", "costa_riego", "pendiente_verificacion"]
Variante = Literal[
    "base", "sin_transicion", "tensor_completo", "horizonte_temprano", "sin_arrastre_ndvi", "con_exclusion_por_altitud"
]
Motivo = Literal["fuera_del_ambito_altoandino", "ciclo_fuera_de_ventana_de_extraccion", "tensor_incompleto"]

router = APIRouter()


def repositorio(request: Request) -> Repositorio:
    return request.app.state.repositorio


Repo = Annotated[Repositorio, Depends(repositorio)]


@router.get("/meta")
def meta(repo: Repo):
    return repo.meta()


@router.get("/provincias")
def provincias(repo: Repo, departamento: str | None = None, clasificacion: Clasificacion | None = None):
    return [
        p
        for p in repo.provincias()
        if (departamento is None or p["departamento"] == departamento)
        and (clasificacion is None or p["clasificacion"] == clasificacion)
    ]


@router.get("/cobertura/embudo")
def embudo(repo: Repo):
    return repo.embudo()


@router.get("/cobertura/exclusiones")
def exclusiones(
    repo: Repo, campana: int | None = None, motivo: Motivo | None = None, provincia_id: str | None = None
):
    return [
        e
        for e in repo.exclusiones()
        if (campana is None or e["campana"] == campana)
        and (motivo is None or e["motivo"] == motivo)
        and (provincia_id is None or e["provincia_id"] == provincia_id)
    ]


@router.get("/predicciones")
def predicciones(
    repo: Repo,
    modelo: Modelo | None = None,
    campana_prueba: int | None = None,
    provincia_id: str | None = None,
    semilla: int | None = None,
    limit: Annotated[int, Query(ge=1, le=5000)] = 500,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    if campana_prueba is not None and campana_prueba not in repo.campanas_de_prueba:
        raise Problema(
            422,
            "Campaña no disponible",
            "Solo hay campañas de prueba " + ", ".join(str(c) for c in repo.campanas_de_prueba) + ".",
        )
    df = repo.predicciones()
    for columna, valor in (
        ("modelo", modelo),
        ("campana_prueba", campana_prueba),
        ("provincia_id", provincia_id),
        ("semilla", semilla),
    ):
        if valor is not None:
            df = df[df[columna] == valor]
    pagina = df.iloc[offset : offset + limit]
    return {
        "total": int(len(df)),
        "limit": limit,
        "offset": offset,
        "items": pagina.astype(object).where(pagina.notna(), None).to_dict("records"),
    }


class Tensor(BaseModel):
    valores: list[list[float]] = Field(min_length=6, max_length=6)
    altitud_media_m: float
    prop_quinua: float = Field(ge=0)


class SolicitudInferencia(BaseModel):
    provincia_id: str | None = None
    campana: int | None = None
    tensor: Tensor | None = None


@router.post("/predicciones/inferir")
def inferir(solicitud: SolicitudInferencia, repo: Repo):
    if solicitud.tensor is not None:
        if any(len(fila) != 5 for fila in solicitud.tensor.valores):
            raise Problema(422, "Solicitud inválida", "Cada fila del tensor debe tener 5 fases.")
        raise Problema(
            501,
            "No implementado",
            "Predecir sobre un tensor editado requiere el CNN-1D serializado, que aún no existe.",
        )
    if not solicitud.provincia_id or solicitud.campana is None:
        raise Problema(422, "Solicitud inválida", "Indique provincia_id y campana, o un tensor.")
    if solicitud.campana not in repo.campanas_permitidas:
        raise Problema(
            422,
            "Campaña no permitida",
            "Campañas permitidas: " + ", ".join(str(c) for c in repo.campanas_permitidas) + ".",
        )
    resultado = repo.inferir_observacion(solicitud.provincia_id, solicitud.campana)
    if resultado is None:
        raise Problema(404, "Sin predicción", "La observación no está entre las campañas de prueba del piloto.")
    return resultado


@router.get("/indicadores/y1")
def indicadores_y1(repo: Repo):
    return repo.indicadores_y1()


@router.get("/indicadores/x1-x2")
def indicadores_x1_x2(repo: Repo):
    return repo.indicadores_x1_x2()


@router.get("/indicadores/variabilidad-semillas")
def variabilidad_semillas(repo: Repo):
    return repo.variabilidad_semillas()


@router.get("/explicabilidad")
def explicabilidad(repo: Repo):
    return repo.explicabilidad()


@router.get("/sensibilidades")
def sensibilidades(repo: Repo, variante: Variante | None = None, modelo: Modelo | None = None):
    df = repo.sensibilidades()
    if variante is not None:
        df = df[df["variante"] == variante]
    if modelo is not None:
        df = df[df["modelo"] == modelo]
    return df.astype(object).where(df.notna(), None).to_dict("records")


@router.get("/sensibilidades/estratos")
def sensibilidades_estratos(repo: Repo):
    return repo.sensibilidades_estratos()


@router.get("/preespecificacion")
def preespecificacion(repo: Repo):
    return repo.preespecificacion()
