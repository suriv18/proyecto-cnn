"""Servicio de solo lectura que expone los resultados del piloto a app-quinuaprod.

El contrato es `app-quinuaprod/api/openapi.yaml`; este paquete solo lo implementa. No entrena,
no evalua y no escribe: lee los artefactos que genera el pipeline (`reports/`, `data/processed/`,
`configs/`) y los entrega con la forma que define el contrato.
"""
