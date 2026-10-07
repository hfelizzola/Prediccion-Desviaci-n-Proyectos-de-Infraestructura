# Changelog

Todos los cambios relevantes del proyecto se documentan aquí.
Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/). Versionado:
MAYOR = cambian los resultados reportados, MENOR = nuevos análisis sin cambiar resultados,
PARCHE = correcciones sin efecto en resultados.

## [Unreleased]

### Cambiado
- `versionado.sh init` fija `core.eol=lf` y `core.autocrlf=false` para no reescribir archivos con
  CRLF dentro de Google Drive.

## [v1.0.0] - 2026-10-07

### Agregado
- Código, consultas SoQL y notebooks usados en la tesis doctoral (pipeline ETL → ingeniería de
  variables → ajuste con Optuna → análisis ANOVA/SHAP).
- Control de versiones: `.gitignore`, `.gitattributes`, `scripts/versionado.sh` con hook
  pre-commit contra secretos y archivos grandes, manifiesto SHA-256 de datos y modelos.
- `README.md`, `ROADMAP.md`, `.env.example`.

### Seguridad
- Credenciales de la API Socrata eliminadas de `extractData.py`, `ETL.ipynb` y
  `modeloPrediccionAdicionesObras.ipynb`; ahora se leen de variables de entorno
  (`SOCRATA_APP_TOKEN`, `SOCRATA_USERNAME`, `SOCRATA_PASSWORD`). La lógica de los notebooks no cambió.
