# Changelog

Todos los cambios relevantes del proyecto se documentan aquí.
Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/). Versionado:
MAYOR = cambian los resultados reportados, MENOR = nuevos análisis sin cambiar resultados,
PARCHE = correcciones sin efecto en resultados.

## [Unreleased]

Fase 1 del roadmap: reorganización del código **sin cambiar resultados**.

### Agregado
- Paquete `secop_dev` (`src/`, código en inglés) con la lógica de los notebooks de la tesis:
  extracción Socrata, integración SECOP I/II, limpieza, multas, TerriData, ingeniería de variables,
  ajuste con Optuna, ANOVA de escenarios, SHAP y comparación con la tesis.
- CLI `secop-dev` (`data`, `train`, `doe`, `shap`, `compare-v1`, `extract`, …). El entrenamiento es
  reanudable y cada corrida guarda modelo, hiperparámetros, métricas, trials y metadatos (commit,
  hash de configuración, versiones).
- `configs/data.yaml`, `features.yaml`, `experiments.yaml`: todos los parámetros del estudio.
- `configs/frozen/v1_outlier_uids.csv`: contratos que eliminó el Isolation Forest de la tesis
  (no tenía semilla), recuperados para reproducir la muestra.
- 24 pruebas con `pytest`, incluidas las de regresión que verifican que se regeneran exactamente
  `contratosSECOP.csv`, `contratosSECOPCleaned.csv`, `analisis_multas_por_entidad.csv` y
  `trainData.csv`.
- `pyproject.toml` + `uv.lock` con las versiones del entorno de la tesis (Python 3.11).
- `notebooks/colab_pipeline.ipynb`: ejecución completa en Colab desde GitHub.
- `ruff` y `.pre-commit-config.yaml` (ruff, nbstripout); el hook de `versionado.sh` los ejecuta.
- Hallazgo M16 en el roadmap: `uid` duplicados que se multiplican al unir el HHI.

### Cambiado
- Notebooks y scripts originales movidos a `notebooks/tesis_v1/`, consultas a `sql/` y tablas
  LaTeX a `reports/tesis_v1/`. `requirements.txt` (pip freeze de Colab) queda como
  `notebooks/tesis_v1/requirements-colab-2025.txt`.
- El cálculo de multas por entidad y año usa búsqueda binaria en vez de un bucle fila a fila
  (mismo resultado; ahora tarda 0,07 s).
- El orden de las variables de cada escenario lo fija `configs/features.yaml` (en la tesis venía
  de un `set` de Python y cambiaba entre sesiones).
- `versionado.sh init` fija `core.eol=lf` y `core.autocrlf=false` para no reescribir archivos con
  CRLF dentro de Google Drive; `outputs/` se incluye en el manifiesto de artefactos.

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
