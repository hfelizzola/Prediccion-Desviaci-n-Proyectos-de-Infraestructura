# Predicción interpretable de desviaciones en tiempo y costo en proyectos de infraestructura (SECOP)

Modelos de machine learning interpretables para predecir, en el momento de la firma, si un contrato
público de obra en Colombia tendrá **adiciones en costo** o **adiciones en tiempo**. Los datos vienen
del Sistema Electrónico de Contratación Pública (SECOP I y II, datos.gov.co), de TerriData (DNP) y del
registro de multas y sanciones del SECOP.

| Versión | Contenido |
|---------|-----------|
| `v1.0.0` | Código y notebooks tal como se usaron en la tesis doctoral (ver `notebooks/tesis_v1/`). |
| `v1.1.0` | Paquete `secop_dev` reproducible: mismos datos y resultados que la tesis, verificados con pruebas de regresión. |

Las mejoras planeadas para el artículo están en [ROADMAP.md](ROADMAP.md) y los cambios entre
versiones en [CHANGELOG.md](CHANGELOG.md).

## Preguntas de investigación

1. ¿Con qué precisión se pueden predecir las desviaciones en tiempo y costo?
2. ¿Cuánto aportan los grupos de variables: proyecto y contratación (G1), entidad contratante (G2),
   contratista (G3) e indicadores territoriales (G4)?
3. ¿Qué variables de cada grupo determinan las desviaciones?

## Estructura

```
├── configs/                 parámetros del estudio (todo número "mágico" vive aquí)
│   ├── data.yaml            fuentes, reglas de limpieza, filtros de muestra, TerriData
│   ├── features.yaml        grupos de variables G1–G5
│   ├── experiments.yaml     escenarios S1–S5, modelos, Optuna, validación
│   └── frozen/              lista de atípicos de la tesis (Isolation Forest sin semilla)
├── src/secop_dev/           código del paquete
│   ├── data/                extracción (API Socrata), integración SECOP I/II, limpieza, multas, TerriData
│   ├── features/            filtros, variables del proyecto y de la entidad, tabla de entrenamiento
│   ├── models/              pipelines, espacios de búsqueda, ajuste con Optuna, experimentos
│   ├── analysis/            ANOVA de escenarios, SHAP, comparación con la tesis
│   └── cli.py               comando `secop-dev`
├── sql/                     consultas SoQL
├── tests/                   pruebas unitarias y de regresión contra la tesis
├── notebooks/
│   ├── colab_pipeline.ipynb ejecuta todo el pipeline en Google Colab
│   └── tesis_v1/            notebooks y scripts originales (congelados)
├── reports/tesis_v1/        tablas LaTeX de la tesis
└── scripts/versionado.sh    gestión de versiones (git + artefactos)
```

## Ejecución

### En Google Colab (recomendado para entrenar)

Abre [`notebooks/colab_pipeline.ipynb`](notebooks/colab_pipeline.ipynb) en Colab y ejecuta las
celdas. El notebook clona este repositorio, crea un entorno Python 3.11 con las versiones fijadas,
corre las pruebas y ejecuta el pipeline. Los datos se leen de la carpeta del proyecto en Drive y
los resultados quedan en `outputs/` dentro de esa misma carpeta. Si la sesión se corta, vuelve a
ejecutar la celda de entrenamiento: las combinaciones terminadas no se repiten.

### En local

```bash
uv venv --python 3.11 ~/.venvs/secop-dev           # entorno FUERA de Google Drive
VIRTUAL_ENV=~/.venvs/secop-dev uv pip install -e ".[dev]"
source ~/.venvs/secop-dev/Scripts/activate          # Linux/macOS: .../bin/activate

pytest                                  # pruebas (incluye regresión contra la tesis)
secop-dev data                          # integrar -> limpiar -> multas -> tabla de entrenamiento
secop-dev train --run-id prueba --model lr --n-trials 2   # prueba rápida
secop-dev train --run-id v1.1           # experimento completo (4 modelos x 5 escenarios x 2 respuestas)
secop-dev compare-v1 --run-id v1.1      # diferencias con las métricas de la tesis
secop-dev doe --run-id v1.1             # validación cruzada por escenario + ANOVA
secop-dev shap --run-id v1.1            # valores SHAP
```

Rutas: por defecto los datos se leen de `Data/` y `terridata/` y las salidas van a `outputs/`. Se
pueden cambiar con las variables de entorno `SECOP_DATA_DIR`, `SECOP_TERRIDATA_DIR` y
`SECOP_OUTPUT_DIR`.

### Flujo de datos

| Comando | Entradas | Salida (`outputs/`) |
|---------|----------|---------------------|
| `secop-dev extract` | API Socrata (SECOP I `f789-7hwg`, SECOP II `jbjy-vk9h`, `p6dx-8zbt`) | `raw/<fecha>/` |
| `secop-dev integrate` | `procesosSECOPI.csv`, `contratosSECOPII.csv`, `procesosSECOPII.csv` | `interim/contratosSECOP.csv` |
| `secop-dev clean` | lo anterior + `dataCorrections.xlsx` | `interim/contratosSECOPCleaned.csv` |
| `secop-dev penalties` | lo anterior + `multas_sanciones_secop.csv` | `interim/analisis_multas_por_entidad.csv` |
| `secop-dev features` | lo anterior + TerriData | `processed/trainData.csv` |
| `secop-dev train` | `processed/trainData.csv` | `runs/<run-id>/<respuesta>/<modelo>/<escenario>/` |

Cada corrida guarda el modelo, los hiperparámetros, las métricas, todos los trials de Optuna y un
`run_metadata.json` con el commit, la configuración y las versiones de las librerías.

## Reproducibilidad respecto a la tesis

`pytest -m regression` comprueba que el paquete regenera **exactamente** los archivos de la tesis
(`contratosSECOP.csv`, `contratosSECOPCleaned.csv`, `analisis_multas_por_entidad.csv` y
`trainData.csv`). Dos decisiones lo hacen posible:

- El Isolation Forest de la tesis no tenía semilla. Las 287 filas (282 `uid`) que eliminó se recuperaron
  comparando la muestra antes y después del filtro, y quedaron congelados en
  `configs/frozen/v1_outlier_uids.csv`.
- Las limitaciones metodológicas detectadas (ROADMAP, sección 1) se conservan a propósito en esta
  versión y están marcadas en el código con `ROADMAP Mx`. Se corrigen en la v2.0.0, una a una.

Una diferencia esperable: en la tesis el orden de las variables de cada escenario venía de un
`set` de Python y cambiaba entre sesiones; ahora lo fija `configs/features.yaml`. Los modelos que
muestrean columnas (Random Forest, XGBoost) pueden variar levemente por ese motivo.

## Datos y modelos (fuera de git)

Los datos crudos e intermedios, los modelos entrenados, los valores SHAP y las salidas (`outputs/`)
**no se versionan en git**. Cada versión registra la huella SHA-256 de esos archivos en
[`MANIFEST/artifacts.sha256`](MANIFEST/) y los empaqueta como adjunto de su release en GitHub:

```bash
bash scripts/versionado.sh verify                                      # ¿mis archivos son los de esta versión?
VERSIONADO_DIST=~/Downloads bash scripts/versionado.sh archive v1.0.0  # paquete para el release
```

Para restaurar una versión: `git checkout vX.Y.Z`, descargar `vX.Y.Z-artifacts.tar.gz` del release,
`tar -xzf vX.Y.Z-artifacts.tar.gz` en la raíz del proyecto y `bash scripts/versionado.sh verify`.

## Credenciales

La extracción usa la API Socrata de datos.gov.co. Copia `.env.example` como `.env` y completa
`SOCRATA_APP_TOKEN`, `SOCRATA_USERNAME` y `SOCRATA_PASSWORD` (instala los extras con
`pip install -e ".[extract]"`). **Nunca** escribas credenciales en el código: el hook pre-commit
bloquea commits que las contengan.

## Gestión de versiones

Todo el flujo de git se maneja con [`scripts/versionado.sh`](scripts/versionado.sh):

```bash
bash scripts/versionado.sh ayuda                       # lista de comandos
bash scripts/versionado.sh status                      # rama, última versión, cambios pendientes
bash scripts/versionado.sh branch fase2-datos          # nueva rama feature/fase2-datos
bash scripts/versionado.sh save "feat(data): paginar extracción"
bash scripts/versionado.sh finish                      # integrar la rama en main
bash scripts/versionado.sh snapshot v1.2.0 "Pipeline de datos reproducible"
bash scripts/versionado.sh publish                     # subir rama y tags a GitHub
```

Convención de versiones: **MAYOR** cuando cambian los resultados reportados (datos, muestra o
metodología), **MENOR** cuando se agregan análisis sin cambiar resultados previos y **PARCHE** para
correcciones que no alteran resultados. Los mensajes de commit siguen
[Conventional Commits](https://www.conventionalcommits.org/es/) (`feat`, `fix`, `refactor`, `data`,
`exp`, `docs`, `test`, …). El hook pre-commit corre además `ruff` y `nbstripout` si `pre-commit`
está instalado en el entorno activo.

En Windows el script se ejecuta desde Git Bash, o desde PowerShell con
`& "C:\Program Files\Git\bin\bash.exe" scripts/versionado.sh <comando>`.
