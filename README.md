# Predicción interpretable de desviaciones en tiempo y costo en proyectos de infraestructura (SECOP)

Modelos de machine learning interpretables para predecir, en el momento de la firma, si un contrato
público de obra en Colombia tendrá **adiciones en costo** o **adiciones en tiempo**. Los datos vienen
del Sistema Electrónico de Contratación Pública (SECOP I y II, datos.gov.co), de TerriData (DNP) y del
registro de multas y sanciones del SECOP.

> **Versión v1.0.0 — tesis doctoral.** Este repositorio guarda el código tal como se usó en la tesis.
> Las mejoras planeadas para el artículo están en [ROADMAP.md](ROADMAP.md) y los cambios entre
> versiones en [CHANGELOG.md](CHANGELOG.md).

## Preguntas de investigación

1. ¿Con qué precisión se pueden predecir las desviaciones en tiempo y costo?
2. ¿Cuánto aportan los grupos de variables: proyecto y contratación (G1), entidad contratante (G2),
   contratista (G3) e indicadores territoriales (G4)?
3. ¿Qué variables de cada grupo determinan las desviaciones?

## Flujo de la versión de la tesis (v1.0.0)

| Paso | Notebook / script | Entradas | Salidas |
|------|-------------------|----------|---------|
| 1. Extracción y limpieza | `ETL.ipynb` (+ `query*.sql`) | API Socrata (SECOP I `f789-7hwg`, SECOP II `jbjy-vk9h`, `p6dx-8zbt`, multas `4n4q-k399`), `Data/dataCorrections.xlsx` | `Data/contratosSECOPCleaned.csv`, `Data/analisis_multas_por_entidad.csv` |
| 2. Ingeniería de variables | `featureEngineering_V2.ipynb` | salidas del paso 1, `terridata/*.zip` | `Data/trainData.csv`, `Data/*Columns.txt` |
| 3. Entrenamiento y ajuste | `train_tuning_optuna_evaluate_v2.ipynb` + `ml_pipeline.py` | `Data/trainData.csv` | `Models V2/*.joblib`, `Data/metrics_*.xlsx`, `Data/*_best_params.xlsx` |
| 4. Análisis e interpretación | `model_analysis_V2.ipynb` | modelos y métricas | `Data/doe_results_cv_*.csv`, `anova_*.tex`, `shap_*.joblib` |

Los demás notebooks (`modeloPrediccionAdicionesObras*.ipynb`, `modelCostDeviation.ipynb`,
`fine_tuning_optuna.ipynb`, `train_tuning_optuna_evaluate.ipynb`, `model_analysis.ipynb`,
`featureEngineering.ipynb`, `subjects_variables_v2.ipynb`, `terridata_analysis.ipynb`,
`train_with_experimental_feature.ipynb`) son iteraciones previas o exploratorias y se conservan como
registro histórico.

Notas de la v1:
- Los notebooks se ejecutaron en Google Colab (Python 3.11, pandas 2.2, scikit-learn 1.6.1,
  xgboost 2.1.4, optuna 4.3, shap 0.47). `requirements.txt` es el `pip freeze` completo de ese
  entorno.
- `ETL.ipynb` escribe algunos CSV en la raíz; en la tesis se movieron a `Data/` a mano.

## Datos y modelos (fuera de git)

Los datos crudos e intermedios, los modelos entrenados y los valores SHAP (~780 MB) **no se versionan
en git**. Cada versión registra la huella SHA-256 de esos archivos en
[`MANIFEST/artifacts.sha256`](MANIFEST/) y se pueden empaquetar para un release de GitHub o Zenodo:

```bash
bash scripts/versionado.sh verify             # ¿mis archivos locales son los de esta versión?
bash scripts/versionado.sh archive v1.0.0     # crea dist/v1.0.0-artifacts.tar.gz
```

Sí se versionan las entradas curadas a mano (`Data/dataCorrections.xlsx`), las listas de variables
(`Data/*Columns.txt`) y las tablas de resultados pequeñas (métricas, hiperparámetros, ANOVA).

## Credenciales

La extracción usa la API Socrata de datos.gov.co. Copia `.env.example` como `.env` y completa
`SOCRATA_APP_TOKEN`, `SOCRATA_USERNAME` y `SOCRATA_PASSWORD`. **Nunca** escribas credenciales en el
código: el hook pre-commit bloquea commits que las contengan.

## Gestión de versiones

Todo el flujo de git se maneja con [`scripts/versionado.sh`](scripts/versionado.sh):

```bash
bash scripts/versionado.sh ayuda                       # lista de comandos
bash scripts/versionado.sh status                      # rama, última versión, cambios pendientes
bash scripts/versionado.sh branch refactor-etl         # nueva rama feature/refactor-etl
bash scripts/versionado.sh save "refactor(etl): mover limpieza a src/"
bash scripts/versionado.sh finish                      # integrar la rama en main
bash scripts/versionado.sh snapshot v1.1.0 "Estructura de paquete reproducible"
bash scripts/versionado.sh publish                     # subir rama y tags a GitHub
```

Convención de versiones: **MAYOR** cuando cambian los resultados reportados (datos, muestra o
metodología), **MENOR** cuando se agregan análisis sin cambiar resultados previos y **PARCHE** para
correcciones que no alteran resultados. Los mensajes de commit siguen
[Conventional Commits](https://www.conventionalcommits.org/es/) (`feat`, `fix`, `refactor`, `data`,
`exp`, `docs`, …).

En Windows el script se ejecuta desde Git Bash, o desde PowerShell con
`& "C:\Program Files\Git\bin\bash.exe" scripts/versionado.sh <comando>`.
