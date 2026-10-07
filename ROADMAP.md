# Roadmap: de la versión de tesis (v1.0.0) al artículo

Objetivo: convertir el proyecto en un pipeline **reproducible de punta a punta** (un comando regenera
datos, modelos, tablas y figuras) con una metodología que resista la revisión de una revista de
primer nivel en infraestructura (p. ej. *J. Construction Engineering and Management*, *Automation in
Construction*, *J. Infrastructure Systems*, *J. Management in Engineering*, *IJPM*).

Cada fase termina en un tag con `scripts/versionado.sh snapshot`. La regla es **separar
refactorización de cambios metodológicos**: primero se reorganiza el código demostrando que los
resultados de la v1 se reproducen (v1.x) y luego se corrige la metodología (v2.0.0). Así cualquier
diferencia de resultados se puede atribuir a una decisión metodológica concreta y no a un cambio
accidental de código.

---

## 1. Diagnóstico

### 1.1 Metodología (prioridad para revisores)

| # | Hallazgo | Dónde | Riesgo |
|---|----------|-------|--------|
| M1 | **El filtro de atípicos usa las variables respuesta.** El Isolation Forest incluye `costDeviationPerc`, `timeDeviationPerc`, `haveCostDeviation`, etc., así que la muestra se selecciona según el resultado. Además no tiene `random_state`. | `featureEngineering_V2.ipynb`, celda 35 | Sesgo de selección, no reproducible |
| M2 | **Fuga temporal en el histórico de la entidad.** `histPctDesv*` y `histAvg*Deviation` usan contratos firmados hasta t-1, pero la desviación de un contrato solo se conoce cuando termina. Un contrato de 2 años firmado en t-1 aún no tiene resultado al firmar el contrato focal. | `featureEngineering_V2.ipynb`, `calcular_historico` | Fuga de información futura |
| M3 | **Imputación con información futura.** El HHI se imputa con la mediana de la entidad sobre todos los años y TerriData con el promedio de todos los años (incluidos los posteriores). | `integrar_hhi_*`, `completar_terridata` | Fuga temporal |
| M4 | **La selección de variables ("torneo") usa todo el dataset**, incluido el test, y solo con `haveCostDeviation`, aunque se aplica también a tiempo. | `train_tuning_optuna_evaluate_v2.ipynb`, celdas 37-51 | Sesgo optimista |
| M5 | **Evaluación con un único split aleatorio 70/30 (semilla 42)**, sin intervalos de confianza. Contratos de la misma entidad y de años posteriores quedan en entrenamiento, lo que no refleja el uso real (predecir contratos futuros). | `train_tuning_optuna_evaluate_v2.ipynb` | Métricas infladas y sin incertidumbre |
| M6 | **ANOVA sobre 3 folds del dataset completo** con hiperparámetros ajustados en el train. Los folds no son independientes (supuesto violado) y el test participó en la CV. | `model_analysis_V2.ipynb`, celdas 25-28 | Inferencia inválida |
| M7 | **SMOTEENN después del One-Hot** genera dummies fraccionarias y descalibra las probabilidades. Precision, recall y accuracy se reportan con umbral 0,5 sobre probabilidades re-balanceadas. | `ml_pipeline.train_tuning_evaluate_model` | Métricas de umbral engañosas |
| M8 | **Bug en el espacio condicional del Random Forest.** Si Optuna elige `max_features_float`, el reentrenamiento final ignora ese valor y usa el defecto (`sqrt`). Ocurrió en *time / S1* (0,108 → `sqrt`). Además usa `setattr` en vez de `set_params`. | `ml_pipeline.py` L578-584 | El modelo reportado no es el ajustado |
| M9 | **SHAP sobre un modelo distinto al evaluado**: se reentrena con todos los datos (incluido el test), sin re-muestreo y con otro preprocesamiento. El notebook usa `xgb_model_cost`, cuya definición está comentada. | `model_analysis_V2.ipynb`, celdas 54-61 | Interpretación no trazable |
| M10 | **Muestra final de 4.538 de 28.458 contratos (16%)** por filtros (`contractValueMw ≥ 1000`, `totalCumContracts ≥ 5`) aplicados en los notebooks de modelado, no en el pipeline de datos. Hay 2 contratos de 2024. | notebooks de entrenamiento | Validez externa; hay que reportar el flujo de la muestra |
| M11 | **Entidad identificada por nombre**: 633 `buyerName` frente a 557 NIT. El histórico y el HHI parten la misma entidad en varias. | `calcular_historico`, `calcular_hhi` | Variables de entidad ruidosas |
| M12 | **Año electoral solo presidencial** (2010, 2014, 2018, 2022). Para compradores territoriales importan más las elecciones locales (2011, 2015, 2019, 2023) y la Ley de Garantías. | `featureEngineering_V2.ipynb`, celda 52 | Variable mal especificada |
| M13 | **Deflactación con salario mínimo.** El SMMLV crece por encima de la inflación (16% en 2023), lo que mezcla crecimiento real con precios. | `ETL.ipynb`, celda 105 | Comparabilidad entre años |
| M14 | **Grupo G3 (contratista) débil**: solo `isSME`, `isBusinessGroup` y departamento. Falta el desempeño histórico del contratista y de los integrantes de consorcios. `Data/entity_resolution.py` existe pero no está integrado. | — | Pregunta de investigación 2 subatendida |
| M15 | **Cruce con TerriData por nombre** de departamento y municipio, sin códigos DIVIPOLA ni reporte de la tasa de cruce. | `featureEngineering_V2.ipynb`, celdas 81-85 | Faltantes silenciosos |
| M16 | **`uid` duplicados que se multiplican** *(hallado en la Fase 1)*. `contratosSECOPCleaned.csv` tiene `uid` repetidos (42 en la muestra filtrada). La unión del HHI por `uid` convierte *k* copias en *k²* filas: 28.378 contratos se vuelven 28.458 filas en `trainData.csv` (80 duplicados). | `features/owner.py` (`add_owner_features`) | Observaciones repetidas, posible fuga entre train y test |

### 1.2 Reproducibilidad e ingeniería

| # | Hallazgo |
|---|----------|
| R1 | La lógica vive en 15 notebooks con versiones paralelas (`_V2`, `-Pycaret`, `v2`). `transformData.py` y `ETL.ipynb` implementan la misma limpieza con diferencias (`contractDuration` frente a `contractDurationDays`): no hay una única fuente de verdad. |
| R2 | Rutas fijas a Colab (`drive.mount`, `os.chdir`), `!pip install` dentro de las celdas, `%autoreload`. ETL escribe en la raíz y luego lee de `Data/`. |
| R3 | Los notebooks no corren de arriba a abajo (contadores de ejecución vacíos en ETL y variables definidas en celdas comentadas). |
| R4 | `requirements.txt` es un `pip freeze` de Colab (~600 paquetes: CUDA, torch, tensorflow). Faltan `sodapy`, `unidecode`, `fuzzywuzzy`, `pystout`, `linearmodels`. **Con pandas 3 las columnas de texto son `string` y `select_dtypes(['object'])` las descartaría sin aviso.** |
| R5 | Las consultas SoQL tienen `LIMIT 100000` sin paginación (truncamiento silencioso si crece la fuente) y no se registra la fecha de extracción. SECOP cambia retroactivamente, así que hace falta un snapshot fechado. |
| R6 | Las listas de exclusión por palabra clave están duplicadas entre `querySECOPIProcesos.sql` y `querySECOPIIContratos.sql`, y las reglas manuales (`uid_to_remove`, `dataCorrections.xlsx`) no documentan su motivo. |
| R7 | Los resultados se guardan como diccionarios joblib con el estudio Optuna y el pipeline completos (hasta 62 MB por archivo), sin la configuración ni el commit que los produjo. |

### 1.3 Calidad de código

- Bloques duplicados: los mismos espacios de búsqueda y bucles se repiten 8 veces (4 modelos × 2 respuestas).
- `ml_pipeline.py`: ~600 líneas comentadas; importaciones pesadas sin usar (`nltk`, `yellowbrick`, `lightgbm`, `TfidfVectorizer`); `Pipeline` de sklearn sombreado por el de imblearn; `except:` desnudo; `consolidate_results` muta su entrada.
- Bucles `iterrows`/`apply` fila a fila (`analizar_multas` es O(entidades × años × multas) y `calcular_historico` itera por entidad y año) que se reemplazan por `groupby` + `merge_asof`.
- Columnas con sufijo de grupo en el nombre (`contractValueMw(G1)`): son frágiles. Conviene un registro de grupos en configuración.
- `print` en vez de `logging`, semillas dispersas, sin pruebas automatizadas.

---

## 2. Estructura objetivo

```
├── README.md · CHANGELOG.md · ROADMAP.md · CITATION.cff · LICENSE
├── pyproject.toml            # dependencias mínimas fijadas + ruff + pytest (lock con uv)
├── configs/
│   ├── data.yaml             # años, filtros de muestra, umbrales, fuentes y fecha de corte
│   ├── features.yaml         # grupos G1–G4 y variables de cada uno
│   └── experiments.yaml      # escenarios S1–S5, modelos, espacios de búsqueda, validación
├── sql/                      # consultas SoQL (exclusiones generadas desde config)
├── data/                     # fuera de git (manifiesto o DVC)
│   ├── raw/        (snapshot fechado de la API, inmutable)
│   ├── external/   (TerriData, salario mínimo, índices de costos, calendario electoral)
│   ├── manual/     (dataCorrections.xlsx con columna "motivo"; en git)
│   ├── interim/ · processed/
├── src/secop_dev/
│   ├── config.py · logging.py
│   ├── data/       extract.py · clean.py · integrate.py · validate.py (pandera)
│   ├── features/   project.py · buyer_history.py · contractor_history.py · territorial.py · build.py
│   ├── models/     pipelines.py · search_spaces.py · train.py · evaluate.py
│   ├── interpret/  shap_analysis.py
│   ├── stats/      compare.py  (Nadeau-Bengio, DeLong, Friedman-Nemenyi)
│   └── viz/        figures.py  (estilo de revista)
├── scripts/                  # CLI delgadas: extract · build_features · train · evaluate · figures · versionado.sh
├── notebooks/
│   ├── 01_eda.ipynb … 05_interpretacion.ipynb   (solo importan src/, sin lógica propia)
│   └── tesis_v1/            (notebooks originales, congelados)
├── reports/  figures/ · tables/  (generados)
├── tests/                    # unitarios + regresión contra la v1
└── dvc.yaml (o Makefile)     # DAG: extract → clean → features → train → evaluate → figures
```

---

## 3. Fases

### Fase 0 — Congelar la versión de la tesis → `v1.0.0` ✅
- [x] Eliminar credenciales del código (variables de entorno).
- [x] `.gitignore`, `.gitattributes`, hook pre-commit, `scripts/versionado.sh`.
- [x] Manifiesto SHA-256 de datos y modelos; tag `v1.0.0`.
- [x] Push a GitHub privado (`hfelizzola/Prediccion-Desviaci-n-Proyectos-de-Infraestructura`).
- [x] Subir `v1.0.0-artifacts.tar.gz` como asset del release `v1.0.0` en GitHub.
- [x] Rotar la contraseña de datos.gov.co (estuvo en texto plano en Drive).

### Fase 1 — Fundaciones de ingeniería → `v1.1.0` (sin cambiar resultados)
- [x] `pyproject.toml` con dependencias fijadas a las versiones de la tesis (pandas 2.2.2, scikit-learn 1.6.1, xgboost 2.1.4, optuna 4.3.0, shap 0.47.2…) y `uv.lock`. Colab instala exactamente el lock (`uv sync --frozen`, Python 3.11).
- [x] Notebooks y scripts originales en `notebooks/tesis_v1/` (con `git mv`); consultas en `sql/`; tablas LaTeX en `reports/tesis_v1/`.
- [x] Paquete `src/secop_dev/` en inglés: `data/` (extracción, integración, limpieza, multas, TerriData), `features/`, `models/`, `analysis/` (ANOVA, SHAP, comparación con v1) y CLI `secop-dev`.
- [x] `configs/data.yaml`, `features.yaml` y `experiments.yaml` con todos los parámetros de la tesis.
- [x] **Prueba de regresión de datos**: el paquete reproduce **exactamente** `contratosSECOP.csv`, `contratosSECOPCleaned.csv`, `analisis_multas_por_entidad.csv` y `trainData.csv` (`pytest -m regression`). Los atípicos del Isolation Forest sin semilla quedaron congelados en `configs/frozen/v1_outlier_uids.csv`.
- [ ] **Reproducción de métricas**: correr `notebooks/colab_pipeline.ipynb` (RF, XGBoost y KNN) y revisar `comparison_v1.csv`; luego `snapshot v1.1.0`. La regresión logística ya se verificó localmente: 9 de 10 combinaciones idénticas a la tesis y *costo/S4* a ≤ 0,22 puntos (orden de variables).
- [x] `ruff`, `pytest` (24 pruebas), `.pre-commit-config.yaml` con `nbstripout`; el hook de `versionado.sh` los ejecuta si `pre-commit` está disponible.
- [x] `logging`; cada corrida guarda modelo, `best_params.json`, `metrics.json`, `trials.csv` y `run_metadata.json` (commit, hash de configuración, versiones). Entrenamiento reanudable.
- [x] Colab: `notebooks/colab_pipeline.ipynb` clona el repo, instala el lock y lee los datos de Drive.
- [x] Multas vectorizadas (de bucle fila a fila a búsqueda binaria, resultado idéntico).

### Fase 2 — Pipeline de datos reproducible → `v1.2.0`
- [ ] Extracción paginada (`$limit`/`$offset`) con snapshot fechado en `data/raw/<fecha>/` y metadatos (fecha, consulta, número de registros, hash).
- [ ] Generar las cláusulas `NOT LIKE` desde una sola lista en `configs/data.yaml` y compartirla entre SECOP I y II.
- [ ] `dataCorrections.xlsx` y `uid_to_remove` con columna **motivo** y fuente de verificación.
- [ ] Validación de esquemas con `pandera` (tipos, rangos, unicidad de `uid`, tasas de cruce mínimas).
- [ ] Tabla de **flujo de la muestra** (estilo CONSORT): registros extraídos → tras cada filtro → muestra final, por fuente y año.
- [ ] DAG con DVC (`dvc repro`) o Makefile: un comando regenera todo. DVC con remoto en Google Drive reemplaza al manifiesto manual.
- [ ] Vectorizar `cumulative_history` (`groupby` + `merge_asof`); hoy toma ~45 s de los ~50 s de `secop-dev features`.

### Fase 3 — Corrección metodológica → `v2.0.0` (cambian los resultados)
Cada punto va en su propio commit `exp(...)`, con una tabla de sensibilidad frente a la v1.
- [ ] **M1**: detección de atípicos solo con variables ex ante, con semilla fija; reportar la sensibilidad con y sin el filtro.
- [ ] **M2 / M3**: indicadores históricos *as-of*: solo contratos **terminados** antes de la fecha de firma del contrato focal (`merge_asof` por fecha). Imputación con el último valor disponible ≤ t-1 (sin información futura) y dentro del pipeline.
- [ ] **M11**: histórico, HHI y multas indexados por NIT normalizado, no por nombre.
- [ ] **M16**: deduplicar contratos por `uid` (regla documentada) antes de construir variables; unir por claves únicas.
- [ ] **M12**: calendario electoral nacional y local + ventana de la Ley de Garantías (4 meses antes de cada elección).
- [ ] **M13**: precios constantes con un índice de costos de construcción (ICOCED/ICCP, DANE); SMMLV como análisis de robustez.
- [ ] **M15**: cruces con TerriData por código DIVIPOLA, reportando la tasa de cruce.
- [ ] **M14**: historial del contratista (desviaciones previas, experiencia, número de contratos activos), con resolución de entidades y desagregación de consorcios y uniones temporales.
- [ ] Documentar la definición de la respuesta: tope legal del 50% en adiciones (Ley 80, art. 40, par.), tratamiento de desviaciones negativas, conversión de meses a días.

### Fase 4 — Evaluación de nivel revista → `v2.1.0`
- [ ] **Validación temporal** como diseño principal: entrenar con 2015–2020, validar con 2021 y probar con 2022–2023 (out-of-time). Como robustez, `GroupKFold` por entidad (entidades nunca vistas).
- [ ] **Ajuste anidado**: la selección de variables y de hiperparámetros solo dentro de los folds de entrenamiento (corrige M4). Optuna con `set_params` sobre el pipeline y espacios condicionales bien mapeados (corrige M8).
- [ ] **Desbalance** (cost ≈ 33% positivos, time ≈ 49%): `class_weight`/`scale_pos_weight` en lugar de SMOTEENN; si se re-muestrea, con SMOTENC y dentro del fold (corrige M7).
- [ ] **Métricas**: ROC-AUC, **PR-AUC**, **Brier**, curvas de calibración (y calibración isotónica si hace falta). Umbral elegido en validación según una función de costo del gestor, no 0,5.
- [ ] **Línea base**: tasa histórica de la entidad y regresión logística con variables de G1, contra las que se compara cualquier mejora.
- [ ] Modelos: agregar LightGBM/CatBoost (CatBoost maneja categóricas de alta cardinalidad como el departamento) y EBM (*Explainable Boosting Machine*) como modelo interpretable por diseño, que encaja con el título del artículo.
- [ ] **Incertidumbre e inferencia** (corrige M6): intervalos bootstrap en el test temporal; CV repetida 5×5 con *corrected resampled t-test* (Nadeau & Bengio, 2003) o t-test bayesiano correlacionado (Benavoli et al., 2017); DeLong para comparar AUC; Friedman + Nemenyi entre modelos.
- [ ] **Interpretabilidad** (corrige M9): SHAP del pipeline final evaluado y calculado sobre el test; SHAP agregado por grupo G1–G4 (responde directamente a la pregunta 2); gráficos de dependencia e interacciones; estabilidad del ranking entre folds y semillas; *grouped permutation importance* como contraste.
- [ ] Respuesta continua u ordinal (magnitud de la desviación, `costDeviationLevel` ya existe) como extensión: muchos revisores preguntan por el "cuánto", no solo el "si".
- [ ] Opcional: variables de texto del objeto del contrato (TF-IDF o embeddings en español) en lugar de las regex de `workCategory`.

### Fase 5 — Paquete de publicación → `v3.0.0` / tag `paper-<revista>-<ronda>`
- [ ] `scripts/make_figures.py` y `make_tables.py`: todas las figuras y tablas del manuscrito se generan desde los resultados (con estilo coherente y exportación a PDF/LaTeX).
- [ ] Reporte con la estructura de TRIPOD+AI adaptada: fuente de datos, flujo de la muestra, manejo de faltantes, validación y calibración.
- [ ] Model card y datasheet del dataset.
- [ ] `CITATION.cff`, `LICENSE` (MIT para el código y CC-BY-4.0 para los datos derivados), release en Zenodo con DOI (código y datos procesados) y sección *Data and code availability*.
- [ ] Tag por ronda de revisión (`paper-jcem-r1`, `paper-jcem-r2`) para poder responder a los revisores con el estado exacto del código.

---

## 4. Flujo de trabajo con git en cada fase

```bash
bash scripts/versionado.sh branch fase1-estructura          # rama de trabajo
# ... cambios ...
bash scripts/versionado.sh save "refactor(features): mover calcular_historico a src/"
bash scripts/versionado.sh save "test(features): regresión contra trainData v1"
bash scripts/versionado.sh finish                           # merge a main
bash scripts/versionado.sh snapshot v1.1.0 "Estructura de paquete; resultados idénticos a v1.0.0"
bash scripts/versionado.sh publish
bash scripts/versionado.sh archive v1.1.0                   # solo si cambiaron datos o modelos
```

Tipos de commit sugeridos: `feat`, `fix`, `refactor`, `test`, `docs`, `data` (cambios en datos o
extracción), `exp` (cambios metodológicos o experimentos que alteran resultados), `build`, `chore`.
