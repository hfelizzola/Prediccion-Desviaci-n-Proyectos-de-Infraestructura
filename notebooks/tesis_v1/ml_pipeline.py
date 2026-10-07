# This is a machine learning pipeline script. 
# Import data science libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

#from transformData import dataCleaning
pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore", category=FutureWarning)

# Data Preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from yellowbrick.model_selection import FeatureImportances

# Model Selection & Validation
from sklearn.model_selection import train_test_split, cross_val_score

# Cross validation and hyperparameters
import optuna
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

# Pipeline & Imbalanced Data Handling
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import loguniform


# Model Evaluation Metrics
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
    RocCurveDisplay
)

# Feature Analysis
from sklearn.inspection import permutation_importance

# Natural Language Processing
import nltk

# Utils 
import warnings
import os
import json

def create_preprocessing_pipeline(numeric_columns, categorical_columns,
                                 num_strategy='median', cat_strategy='most_frequent'):
    """
    Crea un pipeline de preprocesamiento para características numéricas y categóricas.

    Parámetros:
    -----------
    numeric_columns : list
        Lista de nombres de columnas numéricas
    categorical_columns : list
        Lista de nombres de columnas categóricas
    num_strategy : str, opcional
        Estrategia de imputación para columnas numéricas (default: 'median')
    cat_strategy : str, opcional
        Estrategia de imputación para columnas categóricas (default: 'most_frequent')

    Returns:
    --------
    ColumnTransformer
        Pipeline de preprocesamiento configurado
    """
    # Pipeline para columnas numéricas
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=num_strategy, add_indicator=True)),  # Indicador para valores faltantes
        ('scaler', StandardScaler())  # Escalar después de imputar para evitar sesgos
    ])

    # Pipeline para columnas categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=cat_strategy, add_indicator=True)),  # Indicador para valores faltantes
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # sparse_output=False para compatibilidad
    ])

    # Combinar transformaciones
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_columns),
            ('cat', categorical_transformer, categorical_columns)
        ],
        remainder='passthrough'  # Mantener otras columnas que no estén en las listas
    )

    return preprocessor

def evaluate_model(model, X_test, y_test):
    """
    Evalúa un modelo entrenado en el conjunto de prueba.

    Parámetros:
    -----------
    model : objeto estimador entrenado
        El modelo a evaluar
    X_test : array-like
        Características de prueba
    y_test : array-like
        Etiquetas reales de prueba

    Returns:
    --------
    dict
        Diccionario con las métricas de evaluación
    """
    # Realizar predicciones
    y_pred = model.predict(X_test)

    # Para probabilidades (para ROC AUC)
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        has_proba = True
    except (AttributeError, IndexError):
        has_proba = False

    # Calcular métricas
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

    # Añadir ROC AUC si es posible calcularlo
    if has_proba and len(np.unique(y_test)) == 2:  # Solo para clasificación binaria
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)

    return metrics

def create_fold_scores_df(cv_scores, model_name, stage='S1'):
    """
    Crea un DataFrame con los resultados de cada fold en la validación cruzada.

    Parámetros:
    -----------
    cv_scores : dict
        Diccionario con los resultados de cross_validate
    model_name : str
        Nombre del modelo para la columna 'Model'
    stage : str, opcional
        Etapa del modelo para la columna 'Stage' (default: 'S1')

    Returns:
    --------
    DataFrame
        DataFrame con los resultados por fold
    """
    # Crear una lista para almacenar los resultados de cada fold
    fold_results = []

    # Identificar las métricas disponibles
    metrics = [key.replace('test_', '') for key in cv_scores.keys() if key.startswith('test_')]

    # Para cada fold
    for fold_idx in range(len(cv_scores['test_' + metrics[0]])):
        # Crear un diccionario para este fold
        fold_dict = {
            'Model': model_name,
            'Stage': stage,
            'Fold': fold_idx + 1  # Fold indexado desde 1
        }

        # Añadir cada métrica para este fold
        for metric in metrics:
            fold_dict[metric] = cv_scores['test_' + metric][fold_idx]

        fold_results.append(fold_dict)

    # Convertir la lista a DataFrame
    fold_scores_df = pd.DataFrame(fold_results)

    return fold_scores_df

def create_mean_std_scores_df(cv_scores, scoring):
    """
    Crea un DataFrame con las medias y desviaciones estándar de las métricas de validación cruzada.

    Parámetros:
    -----------
    cv_scores : dict
        Diccionario con los resultados de cross_validate
    scoring : list o dict
        Métricas evaluadas en cross_validate

    Returns:
    --------
    DataFrame
        DataFrame con las medias y desviaciones estándar de las métricas
    """
    # Preparar un diccionario para los resultados
    results = {}

    # Determinar las métricas disponibles
    if isinstance(scoring, dict):
        metrics = list(scoring.keys())
    else:
        metrics = scoring if isinstance(scoring, list) else [scoring]

    # Para cada métrica
    for metric in metrics:
        # Intentar obtener los resultados para esta métrica
        try:
            mean_key = f'test_{metric}'
            if mean_key in cv_scores:
                results[f'{metric}_mean'] = np.mean(cv_scores[mean_key])
                results[f'{metric}_std'] = np.std(cv_scores[mean_key])
        except KeyError:
            # Si la métrica no está disponible, ignorarla
            pass

    # Convertir a DataFrame
    mean_std_df = pd.DataFrame([results])

    return mean_std_df

# def plot_feature_importance(model, X, y=None, feature_names=None, n_top=10, figsize=(10, 6)):
#     """
#     Visualiza las características más importantes del modelo.

#     Parámetros:
#     -----------
#     model : objeto estimador entrenado
#         El modelo para el cual extraer importancias
#     X : array-like
#         Datos para calcular importancia de permutación si es necesario
#     y : array-like, opcional
#         Etiquetas para calcular importancia de permutación (requerido para este método)
#     feature_names : list, opcional
#         Nombres de las características (default: None)
#     n_top : int, opcional
#         Número de características principales a mostrar (default: 10)
#     figsize : tuple, opcional
#         Tamaño de la figura (default: (10, 6))

#     Returns:
#     --------
#     fig
#         Objeto de figura matplotlib
#     """
#     # Inicializar variables
#     importance = None
#     importance_type = "No disponible"

#     # Intentar diferentes métodos para obtener importancia de características
#     try:
#         # 1. Intentar con feature_importances_ (árboles, bosques, etc.)
        
#         if hasattr(model, 'steps'):
#             final_estimator = model.steps[-1][1]
#             if hasattr(final_estimator, 'feature_importances_'):
#                 importance = final_estimator.feature_importances_
#                 importance_type = "Feature Importance"
#             # Extraer el nombre de las características del preprocesador
#             if hasattr(model.named_steps['preprocessor'], 'transformers_'):
#                 feature_names = []
#                 for name, trans, _ in model.named_steps['preprocessor'].transformers_:
#                     if hasattr(trans, 'get_feature_names_out'):
#                         feature_names.extend(trans.get_feature_names_out(input_features=[name]))
#         else:
#             importance = model.feature_importances_
#             importance_type = "Feature Importance"
        
            
#         # 2. Intentar con coef_ (modelos lineales)
#         elif hasattr(model, 'coef_'):
#             # Para modelos lineales
#             if hasattr(model, 'steps'):
#                 final_estimator = model.steps[-1][1]
#                 if hasattr(final_estimator, 'coef_'):
#                     importance = np.abs(final_estimator.coef_).mean(axis=0) if final_estimator.coef_.ndim > 1 else np.abs(final_estimator.coef_)
#                     importance_type = "Coefficient Magnitude"
#             else:
#                 importance = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
#                 importance_type = "Coefficient Magnitude"

#         # 3. Si no hay métodos directos, usar importancia de permutación
#         if importance is None and y is not None:
#             try:
#                 # Utilizar importancia de permutación (requiere y)
#                 perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
#                 importance = perm_importance.importances_mean
#                 importance_type = "Permutation Importance"
#             except Exception as e:
#                 print(f"No se pudo calcular la importancia de permutación: {e}")
#                 return None
#         elif importance is None and y is None:
#             print("Se requiere el parámetro 'y' para calcular la importancia de permutación.")
#             return None

#         # Si no hay nombres de características, usar índices
#         if feature_names is None:
#             feature_names = [f"Feature {i}" for i in range(len(importance))]

#         # Asegurarse de que la longitud coincida
#         if len(feature_names) != len(importance):
#             print(f"Advertencia: La longitud de feature_names ({len(feature_names)}) no coincide con la longitud de importance ({len(importance)})")
#             # Ajustar la longitud si es necesario
#             if len(feature_names) > len(importance):
#                 feature_names = feature_names[:len(importance)]
#             else:
#                 feature_names = feature_names + [f"Feature {i}" for i in range(len(feature_names), len(importance))]

#         # Crear DataFrame con importancias
#         importance_df = pd.DataFrame({
#             'Feature': feature_names,
#             'Importance': importance
#         }).sort_values('Importance', ascending=False)

#         # Limitar al número superior especificado
#         if n_top is not None and n_top < len(importance_df):
#             importance_df = importance_df.head(n_top)

#         # Crear la figura
#         fig, ax = plt.subplots(figsize=figsize)

#         # Crear un gráfico de barras horizontales usando la nueva sintaxis de seaborn
#         # Corregido: usando hue en lugar de palette directamente
#         sns.barplot(x='Importance', y='Feature', hue='Feature', data=importance_df, ax=ax, legend=False)

#         # Añadir títulos y etiquetas
#         ax.set_title(f'Top {len(importance_df)} Features - {importance_type}', fontsize=14)
#         ax.set_xlabel('Importance', fontsize=12)
#         ax.set_ylabel('Feature', fontsize=12)

#         # Añadir valores en las barras
#         for i, v in enumerate(importance_df['Importance']):
#             ax.text(v + 0.001, i, f"{v:.4f}", va='center')

#         # Ajustar diseño
#         plt.tight_layout()

#         return fig

#     except Exception as e:
#         print(f"Error al crear gráfico de importancia de características: {e}")
#         return None

def train_tuning_evaluate_model(
    X_train,
    y_train,
    model,
    model_name="Model",
    stage=None,
    param_suggest_dict=None,
    X_test=None,
    y_test=None,
    n_trials=50,
    cv=5,
    scoring='roc_auc',
    direction='maximize',
    timeout=None,
    resampling_strategy=None,
    resampling_ratio=None,
    numeric_features=None,
    categorical_features=None,
    random_state=42,
    n_jobs=-1,
    plot_tuning_history=False
):
    """
    Realiza el entrenamiento, optimización de hiperparámetros y evaluación para un modelo de clasificación utilizando Optuna,
    con manejo de características categóricas y numéricas, y múltiples opciones de balanceo.

    Parámetros:
    -----------
    X_train : DataFrame
        Datos de entrenamiento con características.
    y_train : Series
        Variable objetivo.
    model : estimator
        Modelo de clasificación de scikit-learn.
    param_suggest_dict : dict
        Diccionario con funciones de sugerencia para Optuna.
        Ejemplo: {
            'max_depth': lambda trial: trial.suggest_int('max_depth', 3, 10),
            'learning_rate': lambda trial: trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        }
    X_test : DataFrame, default=None
        Datos de prueba con características. Si es None, se utilizará X_train.
    y_test : Series, default=None
        Variable objetivo para los datos de prueba. Si es None, se utilizará y_train.
    n_trials : int, default=50
        Número de pruebas para Optuna.
    cv : int, default=5
        Número de folds para validación cruzada.
    scoring : str, default='roc_auc'
        Métrica para evaluar el rendimiento del modelo.
    direction : str, default='maximize'
        Dirección de optimización: 'maximize' o 'minimize'.
    timeout : int, default=None
        Tiempo límite para la optimización (en segundos).
    resampling_strategy : str, default=None
        Estrategia de resampling:
        - Oversampling: 'smote', 'adasyn', 'random_over'
        - Undersampling: 'random_under', 'cluster_centroids'
        - Combinadas: 'smotetomek', 'smoteenn'
        - None: sin resampling
    resampling_ratio : float, default=None
        Proporción de clases deseada después del resampling.
    numeric_features : list, default=None
        Lista de nombres de características numéricas.
    categorical_features : list, default=None
        Lista de nombres de características categóricas.
    random_state : int, default=42
        Semilla para reproducibilidad.
    n_jobs : int, default=-1
        Número de trabajos paralelos.

    Retorna:
    --------
    dict
        Diccionario con los siguientes elementos:
        - 'best_estimator': Mejor estimador encontrado.
        - 'best_params': Mejores parámetros encontrados.
        - 'best_score': Mejor puntaje obtenido.
        - 'metrics': Diccionario con métricas del mejor modelo.
        - 'study': Objeto estudio de Optuna.
        - 'importance': Importancia de los parámetros.
        - 'results_df': DataFrame con todos los resultados de las pruebas.
    """


    # Si no se especifican las características, inferirlas
    if numeric_features is None and categorical_features is None:
        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # Crear transformadores para cada tipo de característica
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Preprocesamiento de características
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Configurar el resampling según la estrategia seleccionada
    resampler = None
    if resampling_strategy:
        if resampling_ratio is None:
            resampling_ratio = 'auto'

        # Estrategias de oversampling
        if resampling_strategy == 'smote':
            resampler = SMOTE(sampling_strategy=resampling_ratio, random_state=random_state)
        elif resampling_strategy == 'adasyn':
            resampler = ADASYN(sampling_strategy=resampling_ratio, random_state=random_state)
        elif resampling_strategy == 'random_over':
            resampler = RandomOverSampler(sampling_strategy=resampling_ratio, random_state=random_state)

        # Estrategias de undersampling
        elif resampling_strategy == 'random_under':
            resampler = RandomUnderSampler(sampling_strategy=resampling_ratio, random_state=random_state)
        elif resampling_strategy == 'cluster_centroids':
            resampler = ClusterCentroids(sampling_strategy=resampling_ratio, random_state=random_state)

        # Estrategias combinadas
        elif resampling_strategy == 'smotetomek':
            resampler = SMOTETomek(sampling_strategy=resampling_ratio, random_state=random_state)
        elif resampling_strategy == 'smoteenn':
            resampler = SMOTEENN(sampling_strategy=resampling_ratio, random_state=random_state)

    # Crear base pipeline para ser clonada en cada trial
    if resampler:
        base_pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('resampler', resampler),
            ('classifier', model)
        ])
    else:
        base_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

    # Manejar X_test e y_test cuando son None
    if X_test is None:
        X_test = X_train
    if y_test is None:
        y_test = y_train

    # Configurar validación cruzada estratificada
    stratified_cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    # Definir función objetivo para Optuna
    def objective(trial):
        # Clonar el modelo base para este trial
        pipeline = clone(base_pipeline)

        # Aplicar las funciones de sugerencia de parámetros
        params = {}
        for param_name, suggest_func in param_suggest_dict.items():
            params[param_name] = suggest_func(trial)

        # Configurar el modelo con los parámetros sugeridos
        for param_name, param_value in params.items():
            setattr(pipeline.named_steps['classifier'], param_name, param_value)

        # Evaluar el modelo con validación cruzada
        try:
            scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                scoring=scoring,
                cv=stratified_cv,
                n_jobs=n_jobs
            )
            return np.mean(scores)
        except Exception as e:
            # Manejar errores y asignar un valor punitivo
            print(f"Error en trial {trial.number}: {str(e)}")
            return float('-inf') if direction == 'maximize' else float('inf')

    # Crear el estudio de Optuna
    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )

    # Ejecutar la optimización
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=1,  # Ya usamos paralelismo en cross_val_score
        show_progress_bar=True
    )

    # Obtener los mejores parámetros
    best_params = study.best_params
    best_score = study.best_value

    # Aplicar los mejores parámetros al modelo
    best_pipeline = clone(base_pipeline)
    for param_name, param_value in best_params.items():
        setattr(best_pipeline.named_steps['classifier'], param_name, param_value)

    # Entrenar el modelo con los mejores parámetros
    best_pipeline.fit(X_train, y_train)

    # Hacer predicciones
    y_pred = best_pipeline.predict(X_test)
    try:
        y_prob = best_pipeline.predict_proba(X_test)[:, 1]
    except (AttributeError, IndexError):
        # Si el modelo no tiene predict_proba o solo tiene una clase
        y_prob = None

    # Calcular métricas
    metrics = {
        'model': model_name,
        'stage': stage,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
    }

    # Añadir ROC AUC si es posible
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_prob)

    # Obtener importancia de parámetros
    try:
        importance = optuna.importance.get_param_importances(study)
    except:
        importance = {}
        print("No se pudo calcular la importancia de los parámetros")

    # Crear DataFrame con los resultados de las pruebas
    results_df = study.trials_dataframe()

    # Generar un reporte del tuning
    print(f"\n===== {model_name} Tuning Report in {stage} =====")
    print(f"Best {scoring}: {best_score:.4f}")
    print("\nBest parameters:")
    for param, value in best_params.items():
        print(f"- {param}: {value}")
    print("\nParameter importance:")
    for param, importance in importance.items():
        print(f"- {param}: {importance:.4f}")

    # Gráfico de historia de optimización
    if plot_tuning_history:
        plt.figure(figsize=(5, 3))
        optuna.visualization.matplotlib.plot_optimization_history(study, target_name=scoring)
        plt.title("Optimization History")
        plt.tight_layout()
        plt.show()
      
    # Generar un informe del desempeño
    print(f"\n===== {model_name} Performance Report =====")
    print("\nTest Set Metrics:")
    for metric, value in metrics.items():
        if metric not in ['model', 'stage']:
            print(f"- {metric}: {value:.4f}")
    
    

    return {
        'best_estimator': best_pipeline,
        'best_params': best_params,
        'best_score': best_score,
        'metrics': metrics,
        'study': study,
        'importance': importance,
        'results_df': results_df
    }


# def train_evaluate_model(classifier, 
#                          X_train, 
#                          y_train, 
#                          X_test, 
#                          y_test,
#                          numeric_columns=None, 
#                          categorical_columns=None,
#                          preprocessor=None, 
#                          X_full=None, 
#                          y_full=None,
#                          model_name="Model", 
#                          resampling=True,
#                          resampling_strategy='smote',
#                          resampling_ratio=None,
#                          random_state=42,
#                          perform_cv=True, 
#                          cv=5, 
#                          stage='S1',
#                          scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
#                          plot_results=True,
#                          n_top_features=10):
#     """
#     Entrena y evalúa un modelo de machine learning con preprocesamiento,
#     balanceo opcional y validación cruzada.

#     Parámetros:
#     -----------
#     classifier : objeto estimador
#         El clasificador a entrenar (debe seguir la API de scikit-learn)
#     X_train : array-like
#         Características de entrenamiento
#     y_train : array-like
#         Etiquetas de entrenamiento
#     X_test : array-like
#         Características de prueba
#     y_test : array-like
#         Etiquetas de prueba
#     numeric_columns : list, opcional
#         Lista de nombres de columnas numéricas (requerido si preprocessor es None)
#     categorical_columns : list, opcional
#         Lista de nombres de columnas categóricas (requerido si preprocessor es None)
#     preprocessor : ColumnTransformer, opcional
#         Pipeline de preprocesamiento preconfigurado (si no se proporciona, se creará uno)
#     X_full : array-like, opcional
#         Conjunto completo de características para validación cruzada (default: None)
#     y_full : array-like, opcional
#         Conjunto completo de etiquetas para validación cruzada (default: None)
#     model_name : str, opcional
#         Nombre del modelo para los reportes (default: "Model")
#     resampling : bool, opcional
#         Indica si se debe aplicar balanceo de clases (default: True)
#     resampling_strategy : str, opcional
#         Estrategia de balanceo de clases (default: 'smote')
#     random_state : int, opcional
#         Semilla para reproducibilidad (default: 42)
#     perform_cv : bool, opcional
#         Indica si se debe realizar validación cruzada (default: True)
#     cv : int, opcional
#         Número de folds para validación cruzada (default: 5)
#     stage : str, opcional
#         Etapa del modelo para reportes (default: 'S1')
#     scoring : list, opcional
#         Métricas a evaluar en la validación cruzada
#         (default: ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
#     plot_results : bool, opcional
#         Indica si se deben generar gráficos de rendimiento (default: True)
#     n_top_features : int, opcional
#         Número de características principales a mostrar en el gráfico de importancia (default: 10)

#     Returns:
#     --------
#     dict
#         Un diccionario con el modelo entrenado, métricas de evaluación y
#         resultados de validación cruzada
#     """
#     # Validación de argumentos
#     if preprocessor is None:
#         if numeric_columns is None or categorical_columns is None:
#             raise ValueError("Si no se proporciona un preprocessor, se deben especificar "
#                             "las columnas numéricas y categóricas")
#         preprocessor = create_preprocessing_pipeline(numeric_columns, categorical_columns)

#     # Verificar si X_full e y_full fueron proporcionados, si no, usar X_train e y_train
#     if X_full is None or y_full is None:
#         X_full, y_full = X_train.copy(), y_train.copy()

#     # Obtener nombres de características si es posible (para gráficos de importancia)
#     try:
#         if hasattr(X_train, 'columns'):
#             feature_names = X_train.columns.tolist()
#         else:
#             feature_names = None
#     except:
#         feature_names = None

#     # Configurar el resampling según la estrategia seleccionada
#     resampler = None
#     if resampling:
#         if resampling_ratio is None:
#             resampling_ratio = 'auto'
            
#         # Estrategias de oversampling
#         if resampling_strategy == 'smote':
#             resampler = SMOTE(sampling_strategy=resampling_ratio, random_state=random_state)
#         elif resampling_strategy == 'adasyn':
#             resampler = ADASYN(sampling_strategy=resampling_ratio, random_state=random_state)
#         elif resampling_strategy == 'random_over':
#             resampler = RandomOverSampler(sampling_strategy=resampling_ratio, random_state=random_state)
        
#         # Estrategias de undersampling
#         elif resampling_strategy == 'random_under':
#             resampler = RandomUnderSampler(sampling_strategy=resampling_ratio, random_state=random_state)
#         elif resampling_strategy == 'cluster_centroids':
#             resampler = ClusterCentroids(sampling_strategy=resampling_ratio, random_state=random_state)
        
#         # Estrategias combinadas
#         elif resampling_strategy == 'smotetomek':
#             resampler = SMOTETomek(sampling_strategy=resampling_ratio, random_state=random_state)
#         elif resampling_strategy == 'smoteenn':
#             resampler = SMOTEENN(sampling_strategy=resampling_ratio, random_state=random_state)

#     # Crear pipeline con o sin SMOTE
#     if resampling:
#         # Usar ImbPipeline para compatibilidad con SMOTE
#         model = ImbPipeline(steps=[
#             ('preprocessor', preprocessor),
#             ('resampler', resampler),
#             ('classifier', classifier)
#         ])
#     else:
#         model = Pipeline(steps=[
#             ('preprocessor', preprocessor),
#             ('classifier', classifier)
#         ])

#     # Entrenar el modelo
#     try:
#         model.fit(X_train, y_train)
#         print(f"Modelo {model_name} entrenado exitosamente.")

#         # Información sobre el balanceo de clases
#         classes, counts = np.unique(y_train, return_counts=True)
#         print(f"Distribución de clases en conjunto de entrenamiento: {dict(zip(classes, counts))}")

#     except Exception as e:
#         print(f"Error al entrenar el modelo {model_name}: {e}")
#         raise

#     # Evaluar el modelo en el conjunto de prueba
#     model_metrics = evaluate_model(model, X_test, y_test)

#     # Inicializar diccionario de resultados
#     results = {
#         'model': model,
#         'metrics': model_metrics,
#         'model_name': model_name,
#         'stage': stage,
#     }

#     # Realizar validación cruzada si se especifica
#     if perform_cv:
#         try:
#             cv_scores = cross_validate(model, X_full, y_full, cv=cv, scoring=scoring)
#             fold_scores_df = create_fold_scores_df(cv_scores, model_name, stage=stage)
#             mean_std_scores_df = create_mean_std_scores_df(cv_scores, scoring)

#             results['cv_scores'] = cv_scores
#             results['fold_scores_df'] = fold_scores_df
#             results['mean_std_scores_df'] = mean_std_scores_df
#         except Exception as e:
#             print(f"Error en la validación cruzada: {e}")
#             print("Continuando sin validación cruzada...")

#     # Generar un informe del desempeño
#     print(f"\n===== {model_name} Performance Report =====")
#     print("\nTest Set Metrics:")
#     for metric, value in model_metrics.items():
#         print(f"{metric}: {value:.4f}")

#     if perform_cv and 'mean_std_scores_df' in results:
#         print("\nCross-Validation Results:")
#         print(mean_std_scores_df)

#     # Generar gráficos si se solicita
#     if plot_results:
#         try:
#             # Crear una figura con tres subplots (2x2 grid)
#             fig = plt.figure(figsize=(18, 12))

#             # Definir la estructura de subplots
#             gs = fig.add_gridspec(2, 2)
#             ax1 = fig.add_subplot(gs[0, 0])  # Métricas de test
#             ax2 = fig.add_subplot(gs[0, 1])  # Resultados de CV por fold
#             ax3 = fig.add_subplot(gs[1, :])  # Importancia de características (fila completa abajo)

#             # 1. Gráfico de barras para métricas de test
#             metrics_names = list(model_metrics.keys())
#             metrics_values = list(model_metrics.values())
#             bars = ax1.bar(metrics_names, metrics_values, color='skyblue')

#             # Añadir valores sobre las barras
#             for bar in bars:
#                 height = bar.get_height()
#                 ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
#                          f'{height:.3f}', ha='center', va='bottom')

#             ax1.set_title(f'{model_name} Test Metrics', fontsize=14)
#             ax1.set_ylim(0, 1.1)
#             ax1.grid(axis='y', linestyle='--', alpha=0.7)

#             # 2. Si hay resultados de CV, graficar la distribución de métricas por fold
#             if 'fold_scores_df' in results:
#                 # Excluir columnas no numéricas para el gráfico de puntos
#                 numeric_cols = results['fold_scores_df'].drop(columns=['Model', 'Stage', 'Fold']).columns

#                 # Configurar datos para el gráfico
#                 melt_df = pd.melt(results['fold_scores_df'],
#                                   id_vars=['Model', 'Stage', 'Fold'],
#                                   value_vars=numeric_cols,
#                                   var_name='Metric',
#                                   value_name='Score')

#                 # Gráfico de pointplot con parámetros actualizados
#                 # Corregido: usando errorbar='sd' en lugar de ci='sd'
#                 # Corregido: usando linestyle='none' en lugar de join=False
#                 sns.pointplot(x='Metric', y='Score', data=melt_df, errorbar='sd',
#                              linestyle='none', color='darkblue', ax=ax2)

#                 # Añadir puntos individuales para cada fold
#                 sns.stripplot(x='Metric', y='Score', data=melt_df,
#                              jitter=True, size=8, color='orange', alpha=0.6, ax=ax2)

#                 ax2.set_title(f'{model_name} Cross-Validation Results by Fold', fontsize=14)
#                 ax2.set_ylim(0, 1.1)
#                 ax2.grid(axis='y', linestyle='--', alpha=0.7)

#                 # Añadir valores medios sobre los puntos
#                 for i, metric in enumerate(numeric_cols):
#                     mean_val = melt_df[melt_df['Metric'] == metric]['Score'].mean()
#                     ax2.text(i, mean_val + 0.02, f'{mean_val:.3f}',
#                             ha='center', va='bottom', fontweight='bold')

#             # 3. Graficar importancia de características
#             try:
#                 # Para el gráfico de importancia, necesitamos tanto X como y
#                 importance_fig = plot_feature_importance(model, X_train, y_train,
#                                                        feature_names=feature_names,
#                                                        n_top=n_top_features)

#                 if importance_fig is not None:
#                     # Crear un gráfico de importancia directamente en ax3 en lugar de copiar
#                     importance_data = []

#                     # Extraer la importancia como lo hace la función plot_feature_importance
#                     importance = None

#                     # Intentar diferentes métodos para obtener importancia de características
#                     if hasattr(model, 'feature_importances_'):
#                         if hasattr(model, 'steps'):
#                             final_estimator = model.steps[-1][1]
#                             if hasattr(final_estimator, 'feature_importances_'):
#                                 importance = final_estimator.feature_importances_
#                         else:
#                             importance = model.feature_importances_
#                     elif hasattr(model, 'coef_'):
#                         if hasattr(model, 'steps'):
#                             final_estimator = model.steps[-1][1]
#                             if hasattr(final_estimator, 'coef_'):
#                                 importance = np.abs(final_estimator.coef_).mean(axis=0) if final_estimator.coef_.ndim > 1 else np.abs(final_estimator.coef_)
#                         else:
#                             importance = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)

#                     if importance is None and y_train is not None:
#                         try:
#                             perm_importance = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42)
#                             importance = perm_importance.importances_mean
#                         except Exception as e:
#                             print(f"No se pudo calcular la importancia de permutación: {e}")

#                     if importance is not None:
#                         # Si tenemos importancia, crear el gráfico directamente en ax3
#                         if feature_names is None:
#                             feature_names = [f"Feature {i}" for i in range(len(importance))]

#                         # Asegurarse de que la longitud coincida
#                         if len(feature_names) != len(importance):
#                             if len(feature_names) > len(importance):
#                                 feature_names = feature_names[:len(importance)]
#                             else:
#                                 feature_names = feature_names + [f"Feature {i}" for i in range(len(feature_names), len(importance))]

#                         # Crear DataFrame
#                         importance_df = pd.DataFrame({
#                             'Feature': feature_names,
#                             'Importance': importance
#                         }).sort_values('Importance', ascending=False)

#                         # Tomar los n_top principales
#                         if n_top_features is not None and n_top_features < len(importance_df):
#                             importance_df = importance_df.head(n_top_features)

#                         # Graficar importancia directamente en ax3
#                         sns.barplot(x='Importance', y='Feature', hue='Feature', data=importance_df, ax=ax3, legend=False)

#                         # Añadir títulos y etiquetas
#                         ax3.set_title(f'Top {len(importance_df)} Features Importance', fontsize=14)
#                         ax3.set_xlabel('Importance', fontsize=12)
#                         ax3.set_ylabel('Feature', fontsize=12)

#                         # Añadir valores en las barras
#                         for i, v in enumerate(importance_df['Importance']):
#                             ax3.text(v + 0.001, i, f"{v:.4f}", va='center')
#                     else:
#                         ax3.text(0.5, 0.5, "No se pudo calcular la importancia de características",
#                                 ha='center', va='center', fontsize=14)

#                     # Cerrar figura anterior si existe
#                     plt.close(importance_fig)
#                 else:
#                     ax3.text(0.5, 0.5, "No se pudo generar el gráfico de importancia de características",
#                             ha='center', va='center', fontsize=14)
#             except Exception as e:
#                 print(f"Error al generar gráfico de importancia: {e}")
#                 ax3.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')

#             # Ajustar el diseño y mostrar
#             fig.suptitle(f'{model_name} - Performance Overview', fontsize=16)
#             plt.tight_layout()
#             plt.subplots_adjust(top=0.94)  # Ajustar para el título principal
#             plt.show()

#             # Guardar la figura en el diccionario de resultados
#             results['performance_figure'] = fig

#         except Exception as e:
#             print(f"Error al generar gráficos: {e}")

#     return results

def consolidate_results(*results):
    """
    Consolidate model results, calculate mean and std for numeric columns, and return a DataFrame.
    Each result should be a dict with a 'metrics' key and 'Model' and 'Stage' info.
    """
    dfs = []
    for res in results:
        res['metrics'].update({'Model': res.get('model', res.get('Model', ''))})
        res['metrics'].update({'Stage': res.get('stage', res.get('Stage', ''))})
        dfs.append(pd.DataFrame(res['metrics'], index=[0]))
    scores_df = pd.concat(dfs, ignore_index=True)
    # Calcular la media para las columnas numéricas
    mean_values = scores_df.select_dtypes(include=['number']).mean()
    mean_row = pd.Series({'Stage': scores_df['Stage'].iloc[0], 'Model': 'Mean'})
    for col in mean_values.index:
        mean_row[col] = mean_values[col]
    # Calcular la desviación estándar para las columnas numéricas
    std_values = scores_df.select_dtypes(include=['number']).std()
    std_row = pd.Series({'Stage': scores_df['Stage'].iloc[0], 'Model': 'Std'})
    for col in std_values.index:
        std_row[col] = std_values[col]
    # Añadir las filas al DataFrame
    scores_df = pd.concat([scores_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    return scores_df