# gpu_tools.py - Simplified for beginners
from __future__ import annotations
import os
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR, LinearSVR, LinearSVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import optuna

import warnings
warnings.filterwarnings("ignore", message="X has feature names", category=UserWarning)
warnings.filterwarnings('ignore', message='This Pipeline instance is not fitted yet', category=FutureWarning)

def _try_free_memory():
    """Active GPU Memory Management"""
    try:
        import cupy
        # Free all GPU memory pools
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass

def load_data(data_path: str, target: str = None) -> Tuple[pd.DataFrame, str]:
    """Load CSV or Parquet file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot find file: {data_path}")
    
    # Load the file
    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    return df

def build_preprocessing_pipeline(df, target: str):
    """Build a pipeline to clean and prepare the data."""
    X = df.drop(columns=[target])
    y = df[target]

    # Find numeric and categorical columns
    numeric_columns = []
    categorical_columns = []
    
    for column in X.columns:
        if X[column].dtype in ['int64', 'float64']:
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    X_pd = X
    y_np = y.values

    # Data preprocessing pipeline 
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('float32', Float32Transformer(accept_sparse=False)),
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore',
                                 sparse_output=False,
                                 dtype=np.float32)),
    ])

    # Dense Output Configuration: sparse_threshold=0.0
    preprocessor = ColumnTransformer([
        ('numeric', numeric_pipeline, numeric_columns),
        ('categorical', categorical_pipeline, categorical_columns),
    ], sparse_threshold=0.0, verbose_feature_names_out=False) 
    
    return preprocessor, X_pd, y_np

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data for classification (with stratification)."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True, stratify=y)

def split_regression_data(X, y, test_size=0.2, random_state=42):
    """Split data for regression (no stratification)."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)

def train_classification_models(X_train, X_test, y_train, y_test, preprocessor):
    """Train multiple classification models and return results."""
    
    models = [
        ('Logistic Regression', LogisticRegression(max_iter=500)),
        ('Random Forest', RandomForestClassifier(n_estimators=100, max_depth=10)),
        ('Linear Support Vector Machine', LinearSVC())
    ]
    
    results = []
    
    for name, model in models:
        print(f"Training {name}...")
        
        # End-to-end ML pipeline
        full_pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('model', model)
        ])
        
        # Free memory
        _try_free_memory()
        
        # Train model
        full_pipeline.fit(X_train, y_train)
        
        # Calculate metrics
        train_accuracy = full_pipeline.score(X_train, y_train)
        test_predictions = full_pipeline.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_predictions)
        f1 = f1_score(y_test, test_predictions, average='weighted')
        
        results.append({
            'name': name,
            'metrics': {
                'train_accuracy': float(train_accuracy),
                'test_accuracy': float(test_accuracy),
                'f1_weighted': float(f1)
            },
            'pipeline': full_pipeline
        })
    
    # Sort by F1 score (best first)
    results.sort(key=lambda x: x['metrics']['f1_weighted'], reverse=True)
    return results

def train_regression_models(X_train, X_test, y_train, y_test, preprocessor, primary_metric='r2'):
    """Train multiple regression models and return results."""

    models = [
        ('Ridge Regression', Ridge()),
        ('Random Forest', RandomForestRegressor(n_estimators=100, max_depth=10)),
        ('Linear SVR', LinearSVR(max_iter=10000, tol=1e-3))
    ]
    
    results = []
    
    for name, model in models:
        print(f"Training {name}...")
        
        # End-to-end ML pipeline
        full_pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('model', model)
        ])
        
        # Free memory
        _try_free_memory()
        
        # Train model
        full_pipeline.fit(X_train, y_train)
        
        # Calculate metrics
        test_predictions = full_pipeline.predict(X_test)
        r2 = r2_score(y_test, test_predictions)
        rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        mae = mean_absolute_error(y_test, test_predictions)
        
        results.append({
            'name': name,
            'metrics': {
                'r2': float(r2),
                'rmse': float(rmse),
                'mae': float(mae)
            },
            'pipeline': full_pipeline
        })
    
    # Sort by requested metric
    if primary_metric == 'r2':
        results.sort(key=lambda x: x['metrics']['r2'], reverse=True)
    elif primary_metric == 'rmse':
        results.sort(key=lambda x: x['metrics']['rmse'])
    elif primary_metric == 'mae':
        results.sort(key=lambda x: x['metrics']['mae'])
    
    return results

# Classification model optimization
def optimize_logistic_regression(X_train, y_train, preprocessor, n_trials=20):
    """Find best parameters for Logistic Regression."""

    # Free memory
    _try_free_memory()

    def objective(trial):
        # Choose parameters to test
        C = trial.suggest_float('C', 0.01, 100, log=True)
        max_iter = trial.suggest_int('max_iter', 100, 50000)
        
        model = LogisticRegression(C=C, max_iter=max_iter, tol = 1e-3)
        pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('model', model)
        ])
        
        # Use cross-validation to evaluate
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_weighted')
        return scores.mean()
    
    # Run optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Build best model
    best_params = study.best_params
    best_model = LogisticRegression(C=best_params['C'], max_iter=best_params['max_iter'], tol = 1e-3)
    best_pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('model', best_model)
    ])
    best_pipeline.fit(X_train, y_train)
    
    return {
        'name': 'Optimized Logistic Regression',
        'metrics': {
            'train_accuracy': float(best_pipeline.score(X_train, y_train)),
            'cv_score': float(study.best_value)
        },
        'pipeline': best_pipeline,
        'best_params': best_params
    }

def optimize_random_forest(X_train, y_train, preprocessor, n_trials=20):
    """Find best parameters for Random Forest."""

    # Free memory
    _try_free_memory()

    def objective(trial):
        # Choose parameters to test
        n_estimators = trial.suggest_int('n_estimators', 50, 100)
        max_depth = trial.suggest_int('max_depth', 5, 12)
        
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('model', model)
        ])
        
        # Use cross-validation to evaluate
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_weighted')
        return scores.mean()
    
    # Run optimization
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(n_startup_trials=10), 
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Build best model
    best_params = study.best_params
    best_model = RandomForestClassifier(
        n_estimators=best_params['n_estimators'], 
        max_depth=best_params['max_depth'],
        random_state=42
    )
    best_pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('model', best_model)
    ])
    best_pipeline.fit(X_train, y_train)
    
    return {
        'name': 'Optimized Random Forest',
        'metrics': {
            'train_accuracy': float(best_pipeline.score(X_train, y_train)),
            'cv_score': float(study.best_value)
        },
        'pipeline': best_pipeline,
        'best_params': best_params
    }

def optimize_svc(X_train, y_train, preprocessor, n_trials=20):
    """Find best parameters for Support Vector Classifier using Optuna."""  
   
    # Free memory
    _try_free_memory()

    def objective(trial):
        C = trial.suggest_float('C', 0.01, 100, log=True)
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
        max_iter = trial.suggest_int('max_iter', 100, 50000)

        # Only allow legal combinations
        if penalty == 'l1':
            loss = 'squared_hinge'
            dual = False
        else:  # l2
            loss = trial.suggest_categorical('loss', ['hinge', 'squared_hinge'])
            # dual = True for hinge, False or True for squared_hinge
            if loss == 'hinge':
                dual = True
            else:
                dual = trial.suggest_categorical('dual', [True, False])

        model = LinearSVC(
            C=C,
            penalty=penalty,
            loss=loss,
            max_iter=max_iter,
            dual=dual
        )
        pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('model', model)
        ])

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_weighted')
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    # Assign values from search space safely
    penalty = best_params['penalty']
    C = best_params['C']
    max_iter = best_params['max_iter']

    if penalty == 'l1':
        loss = 'squared_hinge'
        dual = False
    else:  # l2
        loss = best_params['loss']
        dual = best_params['dual'] if 'dual' in best_params else (True if loss == 'hinge' else False)

    best_model = LinearSVC(
        C=C,
        penalty=penalty,
        loss=loss,
        max_iter=max_iter,
        dual=dual
    )
    best_pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('model', best_model)
    ])
    best_pipeline.fit(X_train, y_train)

    return {
        'name': 'Optimized Linear SVC',
        'metrics': {
            'train_accuracy': float(best_pipeline.score(X_train, y_train)),
            'cv_score': float(study.best_value)
        },
        'pipeline': best_pipeline,
        'best_params': {
            'C': C,
            'penalty': penalty,
            'loss': loss,
            'max_iter': max_iter,
            'dual': dual
        }
    }

# Regression model optimization
def optimize_ridge_regression(X_train, y_train, preprocessor, n_trials=20):
    """Find best parameters for Ridge regression."""
    
    # Free memory
    _try_free_memory()

    def objective(trial):
        # Choose parameters to test
        alpha = trial.suggest_float('alpha', 10, 10000, log=True)
        
        model = Ridge(alpha=alpha, random_state=42)
        pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('model', model)
        ])
        
        # Use cross-validation to evaluate
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='r2')
        return scores.mean()
    
    # Run optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Build best model
    best_params = study.best_params
    best_model = Ridge(alpha=best_params['alpha'], random_state=42)
    best_pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('model', best_model)
    ])
    best_pipeline.fit(X_train, y_train)
    
    return {
        'name': 'Optimized Ridge Regression',
        'metrics': {
            'train_r2': float(best_pipeline.score(X_train, y_train)),
            'cv_score': float(study.best_value)
        },
        'pipeline': best_pipeline,
        'best_params': best_params
    }

def optimize_random_forest_regressor(X_train, y_train, preprocessor, n_trials=20):
    """Find best parameters for Random Forest Regressor."""
    
    # Free memory
    _try_free_memory()

    def objective(trial):
        # Choose parameters to test
        n_estimators = trial.suggest_int('n_estimators', 10, 30)
        max_depth = trial.suggest_int('max_depth', 3, 8)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 5)
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('model', model)
        ])
        
        # Use cross-validation to evaluate
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='r2')
        return scores.mean()
    
    # Run optimization
    study = optuna.create_study(direction='maximize', 
                                sampler=optuna.samplers.TPESampler(n_startup_trials=10), 
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Build best model
    best_params = study.best_params
    best_model = RandomForestRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_leaf=best_params['min_samples_leaf'],
        random_state=42
    )
    best_pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('model', best_model)
    ])
    best_pipeline.fit(X_train, y_train)
    
    return {
        'name': 'Optimized Random Forest Regressor',
        'metrics': {
            'train_r2': float(best_pipeline.score(X_train, y_train)),
            'cv_score': float(study.best_value)
        },
        'pipeline': best_pipeline,
        'best_params': best_params
    }

def optimize_svr_regressor(X_train, y_train, preprocessor, n_trials=20):
    """Find best parameters for Support Vector Regressor.""" 

    # Free memory
    _try_free_memory()

    def objective(trial):
        # Choose parameters to test
        C = trial.suggest_float('C', 0.01, 100, log=True)
        epsilon = trial.suggest_float('epsilon', 0.01, 1.0, log=True)
        
        model = LinearSVR(C=C, epsilon=epsilon, max_iter=2000, random_state=42)
        pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('model', model)
        ])
        
        # Use cross-validation to evaluate
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='r2')
        return scores.mean()
    
    # Run optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Build best model
    best_params = study.best_params
    best_model = LinearSVR(
        C=best_params['C'],
        epsilon=best_params['epsilon'],
        max_iter=2000,
        random_state=42
    )
    best_pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('model', best_model)
    ])
    best_pipeline.fit(X_train, y_train)
    
    return {
        'name': 'Optimized SVR',
        'metrics': {
            'train_r2': float(best_pipeline.score(X_train, y_train)),
            'cv_score': float(study.best_value)
        },
        'pipeline': best_pipeline,
        'best_params': best_params
    }

def predict(model_path: str, test_data_path: str, output_path: str = "predictions.csv") -> Dict[str, Any]:
    """Make predictions on test data using a saved model."""
    from joblib import load
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data file not found: {test_data_path}")
    
    # Load the model
    model = load(model_path)
    
    # Load test data
    if test_data_path.endswith(".parquet"):
        test_df = pd.read_parquet(test_data_path)
    else:
        test_df = pd.read_csv(test_data_path)
    
    # Make predictions
    predictions = model.predict(test_df)
    
    # Create results dataframe
    results_df = test_df.copy()
    results_df['predictions'] = predictions
    
    # Add probabilities for classification if available
    probabilities = None
    if hasattr(model, 'predict_proba'):
        try:
            probabilities = model.predict_proba(test_df)
            # Try to get classes from model or pipeline
            if hasattr(model, 'classes_'):
                classes = model.classes_
            elif hasattr(model, 'named_steps') and 'model' in model.named_steps:
                classes = model.named_steps['model'].classes_
            else:
                classes = None
                
            if classes is not None:
                for i, class_name in enumerate(classes):
                    results_df[f'prob_{class_name}'] = probabilities[:, i]
        except Exception:
            probabilities = None
    
    # Save results
    results_df.to_csv(output_path, index=False)
    
    # Generate summary statistics
    summary = {
        'test_samples': len(predictions),
        'output_file': output_path,
        'has_probabilities': probabilities is not None
    }
    
    # Add prediction statistics
    if probabilities is not None:  # Classification
        from collections import Counter
        pred_counts = Counter(predictions)
        summary['prediction_distribution'] = dict(pred_counts)
        summary['task_type'] = 'classification'
    else:  # Regression
        summary['prediction_stats'] = {
            'mean': float(predictions.mean()),
            'std': float(predictions.std()),
            'min': float(predictions.min()),
            'max': float(predictions.max())
        }
        summary['task_type'] = 'regression'
    
    return summary

class Float32Transformer:
    """Picklable transformer to convert arrays to float32 for memory efficiency."""
    
    def __init__(self, accept_sparse=False):
        self.accept_sparse = accept_sparse
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Convert sparse to dense if needed
        if hasattr(X, 'toarray') and not self.accept_sparse:
            X = X.toarray()
        return X.astype('float32')
    
    def __reduce__(self):
        return (Float32Transformer, (self.accept_sparse,))