# chat_agent.py - Final simplified version for beginners
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from joblib import dump
from pandas.api.types import is_numeric_dtype

# Import our modules
from src.tools import gpu_tools
from src.tools.exp_store import ExperimentStore
from src import llm


def setup_fake_cupy():
    """Create a fake cupy module if it's not installed."""
    if "cupy" in sys.modules:
        return
    
    try:
        import cupy
    except ImportError:
        # Create a fake cupy module
        fake_cupy = types.SimpleNamespace()
        fake_cupy.get_default_memory_pool = lambda: types.SimpleNamespace(free_all_blocks=lambda: None)
        sys.modules["cupy"] = fake_cupy

class ChatAgent:
    """A friendly chat agent for machine learning tasks."""
    
    def __init__(self):
        setup_fake_cupy()
        
        # Storage for experiments
        self.store = ExperimentStore()
        self.store.clear_all()  # Start fresh
        
        # AI client
        self.llm_client = llm.create_client()
        
        # Current dataset info
        self.df = None
        self.target_column = None

        # Save model and predictions to system temp folder (same as uploaded files)
        import os
        import tempfile
        temp_dir = tempfile.gettempdir()
        self.champion_model_path = os.path.join(temp_dir, "best_model.joblib")
        
        # Conversation memory
        self.conversation = []
        
        # Available tools
        # Dictionary - Tool registry
        self.tools = {
            "load_dataset": self._load_dataset,
            "set_target": self._set_target,
            "describe_data": self._describe_data,
            "preview_data": self._preview_data,
            "train_classification": self._train_classification,
            "train_regression": self._train_regression,
            "optimize_logistic": self._optimize_logistic,
            "optimize_svc": self._optimize_svc,
            "optimize_forest": self._optimize_forest,
            "optimize_ridge": self._optimize_ridge,
            "optimize_forest_regressor": self._optimize_forest_regressor,
            "optimize_svr": self._optimize_svr,
            "show_best_model": self._show_best_model,
            "show_history": self._show_history,
            "help": self._help,
            "predict": self._predict
        }

        # Simple system prompt
        self.system_message = """You are a helpful machine learning assistant. 

                        When users ask to work with data, use the appropriate tool:
                        - load_dataset(path, target) - Load a CSV or Parquet file (target is optional)
                        - set_target(target) - Set the target column
                        - describe_data() - Show info about the loaded dataset
                        - preview_data(rows?) - Show first few rows
                        - train_classification() - Train classification models
                        - train_regression() - Train regression models  
                        - optimize_logistic(trials?) - Optimize logistic regression
                        - optimize_svc(trials?) - Optimize support vector classifier
                        - optimize_forest(trials?) - Optimize random forest classifier
                        - optimize_ridge(trials?) - Optimize ridge regression
                        - optimize_forest_regressor(trials?) - Optimize random forest regressor
                        - optimize_svr(trials?) - Optimize support vector regressor
                        - show_best_model(metric) - Show best model by metric
                        - show_history(limit?) - Show recent experiments
                        - help() - Show available commands
                        - predict(test_data_path, output_path?) - Predict on the test dataset using the best model

                        For general ML questions, answer helpfully without using tools."""

    def _load_dataset(self, path: str, target: str = None) -> str:
        """Load a dataset from file."""

        # Store the original filename for display
        display_name = path
        
        # Check if this is an uploaded filename
        if hasattr(self, 'uploaded_files') and path in self.uploaded_files:
            actual_path = self.uploaded_files[path]
            file_path = Path(actual_path)
            display_name = path
        else:
            # Original logic
            file_path = Path(path)
            display_name = file_path.name
        
        if not file_path.exists():
            # Show available uploaded files if any
            if hasattr(self, 'uploaded_files') and self.uploaded_files:
                available = ", ".join(self.uploaded_files.keys())
                return f"File not found: {path}. Available uploaded files: {available}"
            return f"File not found: {path}. Please check the path."
        
        try:
            # Load using gpu_tools
            self.df = gpu_tools.load_data(str(file_path))
            
            # Store the display name
            self.current_dataset_name = display_name
            
            # Set target if provided
            if target:
                if target not in self.df.columns:
                    available_cols = ", ".join(list(self.df.columns)[:5])
                    return f"Column '{target}' not found. Available columns: {available_cols}..."
                self.target_column = target
                return f"Loaded {display_name} with {len(self.df):,} rows and {len(self.df.columns)} columns. Target: {target}"
            else:
                return f"Loaded {display_name} with {len(self.df):,} rows and {len(self.df.columns)} columns. No target set yet." 

        except Exception as e:
            return f"Error loading file: {str(e)}"

    def _set_target(self, target: str) -> str:
        """Set the target column."""
        if self.df is None:
            return "No dataset loaded. Please load a dataset first."
        
        if target not in self.df.columns:
            available = ", ".join(list(self.df.columns)[:5])
            return f"Column '{target}' not found. Available columns: {available}..."
        
        self.target_column = target
        return f"Target set to: {target}"

    def _describe_data(self) -> str:
        """Generate dataset description using LLM."""
        if self.df is None:
            return "No dataset loaded. Please load a dataset first."
        
        # Gather dataset information
        rows, cols = self.df.shape
        missing_total = self.df.isnull().sum().sum()
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Target analysis
        target_info = ""
        if self.target_column and self.target_column in self.df.columns:
            target_data = self.df[self.target_column]
            unique_values = target_data.nunique()
            
            if is_numeric_dtype(target_data):
                target_info = f"Target '{self.target_column}' is numeric with {unique_values} unique values. Range: {target_data.min():.2f} to {target_data.max():.2f}. Mean: {target_data.mean():.2f}"
            else:
                target_info = f"Target '{self.target_column}' is categorical with {unique_values} unique values."
                if unique_values <= 10:
                    class_counts = target_data.value_counts()
                    target_info += f" Top classes: {dict(class_counts.head(3))}"
        
        # Missing values details
        missing_info = ""
        if missing_total > 0:
            missing_cols = self.df.isnull().sum()[self.df.isnull().sum() > 0]
            missing_info = f"Missing values in {len(missing_cols)} columns: " + ", ".join([f"{col}({count})" for col, count in missing_cols.head(5).items()])
        
        # Replace the prompt variable in the _describe_data method with this:

        prompt = f"""Analyze this dataset and provide a well-formatted description:

            Dataset basics:
            - Shape: {rows:,} rows × {cols} columns
            - Numeric columns: {len(numeric_cols)}
            - Categorical columns: {len(categorical_cols)}
            - Total missing values: {missing_total:,}

            Target information:
            {target_info if target_info else "No target column set"}

            Missing values:
            {missing_info if missing_info else "No missing values"}

            Create a CONCRETE description. Focus on key insights about data quality and ML suitability.
            
            CRITICAL FORMATTING RULES - MUST FOLLOW:
            1. DO NOT include ANY title or heading at the beginning
            2. DO NOT write "Dataset Analysis" or any other title
            3. Start IMMEDIATELY with "**Dataset Overview**" as the first line
            4. NO markdown headers (#, ##, ###) anywhere
            5. Use **bold** for section names only (like **Dataset Overview**, **Missing Values**, etc.)
            6. Use regular bullet points (•) for items
            7. Keep all text at normal size
            
            Begin your response directly with **Dataset Overview** and the content. No title before that."""

        try:
            response = self.llm_client.chat([
                {"role": "system", "content": "You are a data analyst. Create clear, well-formatted dataset descriptions using markdown. Be concise but informative. Use proper markdown formatting with headers (##), bullet points (-), and code blocks (```)."},
                {"role": "user", "content": prompt}
            ])
            
            description = response["choices"][0]["message"]["content"]
            description = description.strip()
        
            if not description.startswith('#'):
                description = "## Dataset Analysis\n\n" + description

            return description
            
        except Exception as e:
            # Fallback to simple description if LLM fails
            return f"""Dataset: {rows:,} rows, {cols} columns
                Target: {self.target_column}
                Missing values: {missing_total:,}
                Numeric columns: {len(numeric_cols)}
                Categorical columns: {len(categorical_cols)}"""

    def _preview_data(self, rows: int = 5) -> str:
        """Generate data preview using LLM with markdown table format."""
        if self.df is None:
            return "No dataset loaded. Please load a dataset first."
        
        rows = max(1, min(20, rows))
        preview_df = self.df.head(rows)
        
        # Convert dataframe to markdown format
        try:
            # This requires: pip install tabulate
            df_markdown = preview_df.to_markdown(index=True)
        except Exception:
            # Fallback to string format
            df_markdown = preview_df.to_string(index=True, max_cols=None, max_colwidth=None)
        
        # Get column info
        column_info = []
        for col in preview_df.columns:
            dtype = str(preview_df[col].dtype)
            non_null = preview_df[col].notna().sum()
            unique = preview_df[col].nunique()
            column_info.append(f"{col} ({dtype}, {non_null}/{rows} non-null, {unique} unique)")
        
        prompt = f"""Here's a preview of the dataset with {rows} DATA rows (not including header):

        Data in markdown table format:
        {df_markdown}

        Column information:
        {chr(10).join('- ' + info for info in column_info)}

        Total dataset size: {len(self.df):,} rows × {len(self.df.columns)} columns

        CRITICAL INSTRUCTIONS:
        1. Display the ENTIRE table exactly as provided above - it contains {rows} data rows plus a header row
        2. The table above has {rows + 1} total lines: 1 header + {rows} data rows
        3. You MUST show ALL {rows} data rows - verify you're showing {rows} numbered/indexed rows
        4. The header row with column names does NOT count as a data row
        5. Do not truncate, summarize, or skip any rows

        Present this data preview with:
        - A brief description (1-2 sentences)
        - The COMPLETE table with all {rows} data rows visible
        - 2-3 key observations about the data
        - Note about total dataset size

        Example: If showing 3 rows, the table should have 4 lines total (header + 3 data rows).
        DO NOT use bullet points when displaying the table."""

        try:
            response = self.llm_client.chat([
                {"role": "system", "content": "You are a data analyst. Present the data preview using the exact markdown table provided. Do not modify or truncate the table."},
                {"role": "user", "content": prompt}
            ])
            
            preview = response["choices"][0]["message"]["content"]
            return preview.strip()
        
        except Exception as e:
            # Fallback that still uses markdown if available
            fallback_output = f"""Dataset Preview ({rows} rows)

                {df_markdown}

                **Dataset Info:**
                - Total rows: {len(self.df):,}
                - Total columns: {len(self.df.columns)}
                - Columns: {', '.join(self.df.columns.tolist())}

                Showing {rows} of {len(self.df):,} total rows."""
            
            return fallback_output

    def _train_classification(self) -> str:
        """Train classification models."""
        if self.df is None:
            return "No dataset loaded. Please load a dataset first."
        
        if not self.target_column:
            return "No target column set. Use set_target() first."
        
        # Check if target is suitable for classification
        target_data = self.df[self.target_column]
        if is_numeric_dtype(target_data) and target_data.nunique() > 20:
            return f"Target '{self.target_column}' looks numeric with {target_data.nunique()} unique values. Try regression instead."
        
        try:
            # Prepare data
            preprocessor, X, y = gpu_tools.build_preprocessing_pipeline(self.df, self.target_column)
            X_train, X_test, y_train, y_test = gpu_tools.split_data(X, y)
            
            # Train models
            results = gpu_tools.train_classification_models(X_train, X_test, y_train, y_test, preprocessor)
            
            # Save best model
            best_model = results[0]
            dump(best_model['pipeline'], self.champion_model_path)
            
            # Save to experiment store
            for result in results:
                self.store.save_experiment({
                    'task': 'classification',
                    'dataset_shape': list(self.df.shape),
                    'target': self.target_column,
                    'model': result['name'],
                    'metrics': result['metrics']
                })
            
            # Create summary
            summary = f"Trained {len(results)} classification models:\n"
            for i, result in enumerate(results, 1):
                acc = result['metrics']['test_accuracy']
                f1 = result['metrics']['f1_weighted']
                summary += f"{i}. {result['name']}: Accuracy={acc:.3f}, F1={f1:.3f}\n"
            
            summary += f"\nBest model ({results[0]['name']}) saved to {self.champion_model_path}"
            return summary
            
        except Exception as e:
            return f"Training failed: {str(e)}"

    def _train_regression(self) -> str:
        """Train regression models."""
        if self.df is None:
            return "No dataset loaded. Please load a dataset first."
        
        if not self.target_column:
            return "No target column set. Use set_target() first."
        
        # Check if target is numeric
        target_data = self.df[self.target_column]
        if not is_numeric_dtype(target_data):
            return f"Target '{self.target_column}' is not numeric. Try classification instead."
        
        try:
            # Prepare data
            preprocessor, X, y = gpu_tools.build_preprocessing_pipeline(self.df, self.target_column)
            X_train, X_test, y_train, y_test = gpu_tools.split_regression_data(X, y)
            
            # Train models
            results = gpu_tools.train_regression_models(X_train, X_test, y_train, y_test, preprocessor)
            
            # Save best model
            best_model = results[0]
            dump(best_model['pipeline'], self.champion_model_path)
            
            # Save to experiment store
            for result in results:
                self.store.save_experiment({
                    'task': 'regression',
                    'dataset_shape': list(self.df.shape),
                    'target': self.target_column,
                    'model': result['name'],
                    'metrics': result['metrics']
                })
            
            # Create summary
            summary = f"Trained {len(results)} regression models:\n"
            for i, result in enumerate(results, 1):
                r2 = result['metrics']['r2']
                rmse = result['metrics']['rmse']
                summary += f"{i}. {result['name']}: R²={r2:.3f}, RMSE={rmse:.3f}\n"
            
            summary += f"\nBest model ({results[0]['name']}) saved to {self.champion_model_path}"
            return summary
            
        except Exception as e:
            return f"Training failed: {str(e)}"
        
    def _format_classification_summary(self, result: Dict, test_accuracy: float) -> str:
        """Format classification optimization results."""
        summary = f"{result['name']}:\n"
        summary += f"Test Accuracy: {test_accuracy:.3f}\n"
        summary += f"CV Score: {result['metrics']['cv_score']:.3f}\n"
        summary += f"Best parameters: {result['best_params']}\n"
        summary += f"Model saved to {self.champion_model_path}"
        return summary

    def _format_regression_summary(self, result: Dict, test_r2: float) -> str:
        """Format regression optimization results."""
        summary = f"{result['name']}:\n"
        summary += f"Test R²: {test_r2:.3f}\n"
        summary += f"CV Score: {result['metrics']['cv_score']:.3f}\n"
        summary += f"Best parameters: {result['best_params']}\n"
        summary += f"Model saved to {self.champion_model_path}"
        return summary

    def _optimize_logistic(self, trials: int = 20) -> str:
        """Optimize logistic regression."""
        if self.df is None:
            return "No dataset loaded. Please load a dataset first."
        
        if not self.target_column:
            return "No target column set. Use set_target() first."
        
        try:
            # Prepare data
            preprocessor, X, y = gpu_tools.build_preprocessing_pipeline(self.df, self.target_column)
            X_train, X_test, y_train, y_test = gpu_tools.split_data(X, y)
            
            # Optimize
            print(f"Optimizing Logistic Regression with {trials} trials...")
            result = gpu_tools.optimize_logistic_regression(X_train, y_train, preprocessor, trials)

            # Test on holdout set
            test_accuracy = result['pipeline'].score(X_test, y_test)
            result['metrics']['test_accuracy'] = test_accuracy
            
            # Save best model
            dump(result['pipeline'], self.champion_model_path)
            
            # Save to experiment store
            self.store.save_experiment({
                'task': 'optimization',
                'dataset_shape': list(self.df.shape),
                'target': self.target_column,
                'model': result['name'],
                'metrics': result['metrics'],
                'best_params': result['best_params']
            })
            
            return self._format_classification_summary(result, test_accuracy)
            
        except Exception as e:
            return f"Optimization failed: {str(e)}"
    
    # Example tool implementation
    def _optimize_svc(self, trials: int = 20) -> str:
        """Optimize Support Vector Classifier."""
        if self.df is None:
            return "No dataset loaded. Please load a dataset first."
        
        if not self.target_column:
            return "No target column set. Use set_target() first."
        
        try:
            # Prepare data
            preprocessor, X, y = gpu_tools.build_preprocessing_pipeline(self.df, self.target_column)
            X_train, X_test, y_train, y_test = gpu_tools.split_data(X, y)
            
            # Optimize
            print(f"Optimizing SVC with {trials} trials...")
            result = gpu_tools.optimize_svc(X_train, y_train, preprocessor, trials)
            
            # Test on holdout set
            test_accuracy = result['pipeline'].score(X_test, y_test)
            result['metrics']['test_accuracy'] = test_accuracy
            
            # Save best model
            dump(result['pipeline'], self.champion_model_path)
            
            # Save to experiment store
            self.store.save_experiment({
                'task': 'optimization',
                'dataset_shape': list(self.df.shape),
                'target': self.target_column,
                'model': result['name'],
                'metrics': result['metrics'],
                'best_params': result['best_params']
            })
            
            return self._format_classification_summary(result, test_accuracy)
            
        except Exception as e:
            return f"Optimization failed: {str(e)}"

    def _optimize_forest(self, trials: int = 20) -> str:
        """Optimize random forest."""
        if self.df is None:
            return "No dataset loaded. Please load a dataset first."
        
        if not self.target_column:
            return "No target column set. Use set_target() first."
        
        try:
            # Prepare data
            preprocessor, X, y = gpu_tools.build_preprocessing_pipeline(self.df, self.target_column)
            X_train, X_test, y_train, y_test = gpu_tools.split_data(X, y)
            
            # Optimize
            print(f"Optimizing Random Forest with {trials} trials...")
            result = gpu_tools.optimize_random_forest(X_train, y_train, preprocessor, trials)

            # Test on holdout set
            test_accuracy = result['pipeline'].score(X_test, y_test)
            result['metrics']['test_accuracy'] = test_accuracy
            
            # Save best model
            dump(result['pipeline'], self.champion_model_path)
            
            # Save to experiment store
            self.store.save_experiment({
                'task': 'optimization',
                'dataset_shape': list(self.df.shape),
                'target': self.target_column,
                'model': result['name'],
                'metrics': result['metrics'],
                'best_params': result['best_params']
            })
            
            return self._format_classification_summary(result, test_accuracy)
            
        except Exception as e:
            return f"Optimization failed: {str(e)}"

    def _optimize_ridge(self, trials: int = 20) -> str:
        """Optimize ridge regression."""
        if self.df is None:
            return "No dataset loaded. Please load a dataset first."
        
        if not self.target_column:
            return "No target column set. Use set_target() first."
        
        try:
            # Prepare data
            preprocessor, X, y = gpu_tools.build_preprocessing_pipeline(self.df, self.target_column)
            X_train, X_test, y_train, y_test = gpu_tools.split_regression_data(X, y)
            
            # Optimize
            print(f"Optimizing Ridge Regression with {trials} trials...")
            result = gpu_tools.optimize_ridge_regression(X_train, y_train, preprocessor, trials)
            
            # Test on holdout set
            test_r2 = result['pipeline'].score(X_test, y_test)
            result['metrics']['test_r2'] = test_r2
            
            # Save best model
            dump(result['pipeline'], self.champion_model_path)
            
            # Save to experiment store
            self.store.save_experiment({
                'task': 'optimization',
                'dataset_shape': list(self.df.shape),
                'target': self.target_column,
                'model': result['name'],
                'metrics': result['metrics'],
                'best_params': result['best_params']
            })
            
            return self._format_regression_summary(result, test_r2)
            
        except Exception as e:
            return f"Optimization failed: {str(e)}"

    def _optimize_forest_regressor(self, trials: int = 20) -> str:
        """Optimize Random Forest Regressor."""
        if self.df is None:
            return "No dataset loaded. Please load a dataset first."
        
        if not self.target_column:
            return "No target column set. Use set_target() first."
        
        try:
            # Prepare data
            preprocessor, X, y = gpu_tools.build_preprocessing_pipeline(self.df, self.target_column)
            X_train, X_test, y_train, y_test = gpu_tools.split_regression_data(X, y)
            
            # Optimize
            print(f"Optimizing Random Forest Regressor with {trials} trials...")
            result = gpu_tools.optimize_random_forest_regressor(X_train, y_train, preprocessor, trials)
            
            # Test on holdout set
            test_r2 = result['pipeline'].score(X_test, y_test)
            result['metrics']['test_r2'] = test_r2
            
            # Save best model
            dump(result['pipeline'], self.champion_model_path)
            
            # Save to experiment store
            self.store.save_experiment({
                'task': 'optimization',
                'dataset_shape': list(self.df.shape),
                'target': self.target_column,
                'model': result['name'],
                'metrics': result['metrics'],
                'best_params': result['best_params']
            })
            
            return self._format_regression_summary(result, test_r2)
            
        except Exception as e:
            return f"Optimization failed: {str(e)}"

    def _optimize_svr(self, trials: int = 20) -> str:
        """Optimize Support Vector Regressor."""
        if self.df is None:
            return "No dataset loaded. Please load a dataset first."
        
        if not self.target_column:
            return "No target column set. Use set_target() first."
        
        try:
            # Prepare data
            preprocessor, X, y = gpu_tools.build_preprocessing_pipeline(self.df, self.target_column)
            X_train, X_test, y_train, y_test = gpu_tools.split_regression_data(X, y)
            
            # Optimize
            print(f"Optimizing SVR with {trials} trials...")
            result = gpu_tools.optimize_svr_regressor(X_train, y_train, preprocessor, trials)
            
            # Test on holdout set
            test_r2 = result['pipeline'].score(X_test, y_test)
            result['metrics']['test_r2'] = test_r2
            
            # Save best model
            dump(result['pipeline'], self.champion_model_path)
            
            # Save to experiment store
            self.store.save_experiment({
                'task': 'optimization',
                'dataset_shape': list(self.df.shape),
                'target': self.target_column,
                'model': result['name'],
                'metrics': result['metrics'],
                'best_params': result['best_params']
            })
            
            return self._format_regression_summary(result, test_r2)
            
        except Exception as e:
            return f"Optimization failed: {str(e)}"

    def _show_best_model(self, metric: str = "accuracy") -> str:
        """Show the best model by a given metric."""
        # Map common metric names
        metric_mapping = {
            'accuracy': 'test_accuracy',
            'acc': 'test_accuracy',
            'f1': 'f1_weighted',
            'r2': 'r2',
            'rmse': 'rmse'
        }
        
        actual_metric = metric_mapping.get(metric.lower(), metric)
        best_experiment = self.store.find_best_experiment(actual_metric)
        
        if not best_experiment:
            return f"No experiments found with metric '{metric}'."
        
        summary = f"Best model by {metric}:\n"
        summary += f"Model: {best_experiment.get('model', 'Unknown')}\n"
        summary += f"Task: {best_experiment.get('task', 'Unknown')}\n"
        summary += f"Target: {best_experiment.get('target', 'Unknown')}\n"
        summary += f"Dataset: {best_experiment.get('dataset_shape', 'Unknown')} (rows, cols)\n"
        
        metrics = best_experiment.get('metrics', {})
        summary += "Metrics:\n"
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                summary += f"  {metric_name}: {value:.3f}\n"
            else:
                summary += f"  {metric_name}: {value}\n"
        
        return summary

    def _show_history(self, limit: int = 5) -> str:
        """Show recent experiment history."""
        experiments = self.store.get_recent_experiments(limit)
        
        if not experiments:
            return "No experiments found."
        
        summary = f"Recent {len(experiments)} experiments:\n\n"
        
        for i, exp in enumerate(experiments, 1):
            summary += f"{i}. {exp.get('model', 'Unknown')} ({exp.get('task', 'unknown')})\n"
            summary += f"   Target: {exp.get('target', 'unknown')}\n"
            summary += f"   Date: {exp.get('date', 'unknown')}\n"
            
            metrics = exp.get('metrics', {})
            if metrics:
                summary += "   Metrics: "
                metric_strs = []
                for name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        metric_strs.append(f"{name}={value:.3f}")
                    else:
                        metric_strs.append(f"{name}={value}")
                summary += ", ".join(metric_strs[:3])  # Show first 3 metrics
                summary += "\n"
            
            summary += "\n"
        
        return summary

    def _predict(self, test_data_path: str, output_path: str = "predictions.csv") -> str:
        """Make predictions on test data using the trained model."""
        import os
        
        # If just a filename (not full path), save to temp directory
        if not os.path.isabs(output_path):
            model_dir = os.path.dirname(self.champion_model_path)
            output_path = os.path.join(model_dir, output_path)

        if not os.path.exists(self.champion_model_path):
            return f"No trained model found at {self.champion_model_path}. Please train a model first."
        
        # Handle uploaded files mapping
        if hasattr(self, 'uploaded_files') and test_data_path in self.uploaded_files:
            actual_path = self.uploaded_files[test_data_path]
        else:
            actual_path = test_data_path

        try:
            # Use gpu_tools predict function
            result = gpu_tools.predict(self.champion_model_path, actual_path, output_path)
            
            # Create summary message
            summary = f"Predictions completed!\n"
            summary += f"Test samples: {result['test_samples']:,}\n"
            summary += f"Results saved to: {result['output_file']}\n\n"
            
            if result['task_type'] == 'classification':
                summary += "Prediction distribution:\n"
                for pred, count in result['prediction_distribution'].items():
                    percentage = (count / result['test_samples']) * 100
                    summary += f"  {pred}: {count} ({percentage:.1f}%)\n"
                
                if result['has_probabilities']:
                    summary += "\nProbability columns included for each class."
            
            else:  # regression
                stats = result['prediction_stats']
                summary += f"Prediction statistics:\n"
                summary += f"  Mean: {stats['mean']:.3f}\n"
                summary += f"  Std: {stats['std']:.3f}\n"
                summary += f"  Min: {stats['min']:.3f}\n"
                summary += f"  Max: {stats['max']:.3f}\n"
            
            return summary
            
        except Exception as e:
            return f"Prediction failed: {str(e)}"

    def _help(self) -> str:
        """Show available commands."""
        return """Available commands:
        Data Loading:
        • load_dataset(path, target) - Load CSV/Parquet file (target optional)
        • set_target(column) - Set target column
        • describe_data() - Show dataset information
        • preview_data(rows?) - Show first few rows

        Model Training:
        • train_classification() - Train classification models
        • train_regression() - Train regression models

        Classification Optimization:
        • optimize_logistic(trials?) - Optimize logistic regression
        • optimize_svc(trials?) - Optimize support vector classifier
        • optimize_forest(trials?) - Optimize random forest classifier

        Regression Optimization:
        • optimize_ridge(trials?) - Optimize ridge regression
        • optimize_forest_regressor(trials?) - Optimize random forest regressor
        • optimize_svr(trials?) - Optimize support vector regressor
        
        Predictions:
        • predict(test_data_path, output_path?) - Make predictions on test data

        Results:
        • show_best_model(metric?) - Show best model
        • show_history(limit?) - Show recent experiments

        Examples:
        "load dataset Titanic-Dataset.csv/Titanic-Dataset-test.csv"
        "set target variable to be 'Survived'"
        "train classification/regression model"
        "optimize svc with 50 trials"
        "optimize forest regressor with 30 trials"
        "show best model by r2"
        "make inference for the test dataset"
        """
    
    def _generate_custom_analysis(self, user_request: str) -> str:
        """Generate custom analysis based on user request using LLM."""
        if self.df is None:
            return "No dataset loaded. Please load a dataset first."
        
        # Get dataset summary for context
        rows, cols = self.df.shape
        columns_info = {col: str(dtype) for col, dtype in self.df.dtypes.items()}
        missing_summary = self.df.isnull().sum().to_dict()
        
        # Sample data for context (first few rows)
        sample_data = self.df.head(3).to_dict()
        
        prompt = f"""User request: "{user_request}"

            Dataset context:
            - Shape: {rows:,} rows × {cols} columns
            - Target column: {self.target_column}
            - Columns and types: {columns_info}
            - Missing values: {dict(list(missing_summary.items())[:10])}

            Sample data (first 3 rows):
            {sample_data}

            Based on this dataset information, respond to the user's request. If they're asking for:
            - Data exploration: Provide insights about the data structure, quality, patterns
            - ML recommendations: Suggest appropriate algorithms, preprocessing steps
            - Analysis: Offer relevant statistical insights or data observations
            - Technical help: Provide specific guidance for their ML workflow

            Give a helpful, detailed response formatted with markdown. Be specific and actionable."""

        try:
            response = self.llm_client.chat([
                {"role": "system", "content": "You are an expert data scientist. Provide helpful, specific advice about datasets and machine learning workflows. Use clear formatting and be practical in your recommendations."},
                {"role": "user", "content": prompt}
            ])
            
            analysis = response["choices"][0]["message"]["content"]
            return analysis.strip()
            
        except Exception as e:
            return f"I encountered an error analyzing your request: {str(e)}"
    
    # Tool Schema Definition
    def _get_tool_specs(self) -> List[Dict]:
        """Define available tools for the LLM."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "load_dataset",
                    "description": "Load a CSV or Parquet dataset. Target column is optional - can be set later with set_target.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to the data file"},
                            "target": {"type": "string", "description": "Target column name (optional - can be set later)"}
                        },
                        "required": ["path"],  # Only path is required now
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "set_target",
                    "description": "Set the target column for prediction",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target": {"type": "string", "description": "Target column name"}
                        },
                        "required": ["target"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "describe_data",
                    "description": "Show information about the loaded dataset",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "preview_data",
                    "description": "Show first few rows of the dataset",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "rows": {"type": "integer", "description": "Number of rows to show", "default": 5}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "train_classification",
                    "description": "Train classification models",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "train_regression",
                    "description": "Train regression models",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "optimize_logistic",
                    "description": "Optimize logistic regression hyperparameters",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "trials": {"type": "integer", "description": "Number of optimization trials", "default": 20}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "optimize_svc",
                    "description": "Optimize Support Vector Classifier hyperparameters",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "trials": {"type": "integer", "description": "Number of optimization trials", "default": 20}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "optimize_forest",
                    "description": "Optimize random forest hyperparameters",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "trials": {"type": "integer", "description": "Number of optimization trials", "default": 20}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "optimize_ridge",
                    "description": "Optimize ridge regression hyperparameters",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "trials": {"type": "integer", "description": "Number of optimization trials", "default": 20}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "optimize_forest_regressor",
                    "description": "Optimize Random Forest Regressor hyperparameters",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "trials": {"type": "integer", "description": "Number of optimization trials", "default": 20}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "optimize_svr",
                    "description": "Optimize Support Vector Regressor hyperparameters",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "trials": {"type": "integer", "description": "Number of optimization trials", "default": 20}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "show_best_model",
                    "description": "Show the best model by a given metric",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "metric": {"type": "string", "description": "Metric to compare by", "default": "accuracy"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "show_history",
                    "description": "Show recent experiment history",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {"type": "integer", "description": "Number of experiments to show", "default": 5}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "help",
                    "description": "Show available commands and examples",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "predict",
                    "description": "Make predictions on test data using the trained model",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "test_data_path": {"type": "string", "description": "Path to the test data file"},
                            "output_path": {"type": "string", "description": "Path to save predictions", "default": "predictions.csv"}
                        },
                        "required": ["test_data_path"]
                    }
                }
            }
        ]

    def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call a tool function."""
        if tool_name not in self.tools:
            return f"Unknown tool: {tool_name}"
        
        try:
            return self.tools[tool_name](**arguments)
        except Exception as e:
            return f"Error in {tool_name}: {str(e)}"
        
    def _paraphrase_question(self, user_message: str) -> str:
        """Generate a paraphrased version of the user's question."""
        paraphrase_prompt = f"""Paraphrase the following user question in a clear, concise way that captures their intent.
        Keep it brief and direct. If it's a command for the ML system, maintain the command structure.
        
        User question: "{user_message}"
        
        Provide ONLY the paraphrased version, nothing else."""
        
        try:
            response = self.llm_client.chat([
                {"role": "system", "content": "You are a helpful assistant that clarifies and paraphrases questions. Be concise."},
                {"role": "user", "content": paraphrase_prompt}
            ])
            
            paraphrased = response["choices"][0]["message"]["content"].strip()
            # Remove quotes if the LLM wrapped it in quotes
            paraphrased = paraphrased.strip('"\'')
            return paraphrased
        except Exception:
            # If paraphrasing fails, return original
            return user_message
    
    def chat(self, user_message: str) -> str:
        """Main chat method - simplified single pass."""
        # Paraphrase the question first
        paraphrased_question = self._paraphrase_question(user_message)
        
        # Add system message if this is the first message
        if not self.conversation:
            self.conversation.append({"role": "system", "content": self.system_message})
        
        # Add user message
        self.conversation.append({"role": "user", "content": user_message})
        
        try:
            # Get response from LLM
            response = self.llm_client.chat(
                messages=self.conversation,
                tools=self._get_tool_specs()
            )
             
            # Extract the message
            message = response["choices"][0]["message"]
            
            # Check if LLM wants to use a tool
            if message.get("tool_calls"):
                tool_call = message["tool_calls"][0]
                tool_name = tool_call["function"]["name"]
                
                try:
                    arguments = json.loads(tool_call["function"]["arguments"])
                except json.JSONDecodeError:
                    arguments = {}
                
                # Call the tool
                result = self._call_tool(tool_name, arguments)
                
                # Add to conversation
                self.conversation.append({"role": "assistant", "content": result})
                
                return paraphrased_question, result
            
            # Regular text response
            else:
                content = message.get("content", "I'm not sure how to help with that.")
                self.conversation.append({"role": "assistant", "content": content})
                return paraphrased_question, content
                
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            self.conversation.append({"role": "assistant", "content": error_msg})
            return paraphrased_question, error_msg