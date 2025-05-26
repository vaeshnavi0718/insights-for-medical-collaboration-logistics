"""
Model training module for Medical Collaboration Logistics (MCL).
Handles model training, evaluation, and hyperparameter tuning.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import optuna
import joblib
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the ModelTrainer.
        
        Args:
            model_type (str): Type of model to use ('random_forest', 'logistic_regression', or 'svm')
        """
        self.model_type = model_type
        self.model = None
        self.best_params = None
        
    def _get_model(self) -> Any:
        """Get the specified model instance."""
        if self.model_type == 'random_forest':
            return RandomForestClassifier()
        elif self.model_type == 'logistic_regression':
            return LogisticRegression()
        elif self.model_type == 'svm':
            return SVC(probability=True)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
    def train_test_split(self, X: pd.DataFrame, y: pd.Series, 
                        test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Split data into training and testing sets.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
        
    def objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial (optuna.Trial): Optuna trial object
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            
        Returns:
            float: Cross-validation score
        """
        if self.model_type == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 10, 100),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
            }
        elif self.model_type == 'logistic_regression':
            params = {
                'C': trial.suggest_float('C', 1e-5, 1e5, log=True),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'saga'])
            }
        elif self.model_type == 'svm':
            params = {
                'C': trial.suggest_float('C', 1e-5, 1e5, log=True),
                'gamma': trial.suggest_float('gamma', 1e-5, 1e5, log=True),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear'])
            }
            
        model = self._get_model()
        model.set_params(**params)
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        return scores.mean()
        
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, 
                               n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimize model hyperparameters using Optuna.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            n_trials (int): Number of optimization trials
            
        Returns:
            Dict[str, Any]: Best hyperparameters
        """
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.objective(trial, X, y), n_trials=n_trials)
        
        self.best_params = study.best_params
        logger.info(f"Best parameters: {self.best_params}")
        return self.best_params
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Train the model with optimized hyperparameters.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target variable
            
        Returns:
            Any: Trained model
        """
        self.model = self._get_model()
        if self.best_params:
            self.model.set_params(**self.best_params)
            
        self.model.fit(X_train, y_train)
        logger.info("Model training completed successfully")
        return self.model
        
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target variable
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
            
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        logger.info("Model evaluation completed successfully")
        return metrics
        
    def save_model(self, path: str):
        """
        Save the trained model to disk.
        
        Args:
            path (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Please train the model first.")
            
        joblib.dump(self.model, path)
        logger.info(f"Model saved successfully to {path}")
        
    def load_model(self, path: str):
        """
        Load a trained model from disk.
        
        Args:
            path (str): Path to the saved model
        """
        self.model = joblib.load(path)
        logger.info(f"Model loaded successfully from {path}") 