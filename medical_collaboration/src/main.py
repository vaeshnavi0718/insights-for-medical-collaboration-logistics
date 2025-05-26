"""
Main script for Medical Collaboration Logistics (MCL).
Demonstrates the usage of data processing, model training, and visualization modules.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from data.data_processor import DataProcessor
from models.model_trainer import ModelTrainer
from visualization.visualizer import Visualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize components
    data_processor = DataProcessor()
    model_trainer = ModelTrainer(model_type='random_forest')
    visualizer = Visualizer()
    
    try:
        # Example: Load and process data
        # Note: Replace with your actual data path
        data_path = 'data/raw/medical_data.csv'
        data = data_processor.load_data(data_path)
        
        # Clean and preprocess data
        cleaned_data = data_processor.clean_data()
        
        # Example: Assuming 'outcome' is the target column
        X, y = data_processor.preprocess_data(target_column='outcome')
        
        # Split data
        X_train, X_test, y_train, y_test = model_trainer.train_test_split(X, y)
        
        # Optimize hyperparameters
        best_params = model_trainer.optimize_hyperparameters(X_train, y_train)
        
        # Train model
        model = model_trainer.train(X_train, y_train)
        
        # Evaluate model
        metrics = model_trainer.evaluate(X_test, y_test)
        logger.info(f"Model metrics: {metrics}")
        
        # Create visualizations
        # Distribution of target variable
        visualizer.plot_distribution(y, 'Distribution of Target Variable')
        
        # Correlation matrix
        visualizer.plot_correlation_matrix(X)
        
        # Feature importance (if using Random Forest)
        if hasattr(model, 'feature_importances_'):
            importance = pd.Series(model.feature_importances_, index=X.columns)
            visualizer.plot_feature_importance(importance)
        
        # Confusion matrix
        y_pred = model.predict(X_test)
        visualizer.plot_confusion_matrix(y_test, y_pred)
        
        # ROC curve
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        visualizer.plot_roc_curve(y_test, y_pred_proba)
        
        # Interactive visualizations
        visualizer.create_interactive_scatter(
            pd.concat([X_test, pd.Series(y_test, name='outcome')], axis=1),
            x=X.columns[0],
            y=X.columns[1],
            color='outcome',
            title='Interactive Scatter Plot of Features'
        )
        
        visualizer.create_interactive_heatmap(X)
        
        # Save all figures
        visualizer.save_figures('figures')
        
        # Save the trained model
        model_trainer.save_model('models/trained_model.joblib')
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 