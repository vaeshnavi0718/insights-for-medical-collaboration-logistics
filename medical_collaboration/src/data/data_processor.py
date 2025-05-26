"""
Data processing module for Medical Collaboration Logistics (MCL).
Handles data loading, cleaning, and preprocessing operations.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the DataProcessor.
        
        Args:
            data_path (str, optional): Path to the data file
        """
        self.data_path = data_path
        self.data = None
        
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from a file.
        
        Args:
            file_path (str, optional): Path to the data file. If None, uses self.data_path
            
        Returns:
            pd.DataFrame: Loaded data
        """
        path = file_path or self.data_path
        if not path:
            raise ValueError("No data path provided")
            
        try:
            file_extension = Path(path).suffix.lower()
            if file_extension == '.csv':
                self.data = pd.read_csv(path)
            elif file_extension in ['.xls', '.xlsx']:
                self.data = pd.read_excel(path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
            logger.info(f"Successfully loaded data from {path}")
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def clean_data(self) -> pd.DataFrame:
        """
        Perform basic data cleaning operations.
        
        Returns:
            pd.DataFrame: Cleaned data
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        try:
            # Remove duplicate rows
            self.data = self.data.drop_duplicates()
            
            # Handle missing values
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            categorical_columns = self.data.select_dtypes(include=['object']).columns
            
            # Fill numeric missing values with median
            for col in numeric_columns:
                self.data[col] = self.data[col].fillna(self.data[col].median())
                
            # Fill categorical missing values with mode
            for col in categorical_columns:
                self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
                
            logger.info("Data cleaning completed successfully")
            return self.data
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise
            
    def preprocess_data(self, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess data for machine learning.
        
        Args:
            target_column (str): Name of the target variable column
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target variable
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        try:
            # Separate features and target
            X = self.data.drop(columns=[target_column])
            y = self.data[target_column]
            
            # Convert categorical variables to dummy variables
            X = pd.get_dummies(X, drop_first=True)
            
            logger.info("Data preprocessing completed successfully")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise 