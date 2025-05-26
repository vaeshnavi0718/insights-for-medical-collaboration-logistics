"""
Visualization module for Medical Collaboration Logistics (MCL).
Handles data visualization and storytelling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional, Union
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Visualizer:
    def __init__(self, style: str = 'seaborn'):
        """
        Initialize the Visualizer.
        
        Args:
            style (str): Plotting style to use
        """
        plt.style.use(style)
        self.figures = []
        
    def plot_distribution(self, data: pd.Series, title: str, 
                         bins: int = 30, figsize: tuple = (10, 6)) -> None:
        """
        Plot distribution of a numerical variable.
        
        Args:
            data (pd.Series): Data to plot
            title (str): Plot title
            bins (int): Number of histogram bins
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize)
        sns.histplot(data=data, bins=bins, kde=True)
        plt.title(title)
        plt.xlabel(data.name)
        plt.ylabel('Frequency')
        self.figures.append(plt.gcf())
        
    def plot_correlation_matrix(self, data: pd.DataFrame, 
                              figsize: tuple = (12, 8)) -> None:
        """
        Plot correlation matrix heatmap.
        
        Args:
            data (pd.DataFrame): Data to plot
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize)
        corr_matrix = data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        self.figures.append(plt.gcf())
        
    def plot_feature_importance(self, importance: pd.Series, 
                              title: str = 'Feature Importance',
                              figsize: tuple = (10, 6)) -> None:
        """
        Plot feature importance.
        
        Args:
            importance (pd.Series): Feature importance values
            title (str): Plot title
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize)
        importance.sort_values().plot(kind='barh')
        plt.title(title)
        plt.xlabel('Importance')
        self.figures.append(plt.gcf())
        
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            labels: Optional[List[str]] = None,
                            figsize: tuple = (8, 6)) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            labels (List[str], optional): Label names
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize)
        cm = pd.crosstab(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        if labels:
            plt.xticks(ticks=range(len(labels)), labels=labels)
            plt.yticks(ticks=range(len(labels)), labels=labels)
        self.figures.append(plt.gcf())
        
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      figsize: tuple = (8, 6)) -> None:
        """
        Plot ROC curve.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred_proba (np.ndarray): Predicted probabilities
            figsize (tuple): Figure size
        """
        from sklearn.metrics import roc_curve, auc
        
        plt.figure(figsize=figsize)
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        self.figures.append(plt.gcf())
        
    def create_interactive_scatter(self, data: pd.DataFrame, x: str, y: str,
                                 color: Optional[str] = None,
                                 title: str = 'Interactive Scatter Plot') -> None:
        """
        Create interactive scatter plot using Plotly.
        
        Args:
            data (pd.DataFrame): Data to plot
            x (str): Column name for x-axis
            y (str): Column name for y-axis
            color (str, optional): Column name for color encoding
            title (str): Plot title
        """
        fig = px.scatter(data, x=x, y=y, color=color, title=title)
        fig.show()
        
    def create_interactive_heatmap(self, data: pd.DataFrame,
                                 title: str = 'Interactive Correlation Heatmap') -> None:
        """
        Create interactive correlation heatmap using Plotly.
        
        Args:
            data (pd.DataFrame): Data to plot
            title (str): Plot title
        """
        corr_matrix = data.corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        fig.update_layout(title=title)
        fig.show()
        
    def save_figures(self, directory: str = 'figures') -> None:
        """
        Save all generated figures to disk.
        
        Args:
            directory (str): Directory to save figures
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        for i, fig in enumerate(self.figures):
            fig.savefig(f'{directory}/figure_{i+1}.png', bbox_inches='tight', dpi=300)
            plt.close(fig)
            
        logger.info(f"Saved {len(self.figures)} figures to {directory}")
        self.figures = [] 