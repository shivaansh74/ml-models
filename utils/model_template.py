"""
Template for Machine Learning Model Implementation

This template provides a standardized structure for implementing machine learning models
in this repository. Each model implementation should follow this structure to maintain
consistency and ease of use.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score


class MLModel:
    """Base class for all machine learning models in this repository."""
    
    def __init__(self, **kwargs):
        """
        Initialize the model with parameters.
        
        Args:
            **kwargs: Model specific parameters.
        """
        self.model = None
        self.is_trained = False
        self.model_type = "Base Model"
        self.model_params = kwargs
    
    def train(self, X_train, y_train, **kwargs):
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training labels/targets
            **kwargs: Additional training parameters
            
        Returns:
            self: The trained model instance
        """
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Features to make predictions on
            
        Returns:
            Predictions from the model
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def evaluate(self, X_test, y_test, task_type="classification"):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: True labels/targets for test data
            task_type: Either "classification" or "regression"
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        predictions = self.predict(X_test)
        
        if task_type == "classification":
            metrics = {
                'accuracy': accuracy_score(y_test, predictions),
                'precision': precision_score(y_test, predictions, average='weighted'),
                'recall': recall_score(y_test, predictions, average='weighted'),
                'f1_score': f1_score(y_test, predictions, average='weighted')
            }
        elif task_type == "regression":
            metrics = {
                'mse': mean_squared_error(y_test, predictions),
                'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                'r2': r2_score(y_test, predictions)
            }
        else:
            raise ValueError("task_type must be either 'classification' or 'regression'")
        
        return metrics
    
    def plot_results(self, X_test, y_test, task_type="classification"):
        """
        Plot the results of the model.
        
        Args:
            X_test: Test features
            y_test: True labels/targets for test data
            task_type: Either "classification" or "regression"
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        predictions = self.predict(X_test)
        
        plt.figure(figsize=(10, 6))
        
        if task_type == "classification":
            # Plot confusion matrix or classification report
            # Implementation details will depend on the specific model
            pass
        elif task_type == "regression":
            # Plot actual vs predicted values
            plt.scatter(y_test, predictions, alpha=0.5)
            plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title(f'{self.model_type} - Actual vs Predicted')
        else:
            raise ValueError("task_type must be either 'classification' or 'regression'")
        
        plt.show()
    
    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def load_model(self, filepath):
        """
        Load a trained model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            self: The loaded model instance
        """
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def __str__(self):
        """Return a string representation of the model."""
        return f"{self.model_type} with parameters: {self.model_params}" 