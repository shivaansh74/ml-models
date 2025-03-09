"""
Linear Regression Model Implementation

This module provides a Linear Regression model implementation that follows
the standardized template structure. Linear Regression is used for predicting
numerical values, such as house prices based on features.
"""

import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.model_template import MLModel


class LinearRegressionModel(MLModel):
    """
    Linear Regression model for regression tasks.
    
    Use case: Predicting house prices based on features such as square footage,
    number of bedrooms, location, etc.
    """
    
    def __init__(self, fit_intercept=True, normalize=False, n_jobs=None, **kwargs):
        """
        Initialize the Linear Regression model.
        
        Args:
            fit_intercept: Whether to calculate the intercept for this model
            normalize: Deprecated since scikit-learn 1.0. Use preprocessing.StandardScaler instead
            n_jobs: Number of jobs to use for the computation
            **kwargs: Additional parameters for the sklearn LinearRegression model
        """
        super().__init__(**kwargs)
        self.model_type = "Linear Regression"
        self.model_params = {
            'fit_intercept': fit_intercept,
            'normalize': normalize,
            'n_jobs': n_jobs,
            **kwargs
        }
        
        self.model = SklearnLinearRegression(
            fit_intercept=fit_intercept,
            n_jobs=n_jobs,
            **kwargs
        )
    
    def train(self, X_train, y_train, **kwargs):
        """
        Train the Linear Regression model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            **kwargs: Additional training parameters
            
        Returns:
            self: The trained model instance
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Store model coefficients and intercept for interpretation
        self.coefficients = self.model.coef_
        self.intercept = self.model.intercept_
        
        return self
    
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
        
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test, task_type="regression"):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: True targets for test data
            task_type: Must be "regression" for Linear Regression
            
        Returns:
            Dictionary of evaluation metrics
        """
        if task_type != "regression":
            raise ValueError("Linear Regression is a regression model. Task type must be 'regression'.")
        
        return super().evaluate(X_test, y_test, task_type="regression")
    
    def get_feature_importance(self, feature_names=None):
        """
        Get the feature importance (coefficients) from the model.
        
        Args:
            feature_names: Names of the features (optional)
            
        Returns:
            DataFrame with feature names and their coefficients
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        import pandas as pd
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.coefficients))]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': self.coefficients
        })
        
        # Sort by absolute coefficient value for better visualization
        importance_df['Abs_Coefficient'] = importance_df['Coefficient'].abs()
        importance_df = importance_df.sort_values('Abs_Coefficient', ascending=False)
        importance_df = importance_df.drop(columns=['Abs_Coefficient'])
        
        return importance_df
    
    def plot_feature_importance(self, feature_names=None, top_n=10):
        """
        Plot the feature importance (coefficients) from the model.
        
        Args:
            feature_names: Names of the features (optional)
            top_n: Number of top features to display
            
        Returns:
            matplotlib plot of feature importance
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        importance_df = self.get_feature_importance(feature_names)
        
        # Take top N features
        importance_df = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Coefficient'])
        plt.xlabel('Coefficient')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Coefficients')
        plt.tight_layout()
        plt.show()
    
    def plot_residuals(self, X, y_true):
        """
        Plot the residuals (errors) of the model predictions.
        
        Args:
            X: Features
            y_true: True targets
            
        Returns:
            matplotlib plot of residuals
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        y_pred = self.predict(X)
        residuals = y_true - y_pred
        
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Residuals vs Predicted values
        plt.subplot(1, 2, 1)
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted Values')
        
        # Subplot 2: Residuals histogram
        plt.subplot(1, 2, 2)
        plt.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0, color='r', linestyle='-')
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.title('Residuals Distribution')
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath):
        """
        Load a trained model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            self: The loaded model instance
        """
        self.model = joblib.load(filepath)
        self.is_trained = True
        
        # Update model coefficients and intercept
        self.coefficients = self.model.coef_
        self.intercept = self.model.intercept_
        
        return self


# Example usage
if __name__ == "__main__":
    # Sample data generation
    np.random.seed(42)
    X = 2 * np.random.rand(100, 5)
    y = 4 + 3 * X[:, 0] + 2 * X[:, 1] - 1 * X[:, 2] + 0.5 * X[:, 3] + np.random.randn(100)
    
    # Feature names
    feature_names = ['size', 'bedrooms', 'location', 'age', 'bathrooms']
    
    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = LinearRegressionModel()
    model.train(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    print("Performance metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Feature importance
    print("\nFeature Importance:")
    importance_df = model.get_feature_importance(feature_names)
    print(importance_df)
    
    # Plot results
    model.plot_results(X_test, y_test, task_type="regression")
    
    # Plot feature importance
    model.plot_feature_importance(feature_names)
    
    # Plot residuals
    model.plot_residuals(X_test, y_test) 