"""
Gradient Boosting Model Implementation

This module provides a Gradient Boosting model implementation that follows
the standardized template structure. Gradient Boosting is a powerful ensemble method
that combines multiple weak learners (typically decision trees) to create a strong
predictive model for both classification and regression tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
import seaborn as sns

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.model_template import MLModel


class GradientBoostingModel(MLModel):
    """
    Gradient Boosting model for both classification and regression tasks.
    
    Use case: Complex problems requiring high predictive accuracy, such as:
    - Classification tasks like fraud detection, customer churn prediction
    - Regression tasks like house price prediction, demand forecasting
    """
    
    def __init__(self, task_type="classification", n_estimators=100, learning_rate=0.1, 
                 max_depth=3, min_samples_split=2, min_samples_leaf=1, 
                 subsample=1.0, max_features=None, **kwargs):
        """
        Initialize the Gradient Boosting model.
        
        Args:
            task_type: Either "classification" or "regression"
            n_estimators: Number of boosting stages (trees)
            learning_rate: Shrinks the contribution of each tree
            max_depth: Maximum depth of individual trees
            min_samples_split: Minimum samples required to split an internal node
            min_samples_leaf: Minimum samples required at a leaf node
            subsample: Fraction of samples used for fitting each tree
            max_features: Number of features to consider for best split
            **kwargs: Additional parameters for the sklearn Gradient Boosting model
        """
        super().__init__(**kwargs)
        self.task_type = task_type
        self.model_type = f"Gradient Boosting ({task_type.capitalize()})"
        self.model_params = {
            'task_type': task_type,
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'subsample': subsample,
            'max_features': max_features,
            **kwargs
        }
        
        # Initialize the appropriate model based on the task type
        if task_type == "classification":
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                subsample=subsample,
                max_features=max_features,
                **kwargs
            )
        elif task_type == "regression":
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                subsample=subsample,
                max_features=max_features,
                **kwargs
            )
        else:
            raise ValueError("task_type must be either 'classification' or 'regression'")
    
    def train(self, X_train, y_train, **kwargs):
        """
        Train the Gradient Boosting model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training labels or target values
            **kwargs: Additional training parameters
            
        Returns:
            self: The trained model instance
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True
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
    
    def predict_proba(self, X):
        """
        Predict class probabilities for the input samples.
        Only available for classification.
        
        Args:
            X: Features to make predictions on
            
        Returns:
            Class probabilities for each sample
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification tasks")
        
        return self.model.predict_proba(X)
    
    def get_feature_importances(self):
        """
        Get the feature importances from the trained model.
        
        Returns:
            Feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        return self.model.feature_importances_
    
    def plot_feature_importances(self, feature_names=None, top_n=None):
        """
        Plot the feature importances.
        
        Args:
            feature_names: Names of the features (if None, indices are used)
            top_n: Number of top features to plot (if None, all features are plotted)
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        importances = self.model.feature_importances_
        
        # If feature names are not provided, use indices
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importances))]
        
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        
        # If top_n is specified, take only the top_n features
        if top_n is not None:
            indices = indices[:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'{self.model_type} - Feature Importances')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()
    
    def plot_training_stages(self, X_test, y_test):
        """
        Plot the deviance at different stages of training.
        
        Args:
            X_test: Test features
            y_test: True labels or target values for test data
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        # Only GradientBoostingRegressor has train_score_
        if hasattr(self.model, 'train_score_'):
            train_score = self.model.train_score_
            
            # Calculate test score at each iteration
            test_score = np.zeros((self.model.n_estimators,), dtype=np.float64)
            for i, y_pred in enumerate(self.model.staged_predict(X_test)):
                test_score[i] = self.model.loss_(y_test, y_pred)
            
            plt.figure(figsize=(10, 6))
            plt.title('Deviance at Different Stages (Trees)')
            plt.plot(np.arange(self.model.n_estimators) + 1, train_score, 'b-',
                    label='Training Set Deviance')
            plt.plot(np.arange(self.model.n_estimators) + 1, test_score, 'r-',
                    label='Test Set Deviance')
            plt.legend(loc='upper right')
            plt.xlabel('Boosting Iterations (Trees)')
            plt.ylabel('Deviance')
            plt.show()
        else:
            print("Training stage plot is only available for regression.")
    
    def plot_learning_curve(self, X_train, y_train, X_test, y_test, 
                          n_estimators_range=None):
        """
        Plot the learning curve over different numbers of trees.
        
        Args:
            X_train: Training features
            y_train: Training labels or target values
            X_test: Test features
            y_test: Test labels or target values
            n_estimators_range: Range of n_estimators to evaluate
        """
        if n_estimators_range is None:
            n_estimators_range = np.linspace(10, self.model.n_estimators, 10).astype(int)
        
        train_scores = []
        test_scores = []
        
        for n_estimators in n_estimators_range:
            # Create a new model with the same parameters but different n_estimators
            if self.task_type == "classification":
                temp_model = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=self.model.learning_rate,
                    max_depth=self.model.max_depth,
                    min_samples_split=self.model.min_samples_split,
                    min_samples_leaf=self.model.min_samples_leaf,
                    subsample=self.model.subsample,
                    max_features=self.model.max_features,
                    random_state=self.model.random_state if hasattr(self.model, 'random_state') else None
                )
            else:
                temp_model = GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    learning_rate=self.model.learning_rate,
                    max_depth=self.model.max_depth,
                    min_samples_split=self.model.min_samples_split,
                    min_samples_leaf=self.model.min_samples_leaf,
                    subsample=self.model.subsample,
                    max_features=self.model.max_features,
                    random_state=self.model.random_state if hasattr(self.model, 'random_state') else None
                )
            
            temp_model.fit(X_train, y_train)
            
            # Calculate scores
            train_scores.append(temp_model.score(X_train, y_train))
            test_scores.append(temp_model.score(X_test, y_test))
        
        plt.figure(figsize=(10, 6))
        plt.title('Learning Curve: Score vs Number of Trees')
        plt.plot(n_estimators_range, train_scores, 'o-', label='Training Score')
        plt.plot(n_estimators_range, test_scores, 'o-', label='Test Score')
        plt.legend(loc='best')
        plt.xlabel('Number of Trees')
        plt.ylabel('Score')
        plt.grid(True)
        plt.show()
        
    def evaluate(self, X_test, y_test, task_type=None):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: True labels or target values for test data
            task_type: If None, uses the task_type specified during initialization
            
        Returns:
            Dictionary of evaluation metrics
        """
        if task_type is None:
            task_type = self.task_type
            
        return super().evaluate(X_test, y_test, task_type=task_type)
    
    def plot_confusion_matrix(self, X_test, y_test, class_names=None):
        """
        Plot the confusion matrix for the test data.
        Only available for classification tasks.
        
        Args:
            X_test: Test features
            y_test: True labels for test data
            class_names: Names of the classes for the axis labels
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        if self.task_type != "classification":
            raise ValueError("Confusion matrix is only available for classification tasks")
        
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names if class_names else "auto",
                   yticklabels=class_names if class_names else "auto")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{self.model_type} - Confusion Matrix')
        plt.show()
    
    def plot_prediction_error(self, X_test, y_test):
        """
        Plot the prediction error for regression tasks.
        
        Args:
            X_test: Test features
            y_test: True target values for test data
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        if self.task_type != "regression":
            raise ValueError("Prediction error plot is only available for regression tasks")
        
        y_pred = self.predict(X_test)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{self.model_type} - Actual vs Predicted')
        
        # Calculate and display MSE and R^2
        mse = mean_squared_error(y_test, y_pred)
        r2 = self.model.score(X_test, y_test)
        plt.annotate(f'MSE: {mse:.4f}\nR²: {r2:.4f}', xy=(0.05, 0.95), xycoords='axes fraction')
        
        plt.show()
    
    def plot_decision_boundary(self, X, y, feature_indices=None, mesh_step_size=0.02):
        """
        Plot the decision boundary for a 2D feature space.
        Only available for classification tasks.
        
        Args:
            X: Features (must be 2D if feature_indices is None)
            y: Labels
            feature_indices: Indices of the two features to plot if X has more than 2 dimensions
            mesh_step_size: Step size for creating the mesh grid
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        if self.task_type != "classification":
            raise ValueError("Decision boundary plot is only available for classification tasks")
        
        # If feature indices are provided, select those features only
        if feature_indices is not None:
            if len(feature_indices) != 2:
                raise ValueError("feature_indices must contain exactly 2 indices")
            
            X_plot = X[:, feature_indices]
            feature_names = [f"Feature {feature_indices[0]}", f"Feature {feature_indices[1]}"]
        else:
            if X.shape[1] != 2:
                raise ValueError("X must have exactly 2 features or feature_indices must be provided")
            
            X_plot = X
            feature_names = ["Feature 1", "Feature 2"]
        
        # Create a mesh grid
        x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
        y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                             np.arange(y_min, y_max, mesh_step_size))
        
        # Make predictions on the mesh grid
        if feature_indices is not None:
            # Create a dummy array with zeros for all features
            mesh_input = np.zeros((xx.ravel().shape[0], X.shape[1]))
            # Fill in the two features we're visualizing
            mesh_input[:, feature_indices[0]] = xx.ravel()
            mesh_input[:, feature_indices[1]] = yy.ravel()
            Z = self.model.predict(mesh_input)
        else:
            Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        
        Z = Z.reshape(xx.shape)
        
        # Plot the decision boundary
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.8)
        
        # Plot the training points
        scatter = plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y, edgecolors='k', alpha=0.8)
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.title(f'{self.model_type} - Decision Boundary')
        plt.colorbar(scatter)
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
        print(f"Model saved to {filepath}")
    
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
        print(f"Model loaded from {filepath}")
        return self 