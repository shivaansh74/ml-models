"""
K-Nearest Neighbors (KNN) Model Implementation

This module provides a K-Nearest Neighbors model implementation that follows
the standardized template structure. KNN is a versatile non-parametric method
used for both classification and regression, where predictions are based on the
closest training examples in the feature space.
"""

import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
import seaborn as sns

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.model_template import MLModel


class KNNModel(MLModel):
    """
    K-Nearest Neighbors model for both classification and regression tasks.
    
    Use case: Classification, regression, and recommendation systems. KNN works 
    particularly well when the decision boundary is irregular and there's enough 
    labeled data available.
    """
    
    def __init__(self, task_type="classification", n_neighbors=5, weights='uniform', 
                 algorithm='auto', leaf_size=30, p=2, metric='minkowski', **kwargs):
        """
        Initialize the KNN model.
        
        Args:
            task_type: Either "classification" or "regression"
            n_neighbors: Number of neighbors to use
            weights: Weight function used in prediction ('uniform', 'distance')
            algorithm: Algorithm used to compute nearest neighbors ('auto', 'ball_tree', 'kd_tree', 'brute')
            leaf_size: Leaf size passed to BallTree or KDTree
            p: Power parameter for the Minkowski metric (p=1: Manhattan, p=2: Euclidean)
            metric: Distance metric to use
            **kwargs: Additional parameters for the sklearn KNN models
        """
        super().__init__(**kwargs)
        self.task_type = task_type
        self.model_type = f"K-Nearest Neighbors ({task_type.capitalize()})"
        self.model_params = {
            'task_type': task_type,
            'n_neighbors': n_neighbors,
            'weights': weights,
            'algorithm': algorithm,
            'leaf_size': leaf_size,
            'p': p,
            'metric': metric,
            **kwargs
        }
        
        # Initialize the appropriate model based on the task type
        if task_type == "classification":
            self.model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                algorithm=algorithm,
                leaf_size=leaf_size,
                p=p,
                metric=metric,
                **kwargs
            )
        elif task_type == "regression":
            self.model = KNeighborsRegressor(
                n_neighbors=n_neighbors,
                weights=weights,
                algorithm=algorithm,
                leaf_size=leaf_size,
                p=p,
                metric=metric,
                **kwargs
            )
        else:
            raise ValueError("task_type must be either 'classification' or 'regression'")
    
    def train(self, X_train, y_train, **kwargs):
        """
        Train the KNN model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training labels or target values
            **kwargs: Additional training parameters
            
        Returns:
            self: The trained model instance
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.X_train = X_train  # Store training data for neighbor plots
        self.y_train = y_train
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
    
    def get_neighbors(self, X, n_neighbors=None):
        """
        Get the k nearest neighbors of test points.
        
        Args:
            X: The query points
            n_neighbors: Number of neighbors to get (defaults to the model's n_neighbors)
            
        Returns:
            Indices of the nearest neighbors in the training data
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        if n_neighbors is None:
            n_neighbors = self.model.n_neighbors
        
        return self.model.kneighbors(X, n_neighbors=n_neighbors, return_distance=False)
    
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
    
    def plot_k_selection(self, X_val, y_val, k_range=None):
        """
        Plot the validation accuracy/error for different values of k (n_neighbors).
        
        Args:
            X_val: Validation features
            y_val: Validation labels or target values
            k_range: Range of k values to try (default: 1 to 30)
        """
        if k_range is None:
            k_range = range(1, 31)
        
        scores = []
        
        for k in k_range:
            # Create a model with the same parameters but different k
            if self.task_type == "classification":
                temp_model = KNeighborsClassifier(
                    n_neighbors=k,
                    weights=self.model.weights,
                    algorithm=self.model.algorithm,
                    leaf_size=self.model.leaf_size,
                    p=self.model.p,
                    metric=self.model.metric
                )
            else:
                temp_model = KNeighborsRegressor(
                    n_neighbors=k,
                    weights=self.model.weights,
                    algorithm=self.model.algorithm,
                    leaf_size=self.model.leaf_size,
                    p=self.model.p,
                    metric=self.model.metric
                )
            
            temp_model.fit(self.X_train, self.y_train)
            scores.append(temp_model.score(X_val, y_val))
        
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, scores, 'o-')
        plt.xlabel('Number of Neighbors (k)')
        
        if self.task_type == "classification":
            plt.ylabel('Validation Accuracy')
            plt.title('KNN Classification: Accuracy vs. k')
        else:
            plt.ylabel('Validation R² Score')
            plt.title('KNN Regression: R² Score vs. k')
        
        # Find the best k
        best_k = k_range[np.argmax(scores)]
        best_score = max(scores)
        plt.axvline(x=best_k, color='r', linestyle='--', 
                   label=f'Best k={best_k}, Score={best_score:.4f}')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        print(f"Best k: {best_k} with {'accuracy' if self.task_type == 'classification' else 'R² score'}: {best_score:.4f}")
        return best_k
    
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
        plt.title(f'{self.model_type} - Decision Boundary (k={self.model.n_neighbors})')
        plt.colorbar(scatter)
        plt.show()
    
    def plot_neighbors(self, X_sample, feature_indices=None, n_neighbors=None):
        """
        Plot the neighbors for a sample point in a 2D feature space.
        
        Args:
            X_sample: Sample point(s) to find neighbors for
            feature_indices: Indices of the two features to plot if the data has more than 2 dimensions
            n_neighbors: Number of neighbors to plot (defaults to the model's n_neighbors)
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        if n_neighbors is None:
            n_neighbors = self.model.n_neighbors
        
        # If X_sample is a single point, convert it to a 2D array
        if len(X_sample.shape) == 1:
            X_sample = X_sample.reshape(1, -1)
        
        # Get the neighbors
        neighbor_indices = self.get_neighbors(X_sample, n_neighbors=n_neighbors)[0]
        
        # If feature indices are provided, select those features only
        if feature_indices is not None:
            if len(feature_indices) != 2:
                raise ValueError("feature_indices must contain exactly 2 indices")
            
            X_train_plot = self.X_train[:, feature_indices]
            X_sample_plot = X_sample[:, feature_indices]
            feature_names = [f"Feature {feature_indices[0]}", f"Feature {feature_indices[1]}"]
        else:
            if self.X_train.shape[1] != 2:
                raise ValueError("Training data must have exactly 2 features or feature_indices must be provided")
            
            X_train_plot = self.X_train
            X_sample_plot = X_sample
            feature_names = ["Feature 1", "Feature 2"]
        
        # Plot all training points
        plt.figure(figsize=(10, 8))
        plt.scatter(X_train_plot[:, 0], X_train_plot[:, 1], c=self.y_train, cmap='viridis', 
                   alpha=0.5, label='Training Data')
        
        # Highlight the neighbors
        plt.scatter(X_train_plot[neighbor_indices, 0], X_train_plot[neighbor_indices, 1], 
                   s=100, facecolors='none', edgecolors='red', linewidth=2, 
                   label=f'{n_neighbors} Nearest Neighbors')
        
        # Plot the sample point
        plt.scatter(X_sample_plot[:, 0], X_sample_plot[:, 1], color='black', s=100, marker='x',
                   label='Sample Point')
        
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.title(f'{self.model_type} - Nearest Neighbors Visualization')
        plt.legend()
        plt.show()
        
        # Return the labels of the neighbors (useful for recommendation systems)
        return self.y_train[neighbor_indices]
    
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