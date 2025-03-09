"""
Support Vector Machine (SVM) Model Implementation

This module provides a Support Vector Machine model implementation that follows
the standardized template structure. SVMs are powerful classifiers that work well
for both linear and non-linear classification tasks, and are especially effective
for handwriting recognition and image classification.
"""

import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.model_template import MLModel


class SVMModel(MLModel):
    """
    Support Vector Machine model for classification tasks.
    
    Use case: Handwriting recognition, image classification, and other high-dimensional data.
    SVMs are particularly effective when the number of dimensions is greater than the number
    of samples and work well on both linear and non-linear problems using different kernels.
    """
    
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', degree=3, probability=True, **kwargs):
        """
        Initialize the Support Vector Machine model.
        
        Args:
            kernel: Specifies the kernel type ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')
            C: Regularization parameter. Higher values -> less regularization
            gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid' kernels
            degree: Degree of the polynomial kernel function ('poly')
            probability: Whether to enable probability estimates (required for predict_proba)
            **kwargs: Additional parameters for the sklearn SVC model
        """
        super().__init__(**kwargs)
        self.model_type = f"Support Vector Machine ({kernel.upper()})"
        self.model_params = {
            'kernel': kernel,
            'C': C,
            'gamma': gamma,
            'degree': degree,
            'probability': probability,
            **kwargs
        }
        
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            degree=degree,
            probability=probability,
            **kwargs
        )
    
    def train(self, X_train, y_train, **kwargs):
        """
        Train the SVM model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training labels
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
        
        Args:
            X: Features to make predictions on
            
        Returns:
            Class probabilities for each sample
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        if not self.model_params.get('probability', False):
            raise ValueError("SVM model was not trained with probability=True")
        
        return self.model.predict_proba(X)
    
    def get_support_vectors(self):
        """
        Get the support vectors from the trained model.
        
        Returns:
            Support vectors used by the SVM model
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        return self.model.support_vectors_
    
    def get_n_support(self):
        """
        Get the number of support vectors for each class.
        
        Returns:
            Dictionary mapping class labels to number of support vectors
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        return {class_label: count for class_label, count in zip(self.model.classes_, self.model.n_support_)}
    
    def evaluate(self, X_test, y_test, task_type="classification"):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: True labels for test data
            task_type: Should be "classification" for SVM
            
        Returns:
            Dictionary of evaluation metrics
        """
        if task_type != "classification":
            raise ValueError("SVM is a classification model. task_type must be 'classification'")
            
        return super().evaluate(X_test, y_test, task_type="classification")
    
    def plot_confusion_matrix(self, X_test, y_test, class_names=None):
        """
        Plot the confusion matrix for the test data.
        
        Args:
            X_test: Test features
            y_test: True labels for test data
            class_names: Names of the classes for the axis labels
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
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
    
    def plot_decision_boundary(self, X, y, feature_indices=None, mesh_step_size=0.02):
        """
        Plot the decision boundary for a 2D feature space.
        
        Args:
            X: Features (must be 2D if feature_indices is None)
            y: Labels
            feature_indices: Indices of the two features to plot if X has more than 2 dimensions
            mesh_step_size: Step size for creating the mesh grid
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
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
        
    def plot_support_vectors(self, X, y, feature_indices=None):
        """
        Plot the support vectors for a 2D feature space.
        
        Args:
            X: Features (must be 2D if feature_indices is None)
            y: Labels
            feature_indices: Indices of the two features to plot if X has more than 2 dimensions
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        # If feature indices are provided, select those features only
        if feature_indices is not None:
            if len(feature_indices) != 2:
                raise ValueError("feature_indices must contain exactly 2 indices")
            
            X_plot = X[:, feature_indices]
            support_vectors = self.model.support_vectors_[:, feature_indices]
            feature_names = [f"Feature {feature_indices[0]}", f"Feature {feature_indices[1]}"]
        else:
            if X.shape[1] != 2:
                raise ValueError("X must have exactly 2 features or feature_indices must be provided")
            
            X_plot = X
            support_vectors = self.model.support_vectors_
            feature_names = ["Feature 1", "Feature 2"]
        
        # Plot all points
        plt.figure(figsize=(10, 8))
        plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y, cmap=plt.cm.Paired, alpha=0.8)
        
        # Highlight support vectors
        plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, linewidth=1, 
                   facecolors='none', edgecolors='k')
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.title(f'{self.model_type} - Support Vectors')
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