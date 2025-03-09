"""
Multilayer Perceptron (MLP) Neural Network Implementation

This module provides a Multilayer Perceptron Neural Network implementation that follows
the standardized template structure. MLPs are versatile neural networks that can learn
non-linear relationships and are suitable for a wide range of tasks including classification
and regression problems.
"""

import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import confusion_matrix, mean_squared_error
import seaborn as sns

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.model_template import MLModel


class MLPModel(MLModel):
    """
    Multilayer Perceptron (MLP) Neural Network model for both classification and regression tasks.
    
    Use case: Complex tasks requiring non-linear mappings between inputs and outputs, 
    such as image classification, pattern recognition, and function approximation.
    """
    
    def __init__(self, task_type="classification", hidden_layer_sizes=(100,), 
                 activation="relu", solver="adam", alpha=0.0001, 
                 batch_size="auto", learning_rate="constant", 
                 learning_rate_init=0.001, max_iter=200, **kwargs):
        """
        Initialize the MLP model.
        
        Args:
            task_type: Either "classification" or "regression"
            hidden_layer_sizes: Number of neurons in each hidden layer
            activation: Activation function ('identity', 'logistic', 'tanh', 'relu')
            solver: The solver for weight optimization ('lbfgs', 'sgd', 'adam')
            alpha: L2 regularization parameter
            batch_size: Size of minibatches for stochastic optimizers
            learning_rate: Learning rate schedule ('constant', 'invscaling', 'adaptive')
            learning_rate_init: Initial learning rate
            max_iter: Maximum number of iterations
            **kwargs: Additional parameters for the sklearn MLP models
        """
        super().__init__(**kwargs)
        self.task_type = task_type
        self.model_type = f"MLP Neural Network ({task_type})"
        self.model_params = {
            'task_type': task_type,
            'hidden_layer_sizes': hidden_layer_sizes,
            'activation': activation,
            'solver': solver,
            'alpha': alpha,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'learning_rate_init': learning_rate_init,
            'max_iter': max_iter,
            **kwargs
        }
        
        # Initialize the appropriate model based on the task type
        if task_type == "classification":
            self.model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                solver=solver,
                alpha=alpha,
                batch_size=batch_size,
                learning_rate=learning_rate,
                learning_rate_init=learning_rate_init,
                max_iter=max_iter,
                **kwargs
            )
        elif task_type == "regression":
            self.model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                solver=solver,
                alpha=alpha,
                batch_size=batch_size,
                learning_rate=learning_rate,
                learning_rate_init=learning_rate_init,
                max_iter=max_iter,
                **kwargs
            )
        else:
            raise ValueError("task_type must be either 'classification' or 'regression'")
    
    def train(self, X_train, y_train, **kwargs):
        """
        Train the MLP model on the provided data.
        
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
    
    def get_loss_curve(self):
        """
        Get the loss curve during training.
        
        Returns:
            Array of loss values at each iteration
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        if not hasattr(self.model, 'loss_curve_'):
            raise ValueError("Loss curve is not available. Make sure the model has been trained.")
        
        return self.model.loss_curve_
    
    def plot_loss_curve(self):
        """
        Plot the loss curve during training.
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        if not hasattr(self.model, 'loss_curve_'):
            raise ValueError("Loss curve is not available. Make sure the model has been trained.")
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.model.loss_curve_)
        plt.title('Loss Curve During Training')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
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
        
        # Calculate and display MSE
        mse = mean_squared_error(y_test, y_pred)
        plt.annotate(f'MSE: {mse:.4f}', xy=(0.05, 0.95), xycoords='axes fraction')
        
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
        
    def visualize_network_architecture(self):
        """
        Visualize the architecture of the neural network.
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        # Extract layer sizes
        layer_sizes = [self.model.n_features_in_] + list(self.model.hidden_layer_sizes) + [self.model.n_outputs_]
        
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        # Hide axes
        ax.axis('off')
        
        # Number of layers
        n_layers = len(layer_sizes)
        
        # Set vertical positions for layers
        layer_top = 0.2
        layer_bottom = 0.8
        
        # Set horizontal positions for neurons
        horizontal_positions = []
        for i, layer_size in enumerate(layer_sizes):
            horizontal_positions.append(np.linspace(0, 1, layer_size + 2)[1:-1])
        
        # Plot nodes
        for i, (layer_pos, layer_size) in enumerate(zip(np.linspace(0, 1, n_layers), layer_sizes)):
            layer_name = "Input" if i == 0 else "Output" if i == n_layers - 1 else f"Hidden {i}"
            
            # Plot the nodes
            node_colors = ['#1f77b4' if i == 0 else '#2ca02c' if i == n_layers - 1 else '#ff7f0e']
            for j, pos in enumerate(horizontal_positions[i]):
                circle = plt.Circle((layer_pos, pos), 0.02, color=node_colors[0], fill=True)
                ax.add_patch(circle)
                
                # Add node labels for small networks
                if max(layer_sizes) < 20:
                    if i == 0:
                        ax.text(layer_pos - 0.1, pos, f"X{j+1}", ha='right', va='center')
                    elif i == n_layers - 1:
                        if self.task_type == "classification":
                            ax.text(layer_pos + 0.1, pos, f"Class {j+1}", ha='left', va='center')
                        else:
                            ax.text(layer_pos + 0.1, pos, f"Y", ha='left', va='center')
            
            # Add layer labels
            ax.text(layer_pos, layer_top + 0.1, layer_name, ha='center', va='center', fontsize=12)
            ax.text(layer_pos, layer_bottom - 0.1, f"Size: {layer_size}", ha='center', va='center')
        
        # Plot edges between nodes
        for i in range(n_layers - 1):
            for j, pos_a in enumerate(horizontal_positions[i]):
                for k, pos_b in enumerate(horizontal_positions[i + 1]):
                    # Only plot a subset of connections for large networks to avoid clutter
                    if max(layer_sizes) > 20 and np.random.random() > 0.1:
                        continue
                    ax.plot([i/(n_layers-1), (i+1)/(n_layers-1)], [pos_a, pos_b], 'k-', alpha=0.1)
        
        plt.title(f'{self.model_type} - Network Architecture\nHidden Layers: {self.model.hidden_layer_sizes}', fontsize=14)
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