"""
Recurrent Neural Network (RNN) Implementation

This module provides a Recurrent Neural Network implementation that follows
the standardized template structure. RNNs are specialized neural networks designed
for sequential data processing, making them suitable for time series analysis,
natural language processing, and other sequence-based tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.model_template import MLModel


class RNNModel(MLModel):
    """
    Recurrent Neural Network (RNN) model for sequence data processing.
    
    Use case: Time series prediction, sentiment analysis, text classification,
    and other tasks involving sequential or temporal data.
    """
    
    def __init__(self, task_type="classification", units=50, activation="tanh", 
                 dropout_rate=0.2, optimizer="adam", loss=None, input_shape=None,
                 batch_size=32, epochs=100, validation_split=0.2, **kwargs):
        """
        Initialize the RNN model.
        
        Args:
            task_type: Either "classification" or "regression"
            units: Number of RNN units (neurons)
            activation: Activation function for RNN layer
            dropout_rate: Dropout rate to prevent overfitting
            optimizer: Optimizer for training ('adam', 'rmsprop', etc.)
            loss: Loss function. If None, will be set automatically based on task_type
            input_shape: Shape of input data (timesteps, features)
            batch_size: Number of samples per gradient update
            epochs: Number of epochs to train the model
            validation_split: Fraction of training data to use as validation
            **kwargs: Additional parameters for the Keras RNN model
        """
        super().__init__(**kwargs)
        self.task_type = task_type
        self.model_type = f"Recurrent Neural Network (RNN) ({task_type})"
        self.units = units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.input_shape = input_shape
        self.history = None
        self.scaler = StandardScaler()
        
        # Set default loss function based on task type
        if loss is None:
            if task_type == "classification":
                self.loss = "categorical_crossentropy"
            else:  # regression
                self.loss = "mean_squared_error"
        else:
            self.loss = loss
        
        self.model_params = {
            'task_type': task_type,
            'units': units,
            'activation': activation,
            'dropout_rate': dropout_rate,
            'optimizer': optimizer,
            'loss': self.loss,
            'input_shape': input_shape,
            'batch_size': batch_size,
            'epochs': epochs,
            'validation_split': validation_split,
            **kwargs
        }
        
        # The model will be built when input_shape is provided either in init or during training
        self.model = None
    
    def _build_model(self, input_shape, num_classes=None):
        """
        Build the RNN model architecture.
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            num_classes: Number of output classes for classification tasks
        """
        model = Sequential()
        
        # Add RNN layer
        model.add(SimpleRNN(
            units=self.units,
            activation=self.activation,
            input_shape=input_shape,
            return_sequences=False
        ))
        
        # Add dropout for regularization
        model.add(Dropout(self.dropout_rate))
        
        # Add output layer based on task type
        if self.task_type == "classification":
            model.add(Dense(num_classes, activation='softmax'))
            metrics = ['accuracy']
        else:  # regression
            model.add(Dense(1))
            metrics = ['mae']
        
        # Compile the model
        model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=metrics
        )
        
        return model
    
    def train(self, X_train, y_train, **kwargs):
        """
        Train the RNN model on the provided data.
        
        Args:
            X_train: Training features (should be shaped for RNN: [samples, timesteps, features])
            y_train: Training labels or target values
            **kwargs: Additional training parameters
            
        Returns:
            self: The trained model instance
        """
        # Update parameters if provided
        for key, value in kwargs.items():
            if key in self.model_params:
                self.model_params[key] = value
                setattr(self, key, value)
        
        # Get input shape from data if not provided
        if self.input_shape is None:
            if len(X_train.shape) == 2:
                # Reshape 2D data to 3D (samples, timesteps, features)
                # Assume each sample is a single timestep if not explicitly provided
                self.input_shape = (1, X_train.shape[1])
                X_train = X_train.reshape(-1, 1, X_train.shape[1])
            else:
                self.input_shape = (X_train.shape[1], X_train.shape[2])
        
        # Normalize data for numerical stability
        if len(X_train.shape) == 3:
            # Reshape to 2D for scaling
            original_shape = X_train.shape
            X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
            X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
            # Reshape back to 3D
            X_train = X_train_scaled.reshape(original_shape)
        else:
            X_train = self.scaler.fit_transform(X_train)
        
        # Prepare target data based on task type
        if self.task_type == "classification":
            # Convert to one-hot encoding for classification
            num_classes = len(np.unique(y_train))
            y_train_encoded = to_categorical(y_train, num_classes=num_classes)
        else:
            # For regression, no transformation needed
            y_train_encoded = y_train
            num_classes = None
        
        # Build the model if not already built
        if self.model is None:
            self.model = self._build_model(self.input_shape, num_classes)
        
        # Set up early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train_encoded,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        self.is_trained = True
        return self
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Features to make predictions on (should be shaped for RNN)
            
        Returns:
            Predicted values
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Prepare input data
        if len(X.shape) == 2:
            # Reshape 2D data to 3D (samples, timesteps, features)
            X = X.reshape(-1, 1, X.shape[1])
        
        # Scale input data
        if len(X.shape) == 3:
            # Reshape to 2D for scaling
            original_shape = X.shape
            X_reshaped = X.reshape(X.shape[0], -1)
            X_scaled = self.scaler.transform(X_reshaped)
            # Reshape back to 3D
            X = X_scaled.reshape(original_shape)
        else:
            X = self.scaler.transform(X)
        
        # Get predictions
        predictions = self.model.predict(X)
        
        # Convert from one-hot encoding for classification
        if self.task_type == "classification":
            return np.argmax(predictions, axis=1)
        
        return predictions
    
    def predict_proba(self, X):
        """
        Return probability estimates for classification tasks.
        
        Args:
            X: Features to make predictions on
            
        Returns:
            Probability estimates for each class
        """
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification tasks")
        
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Prepare input data
        if len(X.shape) == 2:
            # Reshape 2D data to 3D (samples, timesteps, features)
            X = X.reshape(-1, 1, X.shape[1])
        
        # Scale input data
        if len(X.shape) == 3:
            # Reshape to 2D for scaling
            original_shape = X.shape
            X_reshaped = X.reshape(X.shape[0], -1)
            X_scaled = self.scaler.transform(X_reshaped)
            # Reshape back to 3D
            X = X_scaled.reshape(original_shape)
        else:
            X = self.scaler.transform(X)
        
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test, task_type=None):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels or target values
            task_type: Override the task type for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        if task_type is None:
            task_type = self.task_type
        
        y_pred = self.predict(X_test)
        
        if task_type == "classification":
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            return {
                'accuracy': accuracy,
                'classification_report': report
            }
        else:  # regression
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            return {
                'mse': mse,
                'rmse': rmse
            }
    
    def plot_training_history(self):
        """
        Plot the training and validation loss and metrics.
        
        Returns:
            Matplotlib figure
        """
        if self.history is None:
            raise ValueError("Model has not been trained yet or history was not saved.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training & validation loss
        ax1.plot(self.history.history['loss'])
        ax1.plot(self.history.history['val_loss'])
        ax1.set_title('Model Loss')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Validation'], loc='upper right')
        ax1.grid(True)
        
        # Plot training & validation accuracy or MAE
        metric_key = 'accuracy' if self.task_type == 'classification' else 'mae'
        if metric_key in self.history.history:
            ax2.plot(self.history.history[metric_key])
            ax2.plot(self.history.history[f'val_{metric_key}'])
            metric_name = 'Accuracy' if self.task_type == 'classification' else 'Mean Absolute Error'
            ax2.set_title(f'Model {metric_name}')
            ax2.set_ylabel(metric_name)
            ax2.set_xlabel('Epoch')
            ax2.legend(['Train', 'Validation'], loc='lower right' if self.task_type == 'classification' else 'upper right')
            ax2.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, X_test, y_test, class_names=None):
        """
        Plot the confusion matrix for classification tasks.
        
        Args:
            X_test: Test features
            y_test: Test labels
            class_names: Names of the classes
            
        Returns:
            Matplotlib figure
        """
        if self.task_type != "classification":
            raise ValueError("Confusion matrix is only available for classification tasks")
        
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        
        if class_names is not None:
            ax.set_xticklabels(class_names)
            ax.set_yticklabels(class_names)
        
        plt.tight_layout()
        return fig
    
    def plot_prediction_comparison(self, X_test, y_test, n_samples=10):
        """
        Plot the comparison between actual and predicted values for regression tasks.
        
        Args:
            X_test: Test features
            y_test: Test target values
            n_samples: Number of samples to display
            
        Returns:
            Matplotlib figure
        """
        if self.task_type != "regression":
            raise ValueError("Prediction comparison is only available for regression tasks")
        
        y_pred = self.predict(X_test)
        
        # Only show a subset of samples
        indices = np.arange(min(len(y_test), n_samples))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(indices, y_test[indices], 'o-', label='Actual')
        ax.plot(indices, y_pred[indices], 'o-', label='Predicted')
        
        ax.set_title('Time Series Prediction Comparison')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        model_data = {
            'model_params': self.model_params,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        
        # Save Keras model separately
        keras_model_path = filepath + '.keras'
        self.model.save(keras_model_path)
        
        # Save other components
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            self: The loaded model instance
        """
        model_data = joblib.load(filepath)
        
        # Load model parameters
        self.model_params = model_data['model_params']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        
        # Set attributes from model_params
        for key, value in self.model_params.items():
            setattr(self, key, value)
        
        # Load Keras model
        keras_model_path = filepath + '.keras'
        self.model = tf.keras.models.load_model(keras_model_path)
        
        return self 