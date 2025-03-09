"""
Long Short-Term Memory (LSTM) Network Implementation

This module provides a Long Short-Term Memory (LSTM) implementation that follows
the standardized template structure. LSTM networks are advanced recurrent neural networks
designed for sequential data with long-term dependencies, making them suitable for
stock market prediction, time series forecasting, and other complex sequential tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.model_template import MLModel


class LSTMModel(MLModel):
    """
    Long Short-Term Memory (LSTM) model for complex sequential data processing.
    
    Use case: Stock market prediction, time series forecasting with long-term dependencies,
    natural language processing, and other tasks requiring memory of past events.
    """
    
    def __init__(self, task_type="regression", units=64, activation="tanh", 
                 recurrent_activation="sigmoid", dropout_rate=0.2, 
                 recurrent_dropout=0.0, optimizer="adam", loss=None,
                 input_shape=None, batch_size=32, epochs=100, 
                 validation_split=0.2, return_sequences=False,
                 use_bidirectional=False, layers=1, **kwargs):
        """
        Initialize the LSTM model.
        
        Args:
            task_type: Either "classification" or "regression"
            units: Number of LSTM units (neurons)
            activation: Activation function for LSTM layer
            recurrent_activation: Activation function for recurrent step
            dropout_rate: Dropout rate to prevent overfitting
            recurrent_dropout: Dropout rate for recurrent connections
            optimizer: Optimizer for training ('adam', 'rmsprop', etc.)
            loss: Loss function. If None, will be set automatically based on task_type
            input_shape: Shape of input data (timesteps, features)
            batch_size: Number of samples per gradient update
            epochs: Number of epochs to train the model
            validation_split: Fraction of training data to use as validation
            return_sequences: Whether to return sequences (for stacked LSTM)
            use_bidirectional: Whether to use bidirectional LSTM
            layers: Number of LSTM layers
            **kwargs: Additional parameters for the Keras LSTM model
        """
        super().__init__(**kwargs)
        self.task_type = task_type
        self.model_type = f"Long Short-Term Memory (LSTM) ({task_type})"
        self.units = units
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.input_shape = input_shape
        self.return_sequences = return_sequences
        self.use_bidirectional = use_bidirectional
        self.layers = layers
        self.history = None
        
        # Use MinMaxScaler for financial data as it's often more suitable
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
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
            'recurrent_activation': recurrent_activation,
            'dropout_rate': dropout_rate,
            'recurrent_dropout': recurrent_dropout,
            'optimizer': optimizer,
            'loss': self.loss,
            'input_shape': input_shape,
            'batch_size': batch_size,
            'epochs': epochs,
            'validation_split': validation_split,
            'return_sequences': return_sequences,
            'use_bidirectional': use_bidirectional,
            'layers': layers,
            **kwargs
        }
        
        # The model will be built when input_shape is provided either in init or during training
        self.model = None
    
    def _build_model(self, input_shape, num_classes=None):
        """
        Build the LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            num_classes: Number of output classes for classification tasks
        """
        model = Sequential()
        
        # Add LSTM layers
        for i in range(self.layers):
            return_sequences = self.return_sequences or (i < self.layers - 1)
            
            # First layer needs input_shape
            if i == 0:
                if self.use_bidirectional:
                    model.add(tf.keras.layers.Bidirectional(
                        LSTM(
                            units=self.units,
                            activation=self.activation,
                            recurrent_activation=self.recurrent_activation,
                            dropout=self.dropout_rate,
                            recurrent_dropout=self.recurrent_dropout,
                            return_sequences=return_sequences,
                            input_shape=input_shape
                        )
                    ))
                else:
                    model.add(LSTM(
                        units=self.units,
                        activation=self.activation,
                        recurrent_activation=self.recurrent_activation,
                        dropout=self.dropout_rate,
                        recurrent_dropout=self.recurrent_dropout,
                        return_sequences=return_sequences,
                        input_shape=input_shape
                    ))
            else:
                if self.use_bidirectional:
                    model.add(tf.keras.layers.Bidirectional(
                        LSTM(
                            units=self.units,
                            activation=self.activation,
                            recurrent_activation=self.recurrent_activation,
                            dropout=self.dropout_rate,
                            recurrent_dropout=self.recurrent_dropout,
                            return_sequences=return_sequences
                        )
                    ))
                else:
                    model.add(LSTM(
                        units=self.units,
                        activation=self.activation,
                        recurrent_activation=self.recurrent_activation,
                        dropout=self.dropout_rate,
                        recurrent_dropout=self.recurrent_dropout,
                        return_sequences=return_sequences
                    ))
            
            # Add dropout after each LSTM layer except the last one
            if i < self.layers - 1:
                model.add(Dropout(self.dropout_rate))
        
        # Add final dropout for regularization
        model.add(Dropout(self.dropout_rate))
        
        # Add output layer based on task type
        if self.task_type == "classification":
            model.add(Dense(num_classes, activation='softmax'))
            metrics = ['accuracy']
        else:  # regression
            model.add(Dense(1))  # Single output for regression
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
        Train the LSTM model on the provided data.
        
        Args:
            X_train: Training features (should be shaped for LSTM: [samples, timesteps, features])
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
            # For regression, usually we normalize targets for better convergence
            if len(y_train.shape) == 1:
                y_train = y_train.reshape(-1, 1)
            # Store the target scaler for later inverse transformation
            self.target_scaler = MinMaxScaler(feature_range=(0, 1))
            y_train_encoded = self.target_scaler.fit_transform(y_train)
            num_classes = None
        
        # Build the model if not already built
        if self.model is None:
            self.model = self._build_model(self.input_shape, num_classes)
        
        # Set up early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,  # LSTM can take longer to converge
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
            X: Features to make predictions on (should be shaped for LSTM)
            
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
        else:
            # Inverse transform for regression predictions
            return self.target_scaler.inverse_transform(predictions)
    
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
    
    def forecast_future(self, X_seed, num_steps, return_sequences=False):
        """
        Generate multi-step forecasts using the trained model.
        
        Args:
            X_seed: Initial seed data for forecasting (single sample)
            num_steps: Number of steps to forecast into the future
            return_sequences: Whether to return all prediction steps or just the final prediction
            
        Returns:
            Forecasted values
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        if self.task_type != "regression":
            raise ValueError("Multi-step forecasting is only available for regression tasks")
        
        # Ensure X_seed is properly shaped
        if len(X_seed.shape) == 2:
            # If 2D, assume it's already a sequence and add batch dimension
            X_current = X_seed.reshape(1, X_seed.shape[0], X_seed.shape[1])
        else:
            # If not 2D, ensure it's a single sample with correct shape
            X_current = X_seed.reshape(1, -1, X_seed.shape[-1])
        
        # Scale input data
        original_shape = X_current.shape
        X_reshaped = X_current.reshape(X_current.shape[0], -1)
        X_scaled = self.scaler.transform(X_reshaped)
        X_current = X_scaled.reshape(original_shape)
        
        # Initialize storage for predictions
        predictions = []
        
        # Generate future predictions
        for _ in range(num_steps):
            # Predict next step
            next_step = self.model.predict(X_current)
            
            # For multivariate forecasting, we need to reshape the prediction
            # and append it to the input sequence for the next prediction
            if len(next_step.shape) == 2 and next_step.shape[1] == 1:
                next_step_reshaped = next_step.reshape(1, 1, 1)
            else:
                next_step_reshaped = next_step.reshape(1, 1, -1)
            
            # Store the prediction (inverse transformed)
            predictions.append(self.target_scaler.inverse_transform(next_step)[0, 0])
            
            # Update input sequence for next prediction by shifting window
            if X_current.shape[1] > 1:
                # For multi-step input, shift window and add new prediction
                X_current = np.concatenate([
                    X_current[:, 1:, :],  # Remove first timestep
                    next_step_reshaped    # Add new prediction as last timestep
                ], axis=1)
            else:
                # For single-step input, just replace with new prediction
                X_current = next_step_reshaped
        
        if return_sequences:
            return np.array(predictions)
        else:
            return predictions[-1]
    
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
            # Reshape if needed
            if len(y_test.shape) == 1:
                y_test = y_test.reshape(-1, 1)
            if len(y_pred.shape) == 1:
                y_pred = y_pred.reshape(-1, 1)
                
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_test - y_pred))
            
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae
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
    
    def plot_time_series_prediction(self, X_test, y_test, n_samples=None, start_idx=0):
        """
        Plot the comparison between actual and predicted values for time series data.
        
        Args:
            X_test: Test features
            y_test: Test target values
            n_samples: Number of samples to display (if None, show all)
            start_idx: Starting index for plotting
            
        Returns:
            Matplotlib figure
        """
        if self.task_type != "regression":
            raise ValueError("Time series prediction plot is only for regression tasks")
        
        y_pred = self.predict(X_test)
        
        # Reshape if needed
        if len(y_test.shape) == 1:
            y_test = y_test.reshape(-1, 1)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        # Set number of samples to display
        if n_samples is None:
            n_samples = len(y_test) - start_idx
        else:
            n_samples = min(n_samples, len(y_test) - start_idx)
        
        end_idx = start_idx + n_samples
        
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # Create x-axis values (time indices)
        time_idx = np.arange(start_idx, end_idx)
        
        # Plot actual vs predicted values
        ax.plot(time_idx, y_test[start_idx:end_idx], 'b-', label='Actual')
        ax.plot(time_idx, y_pred[start_idx:end_idx], 'r--', label='Predicted')
        
        ax.set_title('Time Series Prediction')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_forecast(self, X_seed, actual_future=None, num_steps=10):
        """
        Plot multi-step forecast with optional comparison to actual values.
        
        Args:
            X_seed: Initial seed data for forecasting
            actual_future: Optional ground truth future values to compare against
            num_steps: Number of steps to forecast
            
        Returns:
            Matplotlib figure
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Generate forecast
        forecast = self.forecast_future(X_seed, num_steps, return_sequences=True)
        
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # Create x-axis time indices
        time_idx = np.arange(num_steps)
        
        # Plot forecast
        ax.plot(time_idx, forecast, 'r--', marker='o', label='Forecast')
        
        # Plot actual future values if provided
        if actual_future is not None:
            if len(actual_future) < num_steps:
                actual_time_idx = np.arange(len(actual_future))
                ax.plot(actual_time_idx, actual_future, 'b-', marker='x', label='Actual')
            else:
                ax.plot(time_idx, actual_future[:num_steps], 'b-', marker='x', label='Actual')
        
        ax.set_title('Multi-step Forecast')
        ax.set_xlabel('Time Steps Ahead')
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
        
        # Save target scaler for regression models
        if self.task_type == "regression" and hasattr(self, 'target_scaler'):
            model_data['target_scaler'] = self.target_scaler
        
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
        
        # Load target scaler for regression models
        if 'target_scaler' in model_data:
            self.target_scaler = model_data['target_scaler']
        
        # Set attributes from model_params
        for key, value in self.model_params.items():
            setattr(self, key, value)
        
        # Load Keras model
        keras_model_path = filepath + '.keras'
        self.model = tf.keras.models.load_model(keras_model_path)
        
        return self
        
    def plot_model_architecture(self):
        """
        Visualize the LSTM model architecture.
        
        Returns:
            None (displays model summary)
        """
        if self.model is None:
            raise ValueError("Model has not been built yet.")
        
        # Print model summary
        self.model.summary()
        
        # Plot model architecture if graphviz and pydot are available
        try:
            from tensorflow.keras.utils import plot_model
            import tempfile
            import os
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                temp_filename = tmp.name
                
            # Save model plot to temporary file
            plot_model(self.model, to_file=temp_filename, show_shapes=True, show_layer_names=True)
            
            # Display the image
            from IPython.display import Image, display
            display(Image(temp_filename))
            
            # Clean up temporary file
            os.unlink(temp_filename)
            
        except ImportError:
            print("To visualize the model architecture, install graphviz and pydot packages.")
            print("You can still view the model summary above.") 