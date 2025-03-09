"""
RNN Time Series Prediction Example

This example demonstrates how to use the RNN model for time series prediction.
We'll create a synthetic time series dataset with seasonal patterns and train
the RNN to predict future values.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.deep_learning import RNNModel

# Set random seed for reproducibility
np.random.seed(42)

def main():
    # Generate synthetic time series data
    print("Generating synthetic time series data...")
    
    # Time steps
    time = np.arange(0, 1000)
    
    # Create a seasonal pattern with noise
    seasonal = 0.5 * np.sin(time * 0.1) + 0.2 * np.cos(time * 0.01)
    trend = 0.001 * time
    noise = 0.1 * np.random.randn(len(time))
    
    # Combine components
    data = seasonal + trend + noise
    
    # Plot the time series
    plt.figure(figsize=(15, 5))
    plt.plot(time, data)
    plt.title('Synthetic Time Series Data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()
    
    # Prepare data for RNN (create sequences)
    X, y = create_sequences(data, seq_length=10)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Create and train the RNN model
    print("\nTraining RNN model...")
    model = RNNModel(
        task_type="regression",
        units=64,
        activation="tanh",
        dropout_rate=0.2,
        optimizer="adam",
        batch_size=32,
        epochs=50,
        validation_split=0.1
    )
    
    model.train(X_train, y_train)
    
    # Evaluate the model
    print("\nEvaluating model performance...")
    eval_metrics = model.evaluate(X_test, y_test)
    print(f"Mean Squared Error: {eval_metrics['mse']:.4f}")
    print(f"Root Mean Squared Error: {eval_metrics['rmse']:.4f}")
    
    # Plot training history
    history_plot = model.plot_training_history()
    plt.show()
    
    # Plot predictions vs actual values
    y_pred = model.predict(X_test)
    
    plt.figure(figsize=(15, 6))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title('RNN Time Series Prediction')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Demonstrate multi-step forecasting
    print("\nGenerating multi-step forecast...")
    
    # Use the last sequence from test data as seed
    seed_sequence = X_test[-1:].copy()
    
    # Generate future steps
    forecast_horizon = 50
    forecasted_values = []
    
    for _ in range(forecast_horizon):
        # Predict the next value
        next_value = model.predict(seed_sequence)[0]
        forecasted_values.append(next_value)
        
        # Update seed sequence by shifting and adding the prediction
        seed_sequence = np.roll(seed_sequence, -1, axis=1)
        seed_sequence[0, -1, 0] = next_value
    
    # Get the actual values from the test set for comparison
    actual_future = y_test[:forecast_horizon]
    
    # Plot forecast vs actual values
    plt.figure(figsize=(15, 6))
    plt.plot(actual_future, label='Actual')
    plt.plot(forecasted_values, label='Forecast')
    plt.title('RNN Multi-step Forecast')
    plt.xlabel('Time Steps Ahead')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Save the trained model
    print("\nSaving model...")
    model.save_model("../data/rnn_time_series_model")
    
    print("\nDone!")
    
def create_sequences(data, seq_length=10):
    """
    Create input sequences and target values for time series prediction.
    
    Args:
        data: Input time series data
        seq_length: Length of input sequences
        
    Returns:
        X: Input sequences, shape (samples, seq_length, features)
        y: Target values
    """
    X = []
    y = []
    
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X to match the input shape for RNN: [samples, timesteps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, y

def demonstrate_sentiment_analysis():
    """
    Demonstrate RNN for sentiment analysis on a small toy dataset.
    This is a simplified example to show how RNN can be used for text classification.
    """
    print("\n--- Bonus: Sentiment Analysis Demo ---")
    
    # Create a toy dataset (simplified for demonstration)
    texts = [
        "I love this product, it's amazing",
        "Great experience, would recommend",
        "Very happy with my purchase",
        "Best service ever",
        "This is wonderful",
        "Terrible experience, would not recommend",
        "Very disappointed with the quality",
        "Worst product I've ever bought",
        "Complete waste of money",
        "Do not buy this product"
    ]
    
    # Labels: 1 for positive, 0 for negative
    labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    
    # Simple tokenization and sequence generation
    # In a real application, you would use a proper tokenizer and word embeddings
    word_index = {}
    current_idx = 1  # Reserve 0 for padding
    
    # Create sequences with fixed length
    max_length = 10
    sequences = np.zeros((len(texts), max_length))
    
    for i, text in enumerate(texts):
        words = text.lower().split()
        for j, word in enumerate(words[:max_length]):
            if word not in word_index:
                word_index[word] = current_idx
                current_idx += 1
            sequences[i, j] = word_index[word]
    
    # Reshape for RNN: [samples, timesteps, features]
    X = sequences.reshape(sequences.shape[0], sequences.shape[1], 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    
    # Create and train the RNN model
    model = RNNModel(
        task_type="classification",
        units=16,
        activation="tanh",
        dropout_rate=0.5,
        optimizer="adam",
        batch_size=2,
        epochs=20,
        validation_split=0.1
    )
    
    model.train(X_train, y_train)
    
    # Evaluate
    eval_metrics = model.evaluate(X_test, y_test)
    print(f"Sentiment Classification Accuracy: {eval_metrics['accuracy']:.4f}")
    
    # Plot confusion matrix
    if len(y_test) > 1:  # Check if we have enough test samples
        cm_plot = model.plot_confusion_matrix(X_test, y_test, class_names=["Negative", "Positive"])
        plt.show()

if __name__ == "__main__":
    main()
    
    # Uncomment to run the sentiment analysis demo
    # demonstrate_sentiment_analysis() 