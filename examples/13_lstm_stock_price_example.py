"""
LSTM Stock Price Prediction Example

This example demonstrates how to use the LSTM model for stock price prediction.
We'll create a synthetic stock price dataset with trend, seasonality, and volatility
patterns, and train an LSTM model to forecast future prices.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.deep_learning import LSTMModel

# Set random seed for reproducibility
np.random.seed(42)

def main():
    # Generate synthetic stock price data
    print("Generating synthetic stock price data...")
    
    # Number of trading days (approximately 2 years)
    n_days = 500
    
    # Create date range
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Generate synthetic stock prices
    prices = generate_synthetic_stock_prices(n_days, initial_price=100.0)
    
    # Create DataFrame
    stock_data = pd.DataFrame({
        'Date': dates,
        'Close': prices,
    })
    
    # Add some technical indicators
    stock_data['MA5'] = stock_data['Close'].rolling(window=5).mean()
    stock_data['MA20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
    
    # Calculate daily returns
    stock_data['Returns'] = stock_data['Close'].pct_change()
    
    # Calculate volatility (standard deviation of returns over 20 days)
    stock_data['Volatility'] = stock_data['Returns'].rolling(window=20).std()
    
    # Drop NaN values
    stock_data = stock_data.dropna()
    
    # Plot the stock price data
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(stock_data['Date'], stock_data['Close'], label='Price')
    plt.plot(stock_data['Date'], stock_data['MA20'], label='20-day MA', alpha=0.7)
    plt.plot(stock_data['Date'], stock_data['MA50'], label='50-day MA', alpha=0.7)
    plt.title('Synthetic Stock Price Data')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(stock_data['Date'], stock_data['Returns'], label='Daily Returns')
    plt.plot(stock_data['Date'], stock_data['Volatility'], label='Volatility (20-day)', color='red')
    plt.title('Returns and Volatility')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Prepare data for LSTM
    feature_columns = ['Close', 'MA5', 'MA20', 'MA50', 'Returns', 'Volatility']
    target_column = 'Close'
    
    # Get features and target
    data = stock_data[feature_columns].values
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences
    X, y = create_sequences(scaled_data, target_idx=0, seq_length=20)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Create and train the LSTM model
    print("\nTraining LSTM model...")
    model = LSTMModel(
        task_type="regression",
        units=64,
        activation="tanh",
        recurrent_activation="sigmoid",
        dropout_rate=0.2,
        recurrent_dropout=0.2,
        optimizer="adam",
        batch_size=16,
        epochs=100,
        validation_split=0.1,
        layers=2,
        use_bidirectional=True
    )
    
    model.train(X_train, y_train)
    
    # Evaluate the model
    print("\nEvaluating model performance...")
    eval_metrics = model.evaluate(X_test, y_test)
    print(f"Mean Squared Error: {eval_metrics['mse']:.4f}")
    print(f"Root Mean Squared Error: {eval_metrics['rmse']:.4f}")
    print(f"Mean Absolute Error: {eval_metrics['mae']:.4f}")
    
    # Plot training history
    history_plot = model.plot_training_history()
    plt.show()
    
    # Get the original scale predictions and actual values
    y_pred = model.predict(X_test)
    
    # Prepare actual close prices for plotting
    actual_prices = stock_data['Close'].values[-len(y_test):]
    
    # Plot predictions vs actual values
    plt.figure(figsize=(15, 6))
    plt.plot(actual_prices, label='Actual Prices')
    plt.plot(y_pred, label='Predicted Prices')
    plt.title('LSTM Stock Price Prediction')
    plt.xlabel('Trading Days')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Use the model to make multi-step predictions
    print("\nGenerating multi-step forecast...")
    
    # Last sequence from test data as seed for forecasting
    seed_sequence = X_test[-1:].copy()
    
    # Number of days to forecast
    forecast_days = 30
    
    # Generate forecast
    forecast = model.forecast_future(seed_sequence, forecast_days, return_sequences=True)
    
    # Plot the forecast
    last_actual_days = 50
    last_actual_prices = actual_prices[-last_actual_days:]
    
    plt.figure(figsize=(15, 6))
    
    # Plot historical prices
    time_hist = np.arange(0, len(last_actual_prices))
    plt.plot(time_hist, last_actual_prices, label='Historical Prices')
    
    # Plot forecasted prices
    time_forecast = np.arange(len(last_actual_prices), len(last_actual_prices) + len(forecast))
    plt.plot(time_forecast, forecast, label='Price Forecast', color='red')
    
    # Add vertical line to separate historical and forecasted data
    plt.axvline(x=len(last_actual_prices)-1, color='k', linestyle='--')
    
    # Annotate the forecast starting point
    plt.annotate('Forecast Start', 
                xy=(len(last_actual_prices)-1, last_actual_prices[-1]),
                xytext=(len(last_actual_prices)-15, last_actual_prices[-1]*1.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    plt.title('LSTM Stock Price Forecast')
    plt.xlabel('Trading Days')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Calculate and visualize forecast accuracy decline
    print("\nVisualizing forecast accuracy decline over time...")
    visualize_forecast_accuracy_decline(model, X_test, y_test, window_size=20)
    
    # Save the trained model
    print("\nSaving model...")
    model.save_model("../data/lstm_stock_price_model")
    
    print("\nDone!")

def generate_synthetic_stock_prices(n_days, initial_price=100.0, trend=0.0002, 
                                   volatility=0.01, cycle_amplitude=10.0, cycle_period=126):
    """
    Generate synthetic stock prices with trend, cyclical patterns, and volatility.
    
    Args:
        n_days: Number of trading days
        initial_price: Starting price
        trend: Daily trend factor
        volatility: Daily volatility
        cycle_amplitude: Amplitude of cyclical pattern
        cycle_period: Period of cycles in days
        
    Returns:
        Array of stock prices
    """
    # Time array
    t = np.arange(n_days)
    
    # Trend component
    trend_component = initial_price * (1 + trend) ** t
    
    # Cyclical component (market cycles)
    cycle_component = cycle_amplitude * np.sin(2 * np.pi * t / cycle_period)
    
    # Weekly seasonality (higher volatility in certain days)
    weekly_vol = np.ones(n_days)
    for i in range(n_days):
        # Higher volatility on first and last days of week (0 and 4)
        day_of_week = i % 5
        if day_of_week == 0 or day_of_week == 4:
            weekly_vol[i] = 1.5
    
    # Random walk component (daily price movements)
    random_walk = np.random.normal(0, volatility, n_days) * weekly_vol
    cumulative_returns = np.cumsum(random_walk)
    
    # Combine components
    prices = trend_component + cycle_component + initial_price * np.exp(cumulative_returns)
    
    # Add some market shocks (sudden drops or spikes)
    n_shocks = 3
    shock_indices = np.random.choice(range(50, n_days-50), n_shocks, replace=False)
    shock_types = np.random.choice([-1, 1], n_shocks)  # -1 for drop, 1 for spike
    
    for idx, shock_type in zip(shock_indices, shock_types):
        shock_magnitude = np.random.uniform(0.05, 0.15)  # 5-15% shock
        shock_recovery_length = np.random.randint(10, 30)  # 10-30 days recovery
        
        # Apply shock
        shock_effect = prices[idx] * shock_type * shock_magnitude
        prices[idx] += shock_effect
        
        # Gradual recovery
        recovery_rate = shock_effect / shock_recovery_length
        for i in range(1, shock_recovery_length+1):
            if idx + i < n_days:
                prices[idx + i] -= recovery_rate * (shock_recovery_length - i) / shock_recovery_length
    
    return prices

def create_sequences(data, target_idx=0, seq_length=20):
    """
    Create sequences of data for LSTM input.
    
    Args:
        data: Input data matrix (samples x features)
        target_idx: Index of the target variable in the data matrix
        seq_length: Length of input sequences
        
    Returns:
        X: Input sequences, shape (samples, seq_length, features)
        y: Target values
    """
    X = []
    y = []
    
    for i in range(len(data) - seq_length):
        # Input sequence
        X.append(data[i:i+seq_length])
        # Target is the next value of the target variable
        y.append(data[i+seq_length, target_idx])
    
    return np.array(X), np.array(y)

def visualize_forecast_accuracy_decline(model, X_test, y_test, window_size=20):
    """
    Visualize how forecast accuracy declines as prediction horizon increases.
    
    Args:
        model: Trained LSTM model
        X_test: Test input sequences
        y_test: Test target values
        window_size: Window size for visualization
    """
    # Select a subset of test data for visualization
    n_samples = min(10, len(X_test) - window_size)
    start_indices = np.linspace(0, len(X_test) - window_size - 1, n_samples, dtype=int)
    
    # Store errors for different forecast horizons
    horizons = list(range(1, window_size + 1))
    mae_by_horizon = np.zeros(len(horizons))
    
    plt.figure(figsize=(15, 12))
    
    # For each starting point
    for i, start_idx in enumerate(start_indices):
        seed = X_test[start_idx:start_idx+1].copy()
        actual_values = y_test[start_idx:start_idx+window_size]
        
        # Generate multi-step forecast
        forecast = model.forecast_future(seed, window_size, return_sequences=True)
        
        # Calculate errors
        errors = np.abs(forecast - actual_values)
        mae_by_horizon += errors
        
        # Plot individual forecasts
        plt.subplot(len(start_indices), 1, i+1)
        plt.plot(horizons, actual_values, 'b-', label='Actual')
        plt.plot(horizons, forecast, 'r--', label='Forecast')
        plt.title(f'Forecast from position {start_idx}')
        if i == 0:
            plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Plot average error by forecast horizon
    mae_by_horizon /= n_samples
    
    plt.figure(figsize=(12, 6))
    plt.plot(horizons, mae_by_horizon, 'o-')
    plt.title('Forecast Error by Horizon')
    plt.xlabel('Forecast Horizon (Days Ahead)')
    plt.ylabel('Mean Absolute Error')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main() 