"""
Linear Regression Example

This script demonstrates how to use the LinearRegressionModel class
to predict house prices based on features.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add the parent directory to the path to import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.data_utils import load_data, split_data, preprocess_features
from models.supervised.linear_regression import LinearRegressionModel


def run_synthetic_data_example():
    """Run a linear regression example with synthetic data."""
    print("\n" + "="*50)
    print("LINEAR REGRESSION WITH SYNTHETIC HOUSING DATA")
    print("="*50)
    
    # Generate synthetic data for house price prediction
    np.random.seed(42)
    n_samples = 1000
    
    # Features: size, bedrooms, bathrooms, age, distance_to_city
    X = np.zeros((n_samples, 5))
    
    # Size in sq.ft (1000-3000)
    X[:, 0] = np.random.uniform(1000, 3000, n_samples)
    
    # Bedrooms (1-5)
    X[:, 1] = np.random.randint(1, 6, n_samples)
    
    # Bathrooms (1-3)
    X[:, 2] = np.random.randint(1, 4, n_samples)
    
    # Age of the house (0-50 years)
    X[:, 3] = np.random.uniform(0, 50, n_samples)
    
    # Distance to city center (0-30 miles)
    X[:, 4] = np.random.uniform(0, 30, n_samples)
    
    # Generate target with some noise
    # Price = base + size_factor + bedroom_factor + bathroom_factor - age_penalty - distance_penalty + noise
    y = (
        150000 +                        # Base price
        100 * X[:, 0] +                 # $100 per sq.ft
        15000 * X[:, 1] +               # $15,000 per bedroom
        25000 * X[:, 2] -               # $25,000 per bathroom
        1000 * X[:, 3] -                # $1,000 penalty per year of age
        5000 * X[:, 4] +                # $5,000 penalty per mile from city
        np.random.normal(0, 25000, n_samples)  # Random noise with std $25,000
    )
    
    # Feature names
    feature_names = ['size_sqft', 'bedrooms', 'bathrooms', 'age_years', 'distance_to_city']
    
    # Create a DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['price'] = y
    
    print("Synthetic housing dataset created:")
    print(df.head())
    print("\nDataset statistics:")
    print(df.describe().round(2))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the model
    print("\nTraining Linear Regression model...")
    model = LinearRegressionModel()
    model.train(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate
    metrics = model.evaluate(X_test_scaled, y_test, task_type="regression")
    print("\nPerformance metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
    
    # Feature importance
    print("\nFeature Importance (Coefficients):")
    importance_df = model.get_feature_importance(feature_names)
    print(importance_df)
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual House Price ($)')
    plt.ylabel('Predicted House Price ($)')
    plt.title('Linear Regression: Actual vs Predicted House Prices')
    plt.tight_layout()
    plt.savefig('examples/linear_regression_predictions.png')
    print("\nSaved plot to 'examples/linear_regression_predictions.png'")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Coefficient'])
    plt.xlabel('Scaled Coefficient')
    plt.ylabel('Feature')
    plt.title('Linear Regression: Feature Importance')
    plt.tight_layout()
    plt.savefig('examples/linear_regression_feature_importance.png')
    print("Saved plot to 'examples/linear_regression_feature_importance.png'")
    
    # Plot residuals
    y_pred = model.predict(X_test_scaled)
    residuals = y_test - y_pred
    
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
    plt.savefig('examples/linear_regression_residuals.png')
    print("Saved plot to 'examples/linear_regression_residuals.png'")
    
    return model


def run_california_housing_example():
    """Run a linear regression example with California housing dataset."""
    print("\n" + "="*50)
    print("LINEAR REGRESSION WITH CALIFORNIA HOUSING DATASET")
    print("="*50)
    
    # Load the California housing dataset
    california = fetch_california_housing()
    X = california.data
    y = california.target
    
    # Feature names
    feature_names = california.feature_names
    
    print(f"Dataset loaded: {X.shape[0]} samples with {X.shape[1]} features")
    print("Features:", feature_names)
    print("Target: Median house value (in $100,000s)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the model
    print("\nTraining Linear Regression model...")
    model = LinearRegressionModel()
    model.train(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate
    metrics = model.evaluate(X_test_scaled, y_test, task_type="regression")
    print("\nPerformance metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Feature importance
    print("\nFeature Importance (Coefficients):")
    importance_df = model.get_feature_importance(feature_names)
    print(importance_df)
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual Median House Value ($100k)')
    plt.ylabel('Predicted Median House Value ($100k)')
    plt.title('Linear Regression: California Housing Prices')
    plt.tight_layout()
    plt.savefig('examples/california_housing_predictions.png')
    print("\nSaved plot to 'examples/california_housing_predictions.png'")
    
    return model


if __name__ == "__main__":
    # Run both examples
    synthetic_model = run_synthetic_data_example()
    california_model = run_california_housing_example()
    
    print("\n" + "="*50)
    print("Examples completed! Check the generated plots.")
    print("="*50) 