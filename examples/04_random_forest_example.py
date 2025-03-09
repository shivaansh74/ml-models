"""
Random Forest Example

This script demonstrates how to use the Random Forest model for both classification and regression tasks.
It includes data loading, preprocessing, model training, evaluation, and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from models.ensemble.random_forest import RandomForestModel
from utils.data_utils import load_dataset, plot_learning_curve

# Set random seed for reproducibility
np.random.seed(42)

def random_forest_classification_example():
    """
    Example of using Random Forests for a classification task using the Breast Cancer dataset.
    """
    print("\n" + "="*80)
    print(" Random Forest - Classification Example (Breast Cancer Dataset) ".center(80, "="))
    print("="*80)
    
    # Load Breast Cancer dataset
    print("\n[1] Loading and preparing the Breast Cancer dataset...")
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    feature_names = cancer.feature_names
    target_names = cancer.target_names
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {len(feature_names)} features including radius, texture, perimeter, area, etc.")
    print(f"Target classes: {target_names} (malignant or benign)")
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Create and train the model
    print("\n[2] Creating and training the Random Forest model...")
    rf_classifier = RandomForestModel(
        task="classification",
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        oob_score=True  # Enable out-of-bag estimation
    )
    
    rf_classifier.train(X_train_scaled, y_train)
    
    # Evaluate the model
    print("\n[3] Evaluating the model...")
    y_pred = rf_classifier.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    metrics = rf_classifier.evaluate(X_test_scaled, y_test)
    print(f"Evaluation metrics: {metrics}")
    
    # Visualize feature importance
    print("\n[4] Visualizing feature importance...")
    feature_names_short = [name[:20] + '...' if len(name) > 20 else name for name in feature_names]
    importance_plot = rf_classifier.plot_feature_importance(feature_names_short, top_n=10)
    plt.tight_layout()
    plt.show()
    
    # Visualize a sample tree
    print("\n[5] Visualizing a sample tree from the forest...")
    tree_plot = rf_classifier.plot_tree(tree_index=0, feature_names=feature_names_short, class_names=target_names)
    plt.tight_layout()
    plt.show()
    
    # Plot OOB error 
    print("\n[6] Plotting OOB error...")
    try:
        oob_plot = rf_classifier.plot_oob_error(X_train_scaled, y_train, max_estimators=100, step=10)
        plt.tight_layout()
        plt.show()
    except ValueError as e:
        print(f"Could not plot OOB error: {e}")
    
    # Plot learning curve
    print("\n[7] Plotting learning curve...")
    plot_learning_curve(
        rf_classifier.model, 
        "Random Forest - Breast Cancer Classification", 
        X, 
        y, 
        cv=5, 
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    plt.show()
    
    print("\n[8] Making predictions on new data...")
    # Example of making predictions on new data (using a subset of the test data)
    new_samples = X_test_scaled[:3]
    
    predictions = rf_classifier.predict(new_samples)
    probabilities = rf_classifier.predict_proba(new_samples)
    
    print("Predictions for new samples:")
    for i, (prediction, proba) in enumerate(zip(predictions, probabilities)):
        print(f"Sample {i+1}: {target_names[prediction]} (Confidence: {max(proba):.4f})")
        print(f"  Probabilities: {dict(zip(target_names, proba))}")
    
    print("\nClassification example complete!")


def random_forest_regression_example():
    """
    Example of using Random Forests for a regression task using the California Housing dataset.
    """
    print("\n" + "="*80)
    print(" Random Forest - Regression Example (California Housing Dataset) ".center(80, "="))
    print("="*80)
    
    # Load California Housing dataset
    print("\n[1] Loading and preparing the California Housing dataset...")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    feature_names = housing.feature_names
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Target: Median house value (in 100,000s)")
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Create and train the model
    print("\n[2] Creating and training the Random Forest model for regression...")
    rf_regressor = RandomForestModel(
        task="regression",
        n_estimators=100,
        max_depth=None,
        min_samples_split=5,
        random_state=42,
        oob_score=True  # Enable out-of-bag estimation
    )
    
    rf_regressor.train(X_train, y_train)
    
    # Evaluate the model
    print("\n[3] Evaluating the model...")
    y_pred = rf_regressor.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    metrics = rf_regressor.evaluate(X_test, y_test, task_type="regression")
    print(f"Evaluation metrics: {metrics}")
    
    # Visualize feature importance
    print("\n[4] Visualizing feature importance...")
    importance_plot = rf_regressor.plot_feature_importance(feature_names)
    plt.tight_layout()
    plt.show()
    
    # Visualize a sample tree
    print("\n[5] Visualizing a sample tree from the forest (limited to depth 3 for clarity)...")
    tree_plot = rf_regressor.plot_tree(tree_index=0, feature_names=feature_names, max_depth=3)
    plt.tight_layout()
    plt.show()
    
    # Plot OOB error 
    print("\n[6] Plotting OOB error...")
    try:
        oob_plot = rf_regressor.plot_oob_error(X_train, y_train, max_estimators=100, step=10)
        plt.tight_layout()
        plt.show()
    except ValueError as e:
        print(f"Could not plot OOB error: {e}")
    
    # Visualize predictions vs actual
    print("\n[7] Visualizing predictions vs actual values...")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Random Forest Regression: Actual vs Predicted Values')
    plt.tight_layout()
    plt.show()
    
    print("\n[8] Making predictions on new data...")
    # Example of making predictions on new data (using a subset of the test data)
    new_samples = X_test[:3]
    
    predictions = rf_regressor.predict(new_samples)
    
    print("Predictions for new samples (housing prices in $100,000s):")
    for i, prediction in enumerate(predictions):
        print(f"Sample {i+1}: ${prediction:.2f} (100,000s)")
    
    print("\nRegression example complete!")


if __name__ == "__main__":
    random_forest_classification_example()
    random_forest_regression_example() 