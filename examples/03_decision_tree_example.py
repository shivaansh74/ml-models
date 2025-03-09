"""
Decision Tree Example

This script demonstrates how to use the Decision Tree model for both classification and regression tasks.
It includes data loading, preprocessing, model training, evaluation, and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from models.supervised.decision_tree import DecisionTreeModel
from utils.data_utils import load_dataset, plot_learning_curve

# Set random seed for reproducibility
np.random.seed(42)

def decision_tree_classification_example():
    """
    Example of using Decision Trees for a classification task using the Iris dataset.
    """
    print("\n" + "="*80)
    print(" Decision Tree - Classification Example (Iris Dataset) ".center(80, "="))
    print("="*80)
    
    # Load Iris dataset
    print("\n[1] Loading and preparing the Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Target classes: {target_names}")
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Create and train the model
    print("\n[2] Creating and training the Decision Tree model...")
    dt_classifier = DecisionTreeModel(
        task="classification",
        max_depth=3,  # Limiting depth to avoid overfitting
        random_state=42
    )
    
    dt_classifier.train(X_train, y_train)
    
    # Evaluate the model
    print("\n[3] Evaluating the model...")
    y_pred = dt_classifier.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    metrics = dt_classifier.evaluate(X_test, y_test)
    print(f"Evaluation metrics: {metrics}")
    
    # Visualize feature importance
    print("\n[4] Visualizing feature importance...")
    importance_plot = dt_classifier.plot_feature_importance(feature_names)
    plt.tight_layout()
    plt.show()
    
    # Visualize the decision tree
    print("\n[5] Visualizing the decision tree...")
    tree_plot = dt_classifier.plot_tree(feature_names=feature_names, class_names=target_names)
    plt.tight_layout()
    plt.show()
    
    # Plot learning curve
    print("\n[6] Plotting learning curve...")
    plot_learning_curve(
        dt_classifier.model, 
        "Decision Tree - Iris Classification", 
        X, 
        y, 
        cv=5, 
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    plt.show()
    
    print("\n[7] Making predictions on new data...")
    # Example of making predictions on new data
    new_samples = np.array([
        [5.1, 3.5, 1.4, 0.2],  # Example of Setosa
        [6.7, 3.1, 4.7, 1.5],  # Example of Versicolor
        [6.3, 3.3, 6.0, 2.5],  # Example of Virginica
    ])
    
    predictions = dt_classifier.predict(new_samples)
    probabilities = dt_classifier.predict_proba(new_samples)
    
    print("Predictions for new samples:")
    for i, (prediction, proba) in enumerate(zip(predictions, probabilities)):
        print(f"Sample {i+1}: {target_names[prediction]} (Confidence: {max(proba):.4f})")
        print(f"  Probabilities: {dict(zip(target_names, proba))}")
    
    print("\nClassification example complete!")


def decision_tree_regression_example():
    """
    Example of using Decision Trees for a regression task using the California Housing dataset.
    """
    print("\n" + "="*80)
    print(" Decision Tree - Regression Example (California Housing Dataset) ".center(80, "="))
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
    print("\n[2] Creating and training the Decision Tree model for regression...")
    dt_regressor = DecisionTreeModel(
        task="regression",
        max_depth=5,  # Limiting depth to avoid overfitting
        min_samples_split=5,
        random_state=42
    )
    
    dt_regressor.train(X_train, y_train)
    
    # Evaluate the model
    print("\n[3] Evaluating the model...")
    y_pred = dt_regressor.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    metrics = dt_regressor.evaluate(X_test, y_test, task_type="regression")
    print(f"Evaluation metrics: {metrics}")
    
    # Visualize feature importance
    print("\n[4] Visualizing feature importance...")
    importance_plot = dt_regressor.plot_feature_importance(feature_names)
    plt.tight_layout()
    plt.show()
    
    # Visualize the decision tree
    print("\n[5] Visualizing the decision tree (limited to depth 3 for clarity)...")
    tree_plot = dt_regressor.plot_tree(feature_names=feature_names, max_depth=3)
    plt.tight_layout()
    plt.show()
    
    # Visualize predictions vs actual
    print("\n[6] Visualizing predictions vs actual values...")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Decision Tree Regression: Actual vs Predicted Values')
    plt.tight_layout()
    plt.show()
    
    print("\n[7] Making predictions on new data...")
    # Example of making predictions on new data (using a subset of the test data)
    new_samples = X_test[:3]
    
    predictions = dt_regressor.predict(new_samples)
    
    print("Predictions for new samples (housing prices in $100,000s):")
    for i, prediction in enumerate(predictions):
        print(f"Sample {i+1}: ${prediction:.2f} (100,000s)")
    
    print("\nRegression example complete!")


if __name__ == "__main__":
    decision_tree_classification_example()
    decision_tree_regression_example() 