"""
Support Vector Machine Example: Handwritten Digit Recognition

This example demonstrates how to use the Support Vector Machine model for 
handwritten digit recognition using the MNIST dataset. The example:

1. Loads and preprocesses the MNIST handwritten digits dataset
2. Trains an SVM model with different kernels (linear and RBF)
3. Evaluates the model performance on test data
4. Visualizes the decision boundary and support vectors
5. Demonstrates how SVMs handle high-dimensional data effectively
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import time

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from models.supervised.svm import SVMModel


def load_mnist_data(use_small_dataset=True):
    """
    Load the MNIST dataset for handwritten digit recognition.
    
    Args:
        use_small_dataset: If True, uses the smaller sklearn digits dataset,
                          otherwise downloads the full MNIST dataset from OpenML
    
    Returns:
        X: Features (digit images)
        y: Labels (digit values 0-9)
        X_images: Reshaped images for visualization
    """
    if use_small_dataset:
        # Load smaller digits dataset from sklearn
        digits = load_digits()
        X = digits.data
        y = digits.target
        # Get original image dimensions for visualization
        X_images = digits.images
        print(f"Loaded smaller digits dataset with {len(X)} examples")
        print(f"Image dimensions: {X_images[0].shape}")
    else:
        try:
            # Try to load the full MNIST dataset
            # This will download the dataset if it's not available locally
            print("Loading MNIST dataset (this may take a moment)...")
            mnist = fetch_openml('mnist_784', version=1, cache=True)
            X = mnist.data
            y = mnist.target.astype(int)
            # Reshape for visualization
            X_images = X.values.reshape(-1, 28, 28)
            print(f"Loaded full MNIST dataset with {len(X)} examples")
            
            # Use only a subset of the data to speed up training
            subset_size = 10000  # Adjust based on your computational resources
            indices = np.random.choice(len(X), subset_size, replace=False)
            X = X.iloc[indices]
            y = y[indices]
            X_images = X_images[indices]
        except Exception as e:
            print(f"Error loading full MNIST dataset: {e}")
            print("Falling back to smaller digits dataset")
            digits = load_digits()
            X = digits.data
            y = digits.target
            X_images = digits.images
    
    return X, y, X_images


def visualize_digits(X_images, y, num_images=10):
    """
    Visualize sample digits from the dataset.
    
    Args:
        X_images: Digit images
        y: Labels
        num_images: Number of images to display
    """
    plt.figure(figsize=(15, 3))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(X_images[i], cmap='gray')
        plt.title(f"Digit: {y[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def train_and_evaluate_svm(X_train, y_train, X_test, y_test, kernel='rbf', C=1.0, gamma='scale'):
    """
    Train and evaluate an SVM model with the specified parameters.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
        C: Regularization parameter
        gamma: Kernel coefficient
        
    Returns:
        trained_model: The trained SVM model
        accuracy: Accuracy on the test set
        training_time: Time taken to train the model
    """
    print(f"\nTraining SVM with {kernel.upper()} kernel (C={C}, gamma={gamma})...")
    
    # Create and train the model
    svm_model = SVMModel(kernel=kernel, C=C, gamma=gamma, probability=True)
    
    # Time the training process
    start_time = time.time()
    svm_model.train(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate the model
    metrics = svm_model.evaluate(X_test, y_test)
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Get number of support vectors
    n_support = svm_model.get_n_support()
    total_support_vectors = sum(n_support.values())
    print(f"\nTotal support vectors: {total_support_vectors} out of {len(X_train)} training examples")
    print("Support vectors per class:")
    for class_label, count in n_support.items():
        print(f"  Class {class_label}: {count} vectors")
    
    return svm_model, metrics['accuracy'], training_time


def main():
    print("=" * 80)
    print("Support Vector Machine Example: Handwritten Digit Recognition")
    print("=" * 80)
    
    # 1. Load and preprocess the data
    print("\n1. Loading MNIST handwritten digits dataset...")
    X, y, X_images = load_mnist_data(use_small_dataset=True)
    
    # Visualize some examples
    print("\n2. Visualizing sample digits from the dataset...")
    visualize_digits(X_images, y)
    
    # Split the data into training and testing sets
    print("\n3. Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # Standardize the features
    print("\n4. Preprocessing the data (standardization)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Train and evaluate SVM models with different kernels
    print("\n5. Training and evaluating SVM models with different kernels...")
    
    # Train linear kernel SVM
    linear_svm, linear_accuracy, linear_time = train_and_evaluate_svm(
        X_train_scaled, y_train, X_test_scaled, y_test, 
        kernel='linear', C=1.0
    )
    
    # Train RBF kernel SVM
    rbf_svm, rbf_accuracy, rbf_time = train_and_evaluate_svm(
        X_train_scaled, y_train, X_test_scaled, y_test, 
        kernel='rbf', C=10.0, gamma=0.01
    )
    
    # Compare the models
    print("\n6. Comparing SVM models with different kernels:")
    comparison_data = {
        'Model': ['Linear SVM', 'RBF SVM'],
        'Accuracy': [linear_accuracy, rbf_accuracy],
        'Training Time (s)': [linear_time, rbf_time]
    }
    
    for i in range(len(comparison_data['Model'])):
        print(f"{comparison_data['Model'][i]}:")
        print(f"  Accuracy: {comparison_data['Accuracy'][i]:.4f}")
        print(f"  Training Time: {comparison_data['Training Time (s)'][i]:.2f} seconds")
    
    # Plot confusion matrix for the best model
    best_model = rbf_svm if rbf_accuracy > linear_accuracy else linear_svm
    best_kernel = 'RBF' if rbf_accuracy > linear_accuracy else 'Linear'
    
    print(f"\n7. Visualizing confusion matrix for the best model ({best_kernel} SVM)...")
    y_pred = best_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {best_kernel} SVM')
    plt.show()
    
    # 8. Visualize decision boundaries using dimensionality reduction
    print("\n8. Visualizing decision boundaries in 2D (using PCA)...")
    
    # Apply PCA to reduce dimensions to 2D for visualization
    pca = PCA(n_components=2)
    X_train_2d = pca.fit_transform(X_train_scaled)
    X_test_2d = pca.transform(X_test_scaled)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")
    
    # Train a new SVM on the 2D data for visualization
    svm_2d = SVMModel(kernel='rbf', C=10.0, gamma=0.1)
    svm_2d.train(X_train_2d, y_train)
    
    # Plot decision boundaries
    plt.figure(figsize=(12, 10))
    
    # Create a mesh grid
    x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
    y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Make predictions on the mesh grid
    Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.4)
    
    # Plot the training points
    scatter = plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, 
                         alpha=0.8, edgecolors='k')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('SVM Decision Boundaries (PCA 2D projection)')
    plt.colorbar(scatter)
    plt.show()
    
    # 9. Visualize support vectors in 2D
    print("\n9. Visualizing support vectors in 2D...")
    svm_2d.plot_support_vectors(X_train_2d, y_train)
    
    # 10. Make predictions on new examples
    print("\n10. Making predictions on test examples...")
    
    # Select some test examples
    n_examples = 5
    example_indices = np.random.choice(len(X_test), n_examples, replace=False)
    example_images = X_test[example_indices]
    example_labels = y_test[example_indices]
    example_scaled = X_test_scaled[example_indices]
    
    # Get predictions from the best model
    predictions = best_model.predict(example_scaled)
    probabilities = best_model.predict_proba(example_scaled)
    
    # Visualize the examples and predictions
    plt.figure(figsize=(15, 6))
    for i in range(n_examples):
        plt.subplot(2, n_examples, i + 1)
        if isinstance(example_images[i], np.ndarray) and example_images[i].ndim == 2:
            plt.imshow(example_images[i], cmap='gray')
        else:
            # Reshape the flattened image
            img_dim = int(np.sqrt(example_images[i].shape[0]))
            plt.imshow(example_images[i].reshape(img_dim, img_dim), cmap='gray')
        plt.title(f"True: {example_labels[i]}")
        plt.axis('off')
        
        # Plot the prediction probabilities
        plt.subplot(2, n_examples, i + 1 + n_examples)
        top_probs = probabilities[i].argsort()[-3:][::-1]
        bars = plt.bar(range(3), [probabilities[i][j] for j in top_probs])
        plt.xticks(range(3), [str(j) for j in top_probs])
        plt.xlabel('Digit')
        plt.ylabel('Probability')
        plt.title(f"Pred: {predictions[i]}")
    
    plt.tight_layout()
    plt.show()
    
    # 11. Save the best model
    print("\n11. Saving the best model...")
    model_path = f"../models/svm_{best_kernel.lower()}_digit_recognition.joblib"
    best_model.save_model(model_path)
    
    print("\nExample complete! The SVM model has been trained for handwritten digit recognition.")


if __name__ == "__main__":
    main() 