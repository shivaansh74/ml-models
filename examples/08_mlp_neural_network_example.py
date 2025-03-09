"""
Multilayer Perceptron (MLP) Neural Network Example: Fashion Image Classification

This example demonstrates how to use the MLP Neural Network model for image classification,
specifically classifying fashion items from the Fashion MNIST dataset. The example shows:

1. Loading and preprocessing the Fashion MNIST dataset
2. Building and training a MLP Neural Network with different architectures
3. Evaluating model performance and comparing configurations
4. Visualizing training curves and decision boundaries
5. Making predictions on new examples
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import time
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from models.deep_learning.mlp import MLPModel


def load_fashion_mnist(use_sample=True, sample_size=10000):
    """
    Load the Fashion MNIST dataset for image classification.
    
    Args:
        use_sample: Whether to use a smaller sample of the data
        sample_size: Number of examples to use if use_sample is True
    
    Returns:
        X: Features (fashion item images)
        y: Labels (fashion item categories)
        X_images: Reshaped images for visualization
        class_names: Names of the fashion categories
    """
    try:
        print("Loading Fashion MNIST dataset (this may take a moment)...")
        # Load the dataset
        X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, parser='auto')
        
        # Convert labels to integers
        y = y.astype(int)
        
        # If using a sample of the data
        if use_sample:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X = X.iloc[indices]
            y = y.iloc[indices]
        
        # Define class names for visualization
        class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
        
        # Reshape for visualization
        X_images = X.values.reshape(-1, 28, 28)
        
        print(f"Loaded Fashion MNIST dataset with {len(X)} examples")
        print(f"Image dimensions: {X_images[0].shape}")
        return X, y, X_images, class_names
    
    except Exception as e:
        print(f"Error loading Fashion MNIST dataset: {e}")
        print("Falling back to smaller dataset...")
        
        # Fallback to the digits dataset
        digits = load_digits()
        X = digits.data
        y = digits.target
        X_images = digits.images
        class_names = [str(i) for i in range(10)]
        
        print(f"Loaded digits dataset with {len(X)} examples")
        print(f"Image dimensions: {X_images[0].shape}")
        return X, y, X_images, class_names


def visualize_fashion_items(X_images, y, class_names, num_images=10):
    """
    Visualize sample fashion items from the dataset.
    
    Args:
        X_images: Fashion item images
        y: Labels
        class_names: Names of the fashion categories
        num_images: Number of images to display
    """
    plt.figure(figsize=(15, 4))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(X_images[i], cmap='gray')
        plt.title(f"{class_names[y[i]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def train_and_evaluate_mlp(X_train, y_train, X_test, y_test, 
                           hidden_layer_sizes=(100,), activation='relu', 
                           max_iter=200, alpha=0.0001):
    """
    Train and evaluate an MLP Neural Network with the specified parameters.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        hidden_layer_sizes: Tuple with the number of neurons in each hidden layer
        activation: Activation function
        max_iter: Maximum number of training iterations
        alpha: L2 regularization parameter
        
    Returns:
        trained_model: The trained MLP model
        accuracy: Accuracy on the test set
        training_time: Time taken to train the model
    """
    # Format hidden layers for display
    hidden_layers_str = ' -> '.join(str(x) for x in hidden_layer_sizes)
    print(f"\nTraining MLP with architecture: {X_train.shape[1]} -> {hidden_layers_str} -> {len(np.unique(y_train))}")
    print(f"Activation: {activation}, Regularization (alpha): {alpha}, Max iterations: {max_iter}")
    
    # Create and train the model
    mlp_model = MLPModel(
        task_type="classification",
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver='adam',
        alpha=alpha,
        batch_size='auto',
        learning_rate='adaptive',
        max_iter=max_iter,
        random_state=42
    )
    
    # Time the training process
    start_time = time.time()
    mlp_model.train(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate the model
    metrics = mlp_model.evaluate(X_test, y_test)
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return mlp_model, metrics['accuracy'], training_time


def main():
    print("=" * 80)
    print("Multilayer Perceptron (MLP) Neural Network Example: Fashion Image Classification")
    print("=" * 80)
    
    # 1. Load and preprocess the data
    print("\n1. Loading Fashion MNIST dataset...")
    X, y, X_images, class_names = load_fashion_mnist(use_sample=True)
    
    # Visualize some examples
    print("\n2. Visualizing sample fashion items from the dataset...")
    visualize_fashion_items(X_images, y, class_names)
    
    # Split the data into training and testing sets
    print("\n3. Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # Scale the features
    print("\n4. Preprocessing the data (standardization)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Train and evaluate MLP models with different architectures
    print("\n5. Training and evaluating MLP models with different architectures...")
    
    # Define different MLP architectures to try
    mlp_configs = [
        {'hidden_layer_sizes': (50,), 'activation': 'relu', 'max_iter': 100, 'alpha': 0.0001},
        {'hidden_layer_sizes': (100,), 'activation': 'relu', 'max_iter': 100, 'alpha': 0.0001},
        {'hidden_layer_sizes': (50, 50), 'activation': 'relu', 'max_iter': 100, 'alpha': 0.0001},
        {'hidden_layer_sizes': (100, 50), 'activation': 'tanh', 'max_iter': 100, 'alpha': 0.001}
    ]
    
    # Train and evaluate each configuration
    models = []
    accuracies = []
    training_times = []
    
    for config in mlp_configs:
        model, accuracy, training_time = train_and_evaluate_mlp(
            X_train_scaled, y_train, X_test_scaled, y_test, 
            hidden_layer_sizes=config['hidden_layer_sizes'],
            activation=config['activation'],
            max_iter=config['max_iter'],
            alpha=config['alpha']
        )
        models.append(model)
        accuracies.append(accuracy)
        training_times.append(training_time)
    
    # 6. Compare models and select the best one
    print("\n6. Comparing models with different architectures:")
    
    # Prepare comparison data
    config_names = [
        f"MLP ({' -> '.join(str(x) for x in config['hidden_layer_sizes'])}, {config['activation']})"
        for config in mlp_configs
    ]
    
    comparison_data = {
        'Model': config_names,
        'Accuracy': accuracies,
        'Training Time (s)': training_times
    }
    
    for i in range(len(comparison_data['Model'])):
        print(f"{comparison_data['Model'][i]}:")
        print(f"  Accuracy: {comparison_data['Accuracy'][i]:.4f}")
        print(f"  Training Time: {comparison_data['Training Time (s)'][i]:.2f} seconds")
    
    # Select the best model based on accuracy
    best_idx = np.argmax(accuracies)
    best_model = models[best_idx]
    best_config_name = config_names[best_idx]
    
    print(f"\nBest model: {best_config_name} with accuracy: {accuracies[best_idx]:.4f}")
    
    # 7. Plot the training loss curve for the best model
    print("\n7. Visualizing the training loss curve for the best model...")
    best_model.plot_loss_curve()
    
    # 8. Plot confusion matrix for the best model
    print("\n8. Visualizing confusion matrix for the best model...")
    best_model.plot_confusion_matrix(X_test_scaled, y_test, class_names=class_names)
    
    # 9. Visualize the neural network architecture
    print("\n9. Visualizing the neural network architecture...")
    best_model.visualize_network_architecture()
    
    # 10. Apply dimensionality reduction and visualize decision boundaries in 2D
    print("\n10. Visualizing decision boundaries in 2D (using PCA)...")
    
    # Apply PCA to reduce dimensions to 2D for visualization
    pca = PCA(n_components=2)
    X_train_2d = pca.fit_transform(X_train_scaled)
    X_test_2d = pca.transform(X_test_scaled)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")
    
    # Train a new MLP on the 2D data for visualization
    mlp_2d = MLPModel(
        task_type="classification",
        hidden_layer_sizes=(50, 25),
        activation='relu',
        max_iter=100
    )
    mlp_2d.train(X_train_2d, y_train)
    
    # Plot decision boundaries
    mlp_2d.plot_decision_boundary(X_train_2d, y_train)
    
    # 11. Make predictions on new examples
    print("\n11. Making predictions on test examples...")
    
    # Select some test examples
    n_examples = 5
    example_indices = np.random.choice(len(X_test), n_examples, replace=False)
    example_images = X_images[X_test.index[example_indices]]
    example_labels = y_test.iloc[example_indices] if hasattr(y_test, 'iloc') else y_test[example_indices]
    example_scaled = X_test_scaled[example_indices]
    
    # Get predictions from the best model
    predictions = best_model.predict(example_scaled)
    probabilities = best_model.predict_proba(example_scaled)
    
    # Visualize the examples and predictions
    plt.figure(figsize=(15, 6))
    for i in range(n_examples):
        plt.subplot(2, n_examples, i + 1)
        plt.imshow(example_images[i], cmap='gray')
        plt.title(f"True: {class_names[example_labels[i]]}")
        plt.axis('off')
        
        # Plot the prediction probabilities
        plt.subplot(2, n_examples, i + 1 + n_examples)
        top_probs_idx = probabilities[i].argsort()[-3:][::-1]
        top_probs = probabilities[i][top_probs_idx]
        plt.bar(range(3), top_probs)
        plt.xticks(range(3), [class_names[idx] for idx in top_probs_idx], rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.title(f"Pred: {class_names[predictions[i]]}")
    
    plt.tight_layout()
    plt.show()
    
    # 12. Save the best model
    print("\n12. Saving the best model...")
    model_path = "../models/mlp_fashion_classifier.joblib"
    best_model.save_model(model_path)
    
    print("\nExample complete! The MLP Neural Network has been trained for fashion image classification.")


if __name__ == "__main__":
    main() 