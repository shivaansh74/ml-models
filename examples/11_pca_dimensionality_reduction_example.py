"""
Principal Component Analysis Example: Dimensionality Reduction and Visualization

This example demonstrates how to use Principal Component Analysis (PCA) for 
dimensionality reduction and visualization of high-dimensional data. The example shows:

1. Loading and exploring the Iris dataset
2. Applying PCA to reduce the dimensionality
3. Analyzing the explained variance and selecting the optimal number of components
4. Visualizing the data in 2D and 3D using the principal components
5. Understanding feature contributions and creating biplots
6. Using PCA for data preprocessing and reconstruction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, fetch_openml, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from models.unsupervised.pca import PCAModel


def load_dataset(dataset_name='iris'):
    """
    Load a dataset for dimensionality reduction.
    
    Args:
        dataset_name: Name of the dataset to load ('iris', 'digits', or 'mnist')
        
    Returns:
        X: Features
        y: Target labels
        feature_names: Names of the features
        target_names: Names of the target classes
    """
    if dataset_name == 'iris':
        # Load Iris dataset (4 features)
        data = load_iris()
        X = data.data
        y = data.target
        feature_names = data.feature_names
        target_names = data.target_names
        print(f"Loaded Iris dataset with {X.shape[0]} samples, {X.shape[1]} features")
        
    elif dataset_name == 'digits':
        # Load digits dataset (64 features)
        data = load_digits()
        X = data.data
        y = data.target
        feature_names = [f"pixel_{i}" for i in range(X.shape[1])]
        target_names = [str(i) for i in range(10)]
        print(f"Loaded Digits dataset with {X.shape[0]} samples, {X.shape[1]} features")
        
    elif dataset_name == 'mnist':
        try:
            # Load MNIST dataset (784 features)
            # This will download the dataset if it's not available locally
            print("Loading MNIST dataset (this may take a moment)...")
            data = fetch_openml('mnist_784', version=1, parser='auto', cache=True)
            X = data.data.to_numpy()
            y = data.target.astype(int).to_numpy()
            feature_names = [f"pixel_{i}" for i in range(X.shape[1])]
            target_names = [str(i) for i in range(10)]
            
            # Use a subset of the data to speed up computation
            print("Using a subset of MNIST for faster computation...")
            n_samples = 2000
            indices = np.random.choice(X.shape[0], n_samples, replace=False)
            X = X[indices]
            y = y[indices]
            
            print(f"Loaded MNIST subset with {X.shape[0]} samples, {X.shape[1]} features")
        except Exception as e:
            print(f"Error loading MNIST: {e}")
            print("Falling back to digits dataset...")
            return load_dataset('digits')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return X, y, feature_names, target_names


def explore_dataset(X, y, feature_names, target_names):
    """
    Explore the dataset and visualize its properties.
    
    Args:
        X: Features
        y: Target labels
        feature_names: Names of the features
        target_names: Names of the target classes
    """
    # Create a DataFrame for easier exploration
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    df['target_name'] = [target_names[i] for i in y]
    
    print("\nDataset shape:", X.shape)
    print("Number of classes:", len(target_names))
    print("Class distribution:")
    print(df['target_name'].value_counts())
    
    # Only create pairplot for datasets with reasonable number of features
    if X.shape[1] <= 10:
        print("\nCreating pairplot (this might take a moment)...")
        
        # For Iris dataset, create a pairplot
        if len(feature_names) == 4:  # Iris has 4 features
            sns.pairplot(df, hue='target_name', vars=feature_names, diag_kind='kde')
            plt.suptitle('Pairplot of Features by Class', y=1.02)
            plt.show()
        else:
            # For other small datasets, show first few features
            selected_features = feature_names[:4]
            sns.pairplot(df, hue='target_name', vars=selected_features, diag_kind='kde')
            plt.suptitle('Pairplot of First 4 Features by Class', y=1.02)
            plt.show()
    
    # For image datasets, visualize some examples
    if X.shape[1] in [64, 784]:  # 8x8 digits or 28x28 MNIST
        plt.figure(figsize=(15, 3))
        
        for i in range(5):
            plt.subplot(1, 5, i + 1)
            
            # Reshape based on dataset
            if X.shape[1] == 64:  # 8x8 digits
                plt.imshow(X[i].reshape(8, 8), cmap='gray')
            else:  # 28x28 MNIST
                plt.imshow(X[i].reshape(28, 28), cmap='gray')
            
            plt.title(f"Class: {target_names[y[i]]}")
            plt.axis('off')
            
        plt.suptitle('Sample Images', y=1.02)
        plt.tight_layout()
        plt.show()


def train_and_evaluate_pca(X, y, feature_names, target_names):
    """
    Train a PCA model and evaluate its performance.
    
    Args:
        X: Features
        y: Target labels
        feature_names: Names of the features
        target_names: Names of the target classes
    
    Returns:
        pca_model: Trained PCA model
    """
    # 1. Initialize PCA model
    print("\n1. Initializing PCA model...")
    pca_model = PCAModel(n_components=None)  # Keep all components initially
    
    # 2. Train the model
    print("\n2. Training PCA model...")
    pca_model.train(X, feature_names=feature_names)
    
    # 3. Get explained variance ratio
    explained_variance_ratio = pca_model.get_explained_variance_ratio()
    cumulative_variance = pca_model.get_cumulative_explained_variance()
    
    print("\nExplained variance by component:")
    for i, variance in enumerate(explained_variance_ratio[:10]):
        print(f"PC{i+1}: {variance:.4f} ({cumulative_variance[i]:.4f} cumulative)")
    
    # 4. Plot explained variance
    print("\n3. Analyzing explained variance...")
    pca_model.plot_explained_variance()
    
    # 5. Find optimal number of components for 95% variance
    n_components_95 = pca_model.get_n_components_for_variance(0.95)
    print(f"\nNumber of components needed for 95% variance: {n_components_95}")
    
    # 6. Create a new PCA model with optimal number of components
    if n_components_95 < X.shape[1]:
        print(f"\n4. Creating new PCA model with {n_components_95} components...")
        pca_model_optimal = PCAModel(n_components=n_components_95)
        pca_model_optimal.train(X, feature_names=feature_names)
    else:
        pca_model_optimal = pca_model
    
    return pca_model, pca_model_optimal


def visualize_pca_results(pca_model, X, y, feature_names, target_names):
    """
    Visualize the results of PCA.
    
    Args:
        pca_model: Trained PCA model
        X: Features
        y: Target labels
        feature_names: Names of the features
        target_names: Names of the target classes
    """
    # 1. Transform the data
    print("\n5. Transforming data with PCA...")
    X_pca = pca_model.transform(X)
    
    # 2. Visualize in 2D
    print("\n6. Visualizing data in 2D...")
    pca_model.plot_components_2d(X, y, components=(0, 1), class_names=target_names)
    
    # 3. Visualize in 3D if there are at least 3 components
    if X_pca.shape[1] >= 3:
        print("\n7. Visualizing data in 3D...")
        pca_model.plot_components_3d(X, y, components=(0, 1, 2), class_names=target_names)
    
    # 4. Plot feature contributions
    print("\n8. Analyzing feature contributions...")
    pca_model.plot_feature_contributions(n_components=min(3, X_pca.shape[1]), feature_names=feature_names)
    
    # 5. Create biplot
    if X.shape[1] < 30:  # Only for datasets with reasonable number of features
        print("\n9. Creating biplot...")
        pca_model.plot_biplot(X, feature_names=feature_names, components=(0, 1))


def evaluate_reconstruction(pca_model, X):
    """
    Evaluate the reconstruction quality of PCA.
    
    Args:
        pca_model: Trained PCA model
        X: Original data
    """
    print("\n10. Evaluating reconstruction quality...")
    
    # Transform the data
    X_pca = pca_model.transform(X)
    
    # Reconstruct the data
    X_reconstructed = pca_model.inverse_transform(X_pca)
    
    # Calculate reconstruction error
    mse = np.mean(np.sum((X - X_reconstructed) ** 2, axis=1))
    print(f"Reconstruction MSE: {mse:.4f}")
    
    # Plot reconstruction error for different numbers of components
    pca_model.plot_reconstruction_error(X)
    
    # For image data, visualize original vs reconstructed
    if X.shape[1] in [64, 784]:
        print("\n11. Visualizing original vs reconstructed images...")
        
        plt.figure(figsize=(15, 6))
        for i in range(5):
            # Original image
            plt.subplot(2, 5, i + 1)
            
            if X.shape[1] == 64:  # 8x8 digits
                plt.imshow(X[i].reshape(8, 8), cmap='gray')
            else:  # 28x28 MNIST
                plt.imshow(X[i].reshape(28, 28), cmap='gray')
                
            plt.title(f"Original {i+1}")
            plt.axis('off')
            
            # Reconstructed image
            plt.subplot(2, 5, i + 6)
            
            if X.shape[1] == 64:
                plt.imshow(X_reconstructed[i].reshape(8, 8), cmap='gray')
            else:
                plt.imshow(X_reconstructed[i].reshape(28, 28), cmap='gray')
                
            plt.title(f"Reconstructed {i+1}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()


def pca_for_preprocessing(X, y, pca_model, target_names):
    """
    Demonstrate using PCA as a preprocessing step for classification.
    
    Args:
        X: Features
        y: Target labels
        pca_model: Trained PCA model
        target_names: Names of the target classes
    """
    print("\n12. Using PCA for preprocessing in a classification task...")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train a classifier on the original data
    clf_original = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_original.fit(X_train, y_train)
    y_pred_original = clf_original.predict(X_test)
    accuracy_original = accuracy_score(y_test, y_pred_original)
    
    print(f"Classifier on original data ({X_train.shape[1]} features):")
    print(f"Accuracy: {accuracy_original:.4f}")
    
    # Transform the data with PCA
    X_train_pca = pca_model.transform(X_train)
    X_test_pca = pca_model.transform(X_test)
    
    # Train a classifier on the PCA-transformed data
    clf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_pca.fit(X_train_pca, y_train)
    y_pred_pca = clf_pca.predict(X_test_pca)
    accuracy_pca = accuracy_score(y_test, y_pred_pca)
    
    print(f"Classifier on PCA-transformed data ({X_train_pca.shape[1]} components):")
    print(f"Accuracy: {accuracy_pca:.4f}")
    
    # Compare the results
    print("\nClassification Report (Original):")
    print(classification_report(y_test, y_pred_original, target_names=target_names))
    
    print("\nClassification Report (PCA):")
    print(classification_report(y_test, y_pred_pca, target_names=target_names))
    
    # Compare the training times (if we had trained the models)
    # Note: In a real application, we would time the training process
    
    # Plot confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original data confusion matrix
    cm_original = pd.crosstab(
        pd.Series(y_test, name='Actual'), 
        pd.Series(y_pred_original, name='Predicted'),
        rownames=['Actual'], 
        colnames=['Predicted']
    )
    
    sns.heatmap(cm_original, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Confusion Matrix (Original)')
    
    # PCA data confusion matrix
    cm_pca = pd.crosstab(
        pd.Series(y_test, name='Actual'), 
        pd.Series(y_pred_pca, name='Predicted'),
        rownames=['Actual'], 
        colnames=['Predicted']
    )
    
    sns.heatmap(cm_pca, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title('Confusion Matrix (PCA)')
    
    plt.tight_layout()
    plt.show()


def main():
    print("=" * 80)
    print("Principal Component Analysis Example: Dimensionality Reduction and Visualization")
    print("=" * 80)
    
    # Ask user to select a dataset
    print("\nSelect a dataset:")
    print("1. Iris (4 features, 3 classes)")
    print("2. Digits (64 features, 10 classes)")
    print("3. MNIST subset (784 features, 10 classes)")
    
    try:
        choice = int(input("Enter your choice (1-3) [default=1]: ") or "1")
        if choice == 1:
            dataset_name = 'iris'
        elif choice == 2:
            dataset_name = 'digits'
        elif choice == 3:
            dataset_name = 'mnist'
        else:
            print("Invalid choice, using Iris dataset.")
            dataset_name = 'iris'
    except:
        print("Invalid input, using Iris dataset.")
        dataset_name = 'iris'
    
    # 1. Load dataset
    print(f"\nLoading {dataset_name} dataset...")
    X, y, feature_names, target_names = load_dataset(dataset_name)
    
    # 2. Explore the dataset
    print("\nExploring the dataset...")
    explore_dataset(X, y, feature_names, target_names)
    
    # 3. Train PCA model and analyze variance
    pca_model, pca_model_optimal = train_and_evaluate_pca(X, y, feature_names, target_names)
    
    # 4. Visualize PCA results
    visualize_pca_results(pca_model, X, y, feature_names, target_names)
    
    # 5. Evaluate reconstruction quality
    evaluate_reconstruction(pca_model_optimal, X)
    
    # 6. Use PCA for preprocessing in a classification task
    pca_for_preprocessing(X, y, pca_model_optimal, target_names)
    
    # 7. Save the optimal model
    print("\n13. Saving the PCA model...")
    model_path = f"../models/pca_{dataset_name}_model.joblib"
    pca_model_optimal.save_model(model_path)
    
    print("\nExample complete! PCA has been used for dimensionality reduction and visualization.")


if __name__ == "__main__":
    main() 