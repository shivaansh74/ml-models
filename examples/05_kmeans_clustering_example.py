"""
K-Means Clustering Example

This script demonstrates how to use the K-Means Clustering model for customer segmentation
and other unsupervised learning tasks. It includes data loading, preprocessing, model training,
evaluation, and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_iris, fetch_olivetti_faces
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # For 3D plots

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from models.unsupervised.kmeans import KMeansModel

# Set random seed for reproducibility
np.random.seed(42)

def kmeans_synthetic_example():
    """
    Example of using K-Means Clustering on synthetic data with clear clusters.
    """
    print("\n" + "="*80)
    print(" K-Means Clustering - Synthetic Data Example ".center(80, "="))
    print("="*80)
    
    # Generate synthetic data with 4 well-separated clusters
    print("\n[1] Generating synthetic data with 4 clusters...")
    X, y_true = make_blobs(
        n_samples=300,
        centers=4,
        cluster_std=0.60,
        random_state=42
    )
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find the optimal number of clusters using the Elbow Method
    print("\n[2] Finding the optimal number of clusters using the Elbow Method...")
    kmeans_temp = KMeansModel(random_state=42)
    elbow_plot = kmeans_temp.plot_elbow_method(X_scaled, k_range=range(1, 10))
    plt.tight_layout()
    plt.show()
    
    # Find the optimal number of clusters using the Silhouette Method
    print("\n[3] Finding the optimal number of clusters using the Silhouette Method...")
    silhouette_plot = kmeans_temp.plot_silhouette_method(X_scaled, k_range=range(2, 10))
    plt.tight_layout()
    plt.show()
    
    # Create and train the model with 4 clusters (which should be optimal for our synthetic data)
    print("\n[4] Creating and training the K-Means model with 4 clusters...")
    kmeans = KMeansModel(
        n_clusters=4,
        init='k-means++',
        n_init=10,
        random_state=42
    )
    
    kmeans.train(X_scaled)
    
    # Evaluate the model
    print("\n[5] Evaluating the model...")
    metrics = kmeans.evaluate(X_scaled)
    
    print(f"Inertia (Sum of squared distances to nearest centroid): {metrics['inertia']:.4f}")
    print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"Calinski-Harabasz Index: {metrics['calinski_harabasz_score']:.4f}")
    
    # Get cluster labels and compare with true labels
    labels = kmeans.predict(X_scaled)
    
    # Visualize the clusters
    print("\n[6] Visualizing the clusters...")
    cluster_plot = kmeans.plot_clusters(X_scaled)
    plt.tight_layout()
    plt.show()
    
    print("\n[7] Predicting clusters for new data points...")
    # Generate new sample points
    new_samples = np.random.rand(5, 2) * 10 - 5  # Random points in the range [-5, 5]
    new_samples_scaled = scaler.transform(new_samples)
    
    # Predict clusters for new samples
    new_labels = kmeans.predict(new_samples_scaled)
    
    print("New samples and their predicted clusters:")
    for i, (sample, label) in enumerate(zip(new_samples, new_labels)):
        print(f"Sample {i+1}: {sample} -> Cluster {label}")
    
    print("\nSynthetic data example complete!")


def kmeans_iris_example():
    """
    Example of using K-Means Clustering on the Iris dataset.
    """
    print("\n" + "="*80)
    print(" K-Means Clustering - Iris Dataset Example ".center(80, "="))
    print("="*80)
    
    # Load Iris dataset
    print("\n[1] Loading and preparing the Iris dataset...")
    iris = load_iris()
    X = iris.data
    y_true = iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Target classes: {class_names}")
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create and train the model with 3 clusters (since there are 3 species in Iris)
    print("\n[2] Creating and training the K-Means model with 3 clusters...")
    kmeans = KMeansModel(
        n_clusters=3,
        init='k-means++',
        n_init=10,
        random_state=42
    )
    
    kmeans.train(X_scaled)
    
    # Evaluate the model
    print("\n[3] Evaluating the model...")
    metrics = kmeans.evaluate(X_scaled)
    
    print(f"Inertia: {metrics['inertia']:.4f}")
    print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"Calinski-Harabasz Index: {metrics['calinski_harabasz_score']:.4f}")
    
    # Get cluster labels
    labels = kmeans.predict(X_scaled)
    
    # Compare with true labels
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    
    print(f"\nAdjusted Rand Index: {ari:.4f}")
    print(f"Normalized Mutual Information: {nmi:.4f}")
    
    # Visualize the clusters
    print("\n[4] Visualizing the clusters...")
    
    # 2D visualization
    cluster_plot_2d = kmeans.plot_clusters(X_scaled, feature_names=feature_names, n_components=2)
    plt.tight_layout()
    plt.show()
    
    # 3D visualization
    cluster_plot_3d = kmeans.plot_clusters(X_scaled, feature_names=feature_names, n_components=3)
    plt.tight_layout()
    plt.show()
    
    # Visualize the actual species vs the clusters found by K-means
    print("\n[5] Visualizing comparison between true species and K-means clusters...")
    
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot the actual species
    plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 2, 1)
    for i, species in enumerate(class_names):
        plt.scatter(X_pca[y_true == i, 0], X_pca[y_true == i, 1], 
                   label=species, alpha=0.7)
    plt.title('True Species')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot the K-means clusters
    plt.subplot(1, 2, 2)
    for i in range(3):
        plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], 
                   label=f'Cluster {i}', alpha=0.7)
    plt.title('K-means Clusters')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nIris dataset example complete!")


def kmeans_customer_segmentation_example():
    """
    Example of using K-Means Clustering for customer segmentation.
    This creates a synthetic customer dataset.
    """
    print("\n" + "="*80)
    print(" K-Means Clustering - Customer Segmentation Example ".center(80, "="))
    print("="*80)
    
    # Create synthetic customer data
    print("\n[1] Creating synthetic customer data...")
    n_customers = 500
    
    # Generate synthetic customer features
    np.random.seed(42)
    
    # Annual income (in thousands)
    income = np.random.normal(45, 25, n_customers)
    income = np.clip(income, 0, 150)
    
    # Spending score (1-100)
    spending = np.random.normal(50, 25, n_customers)
    spending = np.clip(spending, 0, 100)
    
    # Age
    age = np.random.normal(40, 15, n_customers)
    age = np.clip(age, 18, 85)
    
    # Loyalty (years as customer)
    loyalty = np.random.normal(5, 4, n_customers)
    loyalty = np.clip(loyalty, 0, 20)
    
    # Create some correlations
    for i in range(n_customers):
        # Some high income customers spend more
        if income[i] > 100:
            spending[i] = min(100, spending[i] * 1.2)
        
        # Some loyal customers spend more
        if loyalty[i] > 10:
            spending[i] = min(100, spending[i] * 1.1)
    
    # Combine into a dataframe
    X = np.column_stack([income, spending, age, loyalty])
    feature_names = ['Income', 'Spending', 'Age', 'Loyalty']
    
    customer_df = pd.DataFrame(X, columns=feature_names)
    
    print("Synthetic customer data summary:")
    print(customer_df.describe())
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find the optimal number of clusters
    print("\n[2] Finding the optimal number of clusters...")
    kmeans_temp = KMeansModel(random_state=42)
    
    elbow_plot = kmeans_temp.plot_elbow_method(X_scaled, k_range=range(1, 11))
    plt.tight_layout()
    plt.show()
    
    silhouette_plot = kmeans_temp.plot_silhouette_method(X_scaled, k_range=range(2, 11))
    plt.tight_layout()
    plt.show()
    
    # Create and train the model with 5 clusters (typically good for customer segmentation)
    print("\n[3] Creating and training the K-Means model with 5 clusters...")
    n_clusters = 5
    kmeans = KMeansModel(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=10,
        random_state=42
    )
    
    kmeans.train(X_scaled)
    
    # Evaluate the model
    print("\n[4] Evaluating the model...")
    metrics = kmeans.evaluate(X_scaled)
    
    print(f"Inertia: {metrics['inertia']:.4f}")
    print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"Calinski-Harabasz Index: {metrics['calinski_harabasz_score']:.4f}")
    
    # Get cluster labels
    labels = kmeans.predict(X_scaled)
    
    # Add cluster labels to the dataframe
    customer_df['Cluster'] = labels
    
    # Visualize the clusters
    print("\n[5] Visualizing the clusters...")
    
    # 2D visualization (Income vs Spending)
    plt.figure(figsize=(12, 8))
    
    # Plot data points for each cluster
    for i in range(n_clusters):
        cluster_data = customer_df[customer_df['Cluster'] == i]
        plt.scatter(cluster_data['Income'], cluster_data['Spending'], 
                   s=50, alpha=0.7, label=f'Cluster {i}')
    
    # Plot cluster centers
    centers = kmeans.model.cluster_centers_
    centers_original = scaler.inverse_transform(centers)
    
    plt.scatter(centers_original[:, 0], centers_original[:, 1], 
               s=200, c='red', marker='X', label='Centroids')
    
    plt.title('Customer Segments')
    plt.xlabel('Income (thousands)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 3D visualization (Income, Spending, Age)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(n_clusters):
        cluster_data = customer_df[customer_df['Cluster'] == i]
        ax.scatter(cluster_data['Income'], cluster_data['Spending'], cluster_data['Age'], 
                  s=50, alpha=0.7, label=f'Cluster {i}')
    
    ax.scatter(centers_original[:, 0], centers_original[:, 1], centers_original[:, 2], 
              s=200, c='red', marker='X', label='Centroids')
    
    ax.set_title('Customer Segments (3D)')
    ax.set_xlabel('Income (thousands)')
    ax.set_ylabel('Spending Score (1-100)')
    ax.set_zlabel('Age')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Analyze the clusters
    print("\n[6] Analyzing the customer segments...")
    
    # Get cluster statistics
    cluster_stats = customer_df.groupby('Cluster').mean()
    cluster_sizes = customer_df.groupby('Cluster').size()
    
    print("\nCluster sizes:")
    for i, size in enumerate(cluster_sizes):
        print(f"Cluster {i}: {size} customers ({size/n_customers:.1%})")
    
    print("\nCluster centers (average values):")
    print(cluster_stats)
    
    # Create a descriptive name for each cluster based on its characteristics
    cluster_descriptions = []
    
    for i in range(n_clusters):
        cluster_center = cluster_stats.iloc[i]
        
        if cluster_center['Income'] > 70 and cluster_center['Spending'] > 60:
            description = "High-Income High-Spenders"
        elif cluster_center['Income'] > 70 and cluster_center['Spending'] < 40:
            description = "High-Income Budget-Conscious"
        elif cluster_center['Income'] < 30 and cluster_center['Spending'] > 60:
            description = "Value-Seeking Spenders"
        elif cluster_center['Age'] > 50:
            description = "Older Customers"
        elif cluster_center['Loyalty'] > 10:
            description = "Loyal Customers"
        else:
            description = "Average Customers"
        
        cluster_descriptions.append(description)
    
    print("\nCluster descriptions:")
    for i, description in enumerate(cluster_descriptions):
        print(f"Cluster {i}: {description}")
    
    print("\nCustomer segmentation example complete!")


if __name__ == "__main__":
    kmeans_synthetic_example()
    kmeans_iris_example()
    kmeans_customer_segmentation_example() 