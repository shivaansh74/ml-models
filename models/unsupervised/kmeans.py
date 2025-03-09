"""
K-Means Clustering Model Implementation

This module provides a K-Means Clustering model implementation that follows
the standardized template structure. K-Means is an unsupervised learning algorithm
that groups data points into K distinct clusters based on distance to the centroid of each cluster.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import joblib

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.model_template import MLModel


class KMeansModel(MLModel):
    """
    K-Means Clustering model for unsupervised learning tasks.
    
    Use cases:
    - Customer segmentation for marketing campaigns
    - Document clustering for topic modeling
    - Image compression
    - Anomaly detection
    """
    
    def __init__(self, n_clusters=8, init='k-means++', n_init=10, max_iter=300, 
                 tol=1e-4, random_state=None, **kwargs):
        """
        Initialize the K-Means Clustering model.
        
        Args:
            n_clusters: Number of clusters to form
            init: Method for initialization ('k-means++', 'random', or ndarray)
            n_init: Number of times to run the algorithm with different centroid seeds
            max_iter: Maximum number of iterations for a single run
            tol: Relative tolerance for convergence
            random_state: Seed for random number generation
            **kwargs: Additional parameters for the sklearn KMeans model
        """
        super().__init__(**kwargs)
        self.model_type = "K-Means Clustering"
        self.model_params = {
            'n_clusters': n_clusters,
            'init': init,
            'n_init': n_init,
            'max_iter': max_iter,
            'tol': tol,
            'random_state': random_state,
            **kwargs
        }
        
        self.model = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            **kwargs
        )
        
        self.fitted = False

    def train(self, X, **kwargs):
        """
        Train the K-Means Clustering model.
        
        Args:
            X: Training data (features)
            **kwargs: Additional parameters for fit method
            
        Returns:
            self: The trained model instance
        """
        self.model.fit(X, **kwargs)
        self.fitted = True
        self._log_training_complete()
        return self

    def predict(self, X):
        """
        Predict which cluster each sample in X belongs to.
        
        Args:
            X: Data to cluster
            
        Returns:
            Cluster labels for each point
        """
        if not self.fitted:
            raise ValueError("Model has not been trained yet. Call 'train' first.")
        
        return self.model.predict(X)
    
    def transform(self, X):
        """
        Transform X to a cluster-distance space.
        
        Args:
            X: Data to transform
            
        Returns:
            Distances to each cluster center
        """
        if not self.fitted:
            raise ValueError("Model has not been trained yet. Call 'train' first.")
        
        return self.model.transform(X)
    
    def fit_predict(self, X):
        """
        Compute cluster centers and predict cluster index for each sample.
        
        Args:
            X: Data to fit and cluster
            
        Returns:
            Cluster labels for each point
        """
        self.model.fit(X)
        self.fitted = True
        self._log_training_complete()
        return self.model.labels_

    def evaluate(self, X, labels=None):
        """
        Evaluate the model using clustering metrics.
        
        Args:
            X: Data samples
            labels: True labels (if available)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.fitted:
            raise ValueError("Model has not been trained yet. Call 'train' first.")
        
        metrics = {}
        
        # Use predicted labels if none provided
        if labels is None:
            labels = self.model.labels_
        
        # Inertia (sum of squared distances to nearest centroid)
        metrics['inertia'] = self.model.inertia_
        
        # Silhouette score (measure of how similar a point is to its own cluster compared to other clusters)
        try:
            metrics['silhouette_score'] = silhouette_score(X, labels)
        except:
            metrics['silhouette_score'] = None
            
        # Calinski-Harabasz Index (ratio of between-cluster to within-cluster dispersion)
        try:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
        except:
            metrics['calinski_harabasz_score'] = None
            
        return metrics
    
    def plot_clusters(self, X, feature_names=None, n_components=2, figsize=(12, 8)):
        """
        Visualize the clusters in 2D using PCA for dimensionality reduction if needed.
        
        Args:
            X: Data samples
            feature_names: Names of the features
            n_components: Number of PCA components (2 or 3)
            figsize: Size of the figure
            
        Returns:
            Matplotlib figure
        """
        if not self.fitted:
            raise ValueError("Model has not been trained yet. Call 'train' first.")
        
        # Check if X has more than 2 dimensions
        if X.shape[1] > n_components:
            # Apply PCA for visualization
            pca = PCA(n_components=n_components)
            X_reduced = pca.fit_transform(X)
            explained_variance = pca.explained_variance_ratio_.sum()
        else:
            X_reduced = X
            explained_variance = 1.0
        
        # Get cluster labels and centers
        labels = self.model.labels_
        centers = self.model.cluster_centers_
        n_clusters = len(centers)
        
        # Create a colormap
        cmap = plt.cm.get_cmap('viridis', n_clusters)
        
        if n_components == 2:
            # 2D plot
            plt.figure(figsize=figsize)
            
            # Plot data points with cluster colors
            for i in range(n_clusters):
                plt.scatter(X_reduced[labels == i, 0], X_reduced[labels == i, 1], 
                           s=50, c=[cmap(i)], alpha=0.5, label=f'Cluster {i}')
            
            # If PCA was applied, transform centers to PCA space
            if X.shape[1] > 2:
                centers_reduced = pca.transform(centers)
            else:
                centers_reduced = centers
                
            # Plot cluster centers
            plt.scatter(centers_reduced[:, 0], centers_reduced[:, 1], 
                       s=200, c='red', marker='X', label='Centroids')
            
            plt.title(f'K-Means Clustering (K={n_clusters})\n' + 
                     (f'PCA Explained Variance: {explained_variance:.2%}' if X.shape[1] > 2 else ''))
            
            if feature_names is not None and X.shape[1] <= 2:
                plt.xlabel(feature_names[0])
                plt.ylabel(feature_names[1])
            else:
                plt.xlabel('Component 1')
                plt.ylabel('Component 2')
                
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        elif n_components == 3:
            # 3D plot
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot data points with cluster colors
            for i in range(n_clusters):
                ax.scatter(X_reduced[labels == i, 0], X_reduced[labels == i, 1], X_reduced[labels == i, 2],
                          s=50, c=[cmap(i)], alpha=0.5, label=f'Cluster {i}')
            
            # If PCA was applied, transform centers to PCA space
            if X.shape[1] > 3:
                centers_reduced = pca.transform(centers)
            else:
                centers_reduced = centers
                
            # Plot cluster centers
            ax.scatter(centers_reduced[:, 0], centers_reduced[:, 1], centers_reduced[:, 2],
                      s=200, c='red', marker='X', label='Centroids')
            
            ax.set_title(f'K-Means Clustering (K={n_clusters})\n' + 
                        (f'PCA Explained Variance: {explained_variance:.2%}' if X.shape[1] > 3 else ''))
            
            if feature_names is not None and X.shape[1] <= 3:
                ax.set_xlabel(feature_names[0])
                ax.set_ylabel(feature_names[1])
                ax.set_zlabel(feature_names[2])
            else:
                ax.set_xlabel('Component 1')
                ax.set_ylabel('Component 2')
                ax.set_zlabel('Component 3')
                
            plt.legend()
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_elbow_method(self, X, k_range=range(1, 11), figsize=(12, 6)):
        """
        Plot the Elbow method to find the optimal K value.
        
        Args:
            X: Data samples
            k_range: Range of K values to try
            figsize: Size of the figure
            
        Returns:
            Matplotlib figure
        """
        inertia_values = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, **{k: v for k, v in self.model_params.items() if k != 'n_clusters'})
            kmeans.fit(X)
            inertia_values.append(kmeans.inertia_)
        
        plt.figure(figsize=figsize)
        plt.plot(k_range, inertia_values, 'bo-')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Inertia (Sum of Squared Distances)')
        plt.title('Elbow Method for Optimal K')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_silhouette_method(self, X, k_range=range(2, 11), figsize=(12, 6)):
        """
        Plot the Silhouette method to find the optimal K value.
        
        Args:
            X: Data samples
            k_range: Range of K values to try
            figsize: Size of the figure
            
        Returns:
            Matplotlib figure
        """
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, **{k: v for k, v in self.model_params.items() if k != 'n_clusters'})
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)
        
        plt.figure(figsize=figsize)
        plt.plot(k_range, silhouette_scores, 'bo-')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Method for Optimal K')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()

    def save_model(self, filepath):
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            Path where the model was saved
        """
        if not self.fitted:
            raise ValueError("Model has not been trained yet. Call 'train' first.")
            
        joblib.dump(self, filepath)
        return filepath

    def load_model(self, filepath):
        """
        Load the model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        loaded_model = joblib.load(filepath)
        self.__dict__.update(loaded_model.__dict__)
        return self 