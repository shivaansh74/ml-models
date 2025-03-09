"""
Gaussian Mixture Model (GMM) Implementation

This module provides a Gaussian Mixture Model implementation that follows
the standardized template structure. GMMs are probabilistic models that 
represent a mixture of multiple Gaussian distributions, useful for clustering
and anomaly detection tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
import joblib
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.model_template import MLModel


class GMMModel(MLModel):
    """
    Gaussian Mixture Model for clustering and anomaly detection.
    
    Use case: Anomaly detection, density estimation, clustering with 
    soft assignments, and modeling complex data distributions.
    """
    
    def __init__(self, n_components=2, covariance_type='full', tol=1e-3,
                reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                weights_init=None, means_init=None, precisions_init=None,
                random_state=None, **kwargs):
        """
        Initialize the Gaussian Mixture Model.
        
        Args:
            n_components: Number of mixture components
            covariance_type: Type of covariance ('full', 'tied', 'diag', 'spherical')
            tol: Convergence threshold
            reg_covar: Regularization added to covariance matrices
            max_iter: Maximum number of EM iterations
            n_init: Number of initializations
            init_params: Parameters initialization method
            weights_init: Initial weights
            means_init: Initial means
            precisions_init: Initial precision matrices
            random_state: Random seed
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.model_type = "Gaussian Mixture Model"
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init,
            random_state=random_state
        )
        
        self.model_params = {
            'n_components': n_components,
            'covariance_type': covariance_type,
            'tol': tol,
            'reg_covar': reg_covar,
            'max_iter': max_iter,
            'n_init': n_init,
            'init_params': init_params,
            'random_state': random_state
        }
    
    def train(self, X_train, y_train=None, scale_data=True, **kwargs):
        """
        Train the GMM model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Not used for GMM (unsupervised)
            scale_data: Whether to scale the data
            **kwargs: Additional training parameters
            
        Returns:
            self: The trained model instance
        """
        # Update parameters if provided
        for key, value in kwargs.items():
            if key in self.model_params:
                self.model_params[key] = value
                setattr(self, key, value)
                
                # Update sklearn GMM model parameters
                if hasattr(self.model, key):
                    setattr(self.model, key, value)
        
        # Scale data if needed
        if scale_data:
            X_scaled = self.scaler.fit_transform(X_train)
        else:
            X_scaled = X_train
        
        # Fit the model
        self.model.fit(X_scaled)
        
        self.is_trained = True
        return self
    
    def predict(self, X):
        """
        Predict cluster labels for the data.
        
        Args:
            X: Features to make predictions on
            
        Returns:
            Predicted cluster labels
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Predict cluster labels
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Predict probabilities of cluster membership.
        
        Args:
            X: Features to make predictions on
            
        Returns:
            Probabilities of each cluster for each sample
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Predict probabilities
        return self.model.predict_proba(X_scaled)
    
    def score_samples(self, X):
        """
        Compute the log-likelihood of each sample.
        
        Args:
            X: Features to score
            
        Returns:
            Log-likelihood of each sample under the model
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Score samples
        return self.model.score_samples(X_scaled)
    
    def detect_anomalies(self, X, threshold=None, contamination=0.05):
        """
        Detect anomalies in the data.
        
        Args:
            X: Data to check for anomalies
            threshold: Log-likelihood threshold (if None, calculated based on contamination)
            contamination: Expected proportion of outliers in the data
            
        Returns:
            Boolean array indicating anomalies (True) and normal points (False)
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Compute log-likelihood of each sample
        log_probs = self.model.score_samples(X_scaled)
        
        # Determine threshold if not provided
        if threshold is None:
            # Sort log probabilities and find threshold at contamination percentile
            sorted_log_probs = np.sort(log_probs)
            threshold = sorted_log_probs[int(len(X) * contamination)]
        
        # Classify points with log probability below threshold as anomalies
        return log_probs < threshold, log_probs, threshold
    
    def evaluate(self, X_test, y_test=None):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Not typically used for GMM
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Scale the data
        X_scaled = self.scaler.transform(X_test)
        
        # Calculate metrics
        bic = self.model.bic(X_scaled)
        aic = self.model.aic(X_scaled)
        avg_log_likelihood = np.mean(self.model.score_samples(X_scaled))
        
        return {
            'bic': bic,
            'aic': aic,
            'avg_log_likelihood': avg_log_likelihood
        }
    
    def plot_clusters(self, X, feature_names=None, reduce_dim=True, figsize=(12, 8)):
        """
        Plot the data points colored by their assigned clusters.
        
        Args:
            X: Data to plot
            feature_names: Names of the features
            reduce_dim: Whether to reduce dimensionality for visualization
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Get cluster assignments
        labels = self.model.predict(X_scaled)
        
        fig = plt.figure(figsize=figsize)
        
        # Handle high dimensional data
        if X.shape[1] > 2 and reduce_dim:
            # Reduce to 2D for visualization
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X_scaled)
            
            # Plot 2D projection
            ax = fig.add_subplot(111)
            scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
            
            # Add cluster centers
            transformed_centers = pca.transform(self.model.means_)
            ax.scatter(transformed_centers[:, 0], transformed_centers[:, 1], 
                      marker='x', s=100, color='red', label='Cluster Centers')
            
            if feature_names is not None:
                ax.set_xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
                ax.set_ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
            else:
                ax.set_xlabel("Principal Component 1")
                ax.set_ylabel("Principal Component 2")
                
            cbar = plt.colorbar(scatter)
            cbar.set_label('Cluster')
            
        elif X.shape[1] == 2:
            # Plot 2D data directly
            ax = fig.add_subplot(111)
            scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', alpha=0.7)
            
            # Add cluster centers
            ax.scatter(self.model.means_[:, 0], self.model.means_[:, 1],
                      marker='x', s=100, color='red', label='Cluster Centers')
            
            # Draw ellipses for each cluster (for 'full' covariance)
            if self.covariance_type == 'full':
                for i, (mean, covar) in enumerate(zip(self.model.means_, self.model.covariances_)):
                    v, w = np.linalg.eigh(covar)
                    u = w[0] / np.linalg.norm(w[0])
                    angle = np.arctan2(u[1], u[0])
                    angle = 180 * angle / np.pi  # Convert to degrees
                    
                    # Eigenvalues give length of axes
                    v = 2. * np.sqrt(2.) * np.sqrt(v)
                    ell = patches.Ellipse(mean, v[0], v[1], angle=180 + angle, 
                                          alpha=0.3, color=plt.cm.viridis(i / self.n_components))
                    ax.add_artist(ell)
            
            if feature_names is not None and len(feature_names) >= 2:
                ax.set_xlabel(feature_names[0])
                ax.set_ylabel(feature_names[1])
            else:
                ax.set_xlabel("Feature 1")
                ax.set_ylabel("Feature 2")
                
            cbar = plt.colorbar(scatter)
            cbar.set_label('Cluster')
            
        elif X.shape[1] == 1:
            # Plot 1D data
            ax = fig.add_subplot(111)
            
            # Create a scatter plot for 1D data by adding a dummy dimension
            scatter = ax.scatter(X_scaled, np.zeros_like(X_scaled), c=labels, cmap='viridis', alpha=0.7)
            
            # Add cluster centers
            ax.scatter(self.model.means_, np.zeros_like(self.model.means_), 
                      marker='x', s=100, color='red', label='Cluster Centers')
            
            if feature_names is not None and len(feature_names) >= 1:
                ax.set_xlabel(feature_names[0])
            else:
                ax.set_xlabel("Feature 1")
                
            ax.set_yticks([])  # Hide y-axis as it's meaningless
            ax.set_ylabel("Density")
            
            cbar = plt.colorbar(scatter)
            cbar.set_label('Cluster')
            
        else:
            # Plot pair grid for 3+ dimensions
            if reduce_dim:
                ax = fig.add_subplot(111, projection='3d')
                pca = PCA(n_components=3)
                X_3d = pca.fit_transform(X_scaled)
                
                scatter = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=labels, cmap='viridis', alpha=0.7)
                
                # Add cluster centers
                transformed_centers = pca.transform(self.model.means_)
                ax.scatter(transformed_centers[:, 0], transformed_centers[:, 1], transformed_centers[:, 2],
                          marker='x', s=100, color='red', label='Cluster Centers')
                
                if feature_names is not None:
                    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
                    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
                    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.2%})")
                else:
                    ax.set_xlabel("PC1")
                    ax.set_ylabel("PC2")
                    ax.set_zlabel("PC3")
                    
                cbar = plt.colorbar(scatter)
                cbar.set_label('Cluster')
        
        plt.title(f"GMM Clustering with {self.n_components} Components")
        plt.tight_layout()
        return fig
    
    def plot_density(self, X, feature_idx=(0, 1), resolution=100, figsize=(10, 8)):
        """
        Plot the density contours of the Gaussian mixture.
        
        Args:
            X: Data to plot
            feature_idx: Indices of the two features to plot
            resolution: Resolution of the density grid
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        if X.shape[1] < 2:
            raise ValueError("Density plot requires at least 2 features")
        
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Get cluster assignments
        labels = self.model.predict(X_scaled)
        
        # Select the two features to plot
        idx1, idx2 = feature_idx
        
        # Create a mesh grid for density visualization
        x_min, x_max = X_scaled[:, idx1].min() - 1, X_scaled[:, idx1].max() + 1
        y_min, y_max = X_scaled[:, idx2].min() - 1, X_scaled[:, idx2].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                             np.linspace(y_min, y_max, resolution))
        
        # Prepare grid data for scoring
        grid_data = np.zeros((resolution * resolution, X.shape[1]))
        for i in range(X.shape[1]):
            if i == idx1:
                grid_data[:, i] = xx.ravel()
            elif i == idx2:
                grid_data[:, i] = yy.ravel()
            else:
                # Set other dimensions to mean value
                grid_data[:, i] = np.mean(X_scaled[:, i])
        
        # Score the grid points
        Z = -self.model.score_samples(grid_data)
        Z = Z.reshape(xx.shape)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot density contours
        contour = ax.contourf(xx, yy, Z, levels=50, cmap='viridis_r')
        cbar = plt.colorbar(contour)
        cbar.set_label('Negative Log-Likelihood')
        
        # Plot data points
        scatter = ax.scatter(X_scaled[:, idx1], X_scaled[:, idx2], c=labels, 
                           edgecolors='k', s=40, alpha=0.6, cmap='viridis')
        
        # Plot cluster centers
        ax.scatter(self.model.means_[:, idx1], self.model.means_[:, idx2],
                 marker='x', s=100, linewidths=2, color='red', label='Cluster Centers')
        
        if self.covariance_type == 'full':
            # Draw ellipses for each cluster
            for i, (mean, covar) in enumerate(zip(self.model.means_, self.model.covariances_)):
                covar_2d = np.array([[covar[idx1, idx1], covar[idx1, idx2]],
                                     [covar[idx2, idx1], covar[idx2, idx2]]])
                v, w = np.linalg.eigh(covar_2d)
                u = w[0] / np.linalg.norm(w[0])
                angle = np.arctan2(u[1], u[0])
                angle = 180 * angle / np.pi  # Convert to degrees
                
                # 2.5 sigma ellipse (covers ~99% of the data for a Gaussian)
                v = 2.5 * np.sqrt(v)
                ell = patches.Ellipse(mean[[idx1, idx2]], v[0], v[1], angle=180 + angle, 
                                     edgecolor='black', facecolor='none', linewidth=1.5)
                ax.add_artist(ell)
        
        ax.set_xlabel(f'Feature {idx1}')
        ax.set_ylabel(f'Feature {idx2}')
        ax.set_title(f'GMM Density Plot with {self.n_components} Components')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_anomaly_detection(self, X, contamination=0.05, feature_idx=(0, 1), figsize=(10, 8)):
        """
        Plot the data points colored by their anomaly status.
        
        Args:
            X: Data to plot
            contamination: Expected proportion of outliers
            feature_idx: Indices of the two features to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Detect anomalies
        is_anomaly, log_probs, threshold = self.detect_anomalies(X, contamination=contamination)
        
        fig = plt.figure(figsize=figsize)
        
        # Handle high dimensional data
        if X.shape[1] > 2:
            # Reduce to 2D for visualization
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X_scaled)
            
            # Plot 2D projection
            ax = fig.add_subplot(111)
            
            # Plot normal points
            ax.scatter(X_2d[~is_anomaly, 0], X_2d[~is_anomaly, 1], 
                      c='blue', label='Normal', alpha=0.6)
            
            # Plot anomalies
            ax.scatter(X_2d[is_anomaly, 0], X_2d[is_anomaly, 1], 
                      c='red', label='Anomaly', alpha=0.6)
            
            # Add cluster centers
            transformed_centers = pca.transform(self.model.means_)
            ax.scatter(transformed_centers[:, 0], transformed_centers[:, 1], 
                      marker='x', s=100, color='green', label='Cluster Centers')
            
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            
        elif X.shape[1] == 2:
            # Plot 2D data directly
            ax = fig.add_subplot(111)
            
            # Plot normal points
            ax.scatter(X_scaled[~is_anomaly, 0], X_scaled[~is_anomaly, 1], 
                      c='blue', label='Normal', alpha=0.6)
            
            # Plot anomalies
            ax.scatter(X_scaled[is_anomaly, 0], X_scaled[is_anomaly, 1], 
                      c='red', label='Anomaly', alpha=0.6)
            
            # Add cluster centers
            ax.scatter(self.model.means_[:, 0], self.model.means_[:, 1],
                      marker='x', s=100, color='green', label='Cluster Centers')
            
            # Draw decision boundary
            if X.shape[1] == 2:
                # Create a mesh grid
                x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
                y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                     np.linspace(y_min, y_max, 100))
                
                # Calculate log probabilities for each point in the grid
                Z = np.zeros((100, 100))
                for i in range(100):
                    for j in range(100):
                        Z[i, j] = self.model.score_samples(np.array([[xx[i, j], yy[i, j]]]))
                
                # Draw the contour at the threshold level
                ax.contour(xx, yy, Z, levels=[threshold], colors='red', linestyles='dashed')
            
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            
        else:  # 1D data
            ax = fig.add_subplot(111)
            
            # Create a scatter plot for 1D data
            ax.scatter(X_scaled[~is_anomaly], np.zeros(np.sum(~is_anomaly)), 
                      c='blue', label='Normal', alpha=0.6)
            ax.scatter(X_scaled[is_anomaly], np.zeros(np.sum(is_anomaly)), 
                      c='red', label='Anomaly', alpha=0.6)
            
            # Add cluster centers
            ax.scatter(self.model.means_, np.zeros_like(self.model.means_), 
                      marker='x', s=100, color='green', label='Cluster Centers')
            
            ax.set_xlabel("Feature 1")
            ax.set_yticks([])  # Hide y-axis
        
        ax.set_title("GMM Anomaly Detection")
        ax.legend()
        
        plt.tight_layout()
        
        # Add a second figure with the distribution of log probabilities
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.histplot(log_probs, bins=50, ax=ax2)
        ax2.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.2f})')
        ax2.set_xlabel('Log Probability')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Log Probabilities with Anomaly Threshold')
        ax2.legend()
        
        plt.tight_layout()
        
        return fig, fig2
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        model_data = {
            'model': self.model,
            'model_params': self.model_params,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            self: The loaded model instance
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_params = model_data['model_params']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        
        # Set attributes from model_params
        for key, value in self.model_params.items():
            setattr(self, key, value)
        
        return self 