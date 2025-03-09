"""
Principal Component Analysis (PCA) Model Implementation

This module provides a Principal Component Analysis model implementation that follows
the standardized template structure. PCA is a dimensionality reduction technique 
that identifies the axes with maximum variance and projects the data onto a 
lower-dimensional space while preserving as much information as possible.
"""

import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.model_template import MLModel


class PCAModel(MLModel):
    """
    Principal Component Analysis model for dimensionality reduction.
    
    Use case: Dimensionality reduction for visualization, noise reduction, feature 
    extraction, and preprocessing step before applying other machine learning algorithms.
    """
    
    def __init__(self, n_components=None, svd_solver='auto', whiten=False, random_state=None, **kwargs):
        """
        Initialize the PCA model.
        
        Args:
            n_components: Number of components to keep. If None, keep all components.
                          If 0 < n_components < 1, select the number of components such that
                          the amount of variance retained is greater than the specified percentage.
            svd_solver: SVD solver to use ('auto', 'full', 'arpack', 'randomized')
            whiten: When True, the components_ vectors are multiplied by the square root
                   of n_samples and divided by the singular values to ensure uncorrelated outputs
            random_state: Random number generator seed for reproducibility
            **kwargs: Additional parameters for the sklearn PCA model
        """
        super().__init__(**kwargs)
        self.model_type = "Principal Component Analysis"
        self.model_params = {
            'n_components': n_components,
            'svd_solver': svd_solver,
            'whiten': whiten,
            'random_state': random_state,
            **kwargs
        }
        
        self.model = SklearnPCA(
            n_components=n_components,
            svd_solver=svd_solver,
            whiten=whiten,
            random_state=random_state,
            **kwargs
        )
        
        self.scaler = StandardScaler()
        self._apply_scaling = True
    
    def set_scaling(self, apply_scaling=True):
        """
        Set whether to apply scaling before PCA.
        
        Args:
            apply_scaling: Whether to apply StandardScaler before PCA
            
        Returns:
            self: The model instance
        """
        self._apply_scaling = apply_scaling
        return self
    
    def train(self, X_train, y_train=None, **kwargs):
        """
        Train the PCA model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Not used, only for compatibility with the MLModel interface
            **kwargs: Additional training parameters
            
        Returns:
            self: The trained model instance
        """
        # Store original features for later use
        self.original_feature_names = kwargs.get('feature_names', None)
        
        # Apply scaling if enabled
        if self._apply_scaling:
            X_scaled = self.scaler.fit_transform(X_train)
        else:
            X_scaled = X_train
        
        self.model.fit(X_scaled)
        self.is_trained = True
        
        # Store original data dimensions
        self.n_samples, self.n_features = X_train.shape
        
        return self
    
    def transform(self, X):
        """
        Apply dimensionality reduction to X.
        
        Args:
            X: Features to transform
            
        Returns:
            Transformed features
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        # Apply scaling if enabled
        if self._apply_scaling:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.transform(X_scaled)
    
    def inverse_transform(self, X_transformed):
        """
        Transform data back to its original space.
        
        Args:
            X_transformed: Transformed features
            
        Returns:
            X_original: Features in original space
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        X_original = self.model.inverse_transform(X_transformed)
        
        # Apply inverse scaling if enabled
        if self._apply_scaling:
            X_original = self.scaler.inverse_transform(X_original)
        
        return X_original
    
    def fit_transform(self, X, y=None):
        """
        Fit the model with X and apply dimensionality reduction on X.
        
        Args:
            X: Features to fit and transform
            y: Not used, only for compatibility
            
        Returns:
            Transformed features
        """
        self.train(X)
        return self.transform(X)
    
    def get_explained_variance_ratio(self):
        """
        Get the explained variance ratio for each component.
        
        Returns:
            Array of explained variance ratios
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        return self.model.explained_variance_ratio_
    
    def get_cumulative_explained_variance(self):
        """
        Get the cumulative explained variance ratio.
        
        Returns:
            Array of cumulative explained variance ratios
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        return np.cumsum(self.model.explained_variance_ratio_)
    
    def get_components(self):
        """
        Get the principal components.
        
        Returns:
            Array of principal components
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        return self.model.components_
    
    def get_n_components_for_variance(self, explained_variance_threshold=0.95):
        """
        Get the number of components needed to explain a certain amount of variance.
        
        Args:
            explained_variance_threshold: Target explained variance (default: 0.95)
            
        Returns:
            Number of components needed
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        cumulative_variance = self.get_cumulative_explained_variance()
        n_components = np.argmax(cumulative_variance >= explained_variance_threshold) + 1
        
        return n_components
    
    def plot_explained_variance(self):
        """
        Plot the explained variance ratio for each component.
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        # Get explained variance data
        explained_variance_ratio = self.get_explained_variance_ratio()
        cumulative_variance = self.get_cumulative_explained_variance()
        
        # Create plot
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Bar plot for individual explained variance
        ax1.bar(
            range(1, len(explained_variance_ratio) + 1),
            explained_variance_ratio,
            alpha=0.7,
            label='Individual Explained Variance'
        )
        ax1.set_xlabel('Principal Components')
        ax1.set_ylabel('Explained Variance Ratio', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Line plot for cumulative explained variance
        ax2 = ax1.twinx()
        ax2.plot(
            range(1, len(cumulative_variance) + 1),
            cumulative_variance,
            'r-',
            marker='o',
            label='Cumulative Explained Variance'
        )
        ax2.set_ylabel('Cumulative Explained Variance', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Add threshold lines at 0.8, 0.9, and 0.95
        for threshold in [0.8, 0.9, 0.95]:
            n_components = self.get_n_components_for_variance(threshold)
            ax2.axhline(y=threshold, color='green', linestyle='--', alpha=0.5)
            ax2.text(
                len(explained_variance_ratio) * 0.7,
                threshold + 0.01,
                f'{threshold:.0%}',
                color='green'
            )
            ax2.axvline(x=n_components, color='gray', linestyle='--', alpha=0.5)
            ax2.text(
                n_components + 0.1,
                0.5,
                f'{n_components} components',
                color='gray',
                rotation=90
            )
        
        # Set title and show legend
        plt.title('Explained Variance by Principal Components')
        
        # Create a combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        
        plt.tight_layout()
        plt.show()
    
    def plot_components_2d(self, X, y=None, components=(0, 1), class_names=None, feature_names=None):
        """
        Plot the first two principal components.
        
        Args:
            X: Original or already transformed data
            y: Target labels for coloring (optional)
            components: Tuple of component indices to plot (default: first two)
            class_names: Names of the classes for the legend (if y is provided)
            feature_names: Names of the original features
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        # Check if X is already transformed
        if X.shape[1] > self.model.n_components_:
            # Transform the data if it's in the original space
            X_transformed = self.transform(X)
        else:
            # Data is already transformed
            X_transformed = X
        
        # Get the components to plot
        comp1, comp2 = components
        
        if comp1 >= X_transformed.shape[1] or comp2 >= X_transformed.shape[1]:
            raise ValueError(f"Component indices {components} out of range for transformed data with shape {X_transformed.shape}")
        
        # Create the scatter plot
        plt.figure(figsize=(10, 8))
        
        if y is not None:
            # Color by class if y is provided
            scatter = plt.scatter(
                X_transformed[:, comp1],
                X_transformed[:, comp2],
                c=y,
                alpha=0.7,
                edgecolors='k',
                s=50,
                cmap='viridis'
            )
            
            # Add a legend with class names if provided
            if class_names is not None:
                # Convert numerical labels to class names
                handles, labels = scatter.legend_elements()
                unique_classes = np.unique(y)
                
                # If unique_classes are indices into class_names
                if np.issubdtype(unique_classes.dtype, np.integer) and max(unique_classes) < len(class_names):
                    legend_labels = [class_names[i] for i in unique_classes]
                else:
                    # Otherwise, use class_names directly if it has the right length
                    legend_labels = class_names if len(class_names) == len(unique_classes) else [str(c) for c in unique_classes]
                
                plt.legend(handles, legend_labels, title="Classes", loc="best")
            else:
                plt.colorbar(label='Class')
        else:
            # Simple scatter plot if no classes
            plt.scatter(
                X_transformed[:, comp1],
                X_transformed[:, comp2],
                alpha=0.7,
                edgecolors='k',
                s=50
            )
        
        # Get explained variance for the selected components
        explained_variance_ratio = self.get_explained_variance_ratio()
        variance_comp1 = explained_variance_ratio[comp1]
        variance_comp2 = explained_variance_ratio[comp2]
        
        # Set axis labels with explained variance
        plt.xlabel(f'Principal Component {comp1+1} ({variance_comp1:.1%} variance)')
        plt.ylabel(f'Principal Component {comp2+1} ({variance_comp2:.1%} variance)')
        
        plt.title('Principal Component Analysis (PCA) - 2D Projection')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_components_3d(self, X, y=None, components=(0, 1, 2), class_names=None):
        """
        Plot the first three principal components in 3D.
        
        Args:
            X: Original or already transformed data
            y: Target labels for coloring (optional)
            components: Tuple of three component indices to plot
            class_names: Names of the classes for the legend (if y is provided)
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        # Check if we have at least 3 components
        if self.model.n_components_ < 3:
            raise ValueError(f"Model has only {self.model.n_components_} components, need at least 3 for 3D plot.")
        
        # Check if X is already transformed
        if X.shape[1] > self.model.n_components_:
            # Transform the data if it's in the original space
            X_transformed = self.transform(X)
        else:
            # Data is already transformed
            X_transformed = X
        
        # Get the components to plot
        comp1, comp2, comp3 = components
        
        if max(components) >= X_transformed.shape[1]:
            raise ValueError(f"Component indices {components} out of range for transformed data with shape {X_transformed.shape}")
        
        # Create the 3D scatter plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        if y is not None:
            # Color by class if y is provided
            scatter = ax.scatter(
                X_transformed[:, comp1],
                X_transformed[:, comp2],
                X_transformed[:, comp3],
                c=y,
                alpha=0.7,
                edgecolors='k',
                s=50,
                cmap='viridis'
            )
            
            # Add a legend with class names if provided
            if class_names is not None:
                # Convert numerical labels to class names
                handles, labels = scatter.legend_elements()
                unique_classes = np.unique(y)
                
                # If unique_classes are indices into class_names
                if np.issubdtype(unique_classes.dtype, np.integer) and max(unique_classes) < len(class_names):
                    legend_labels = [class_names[i] for i in unique_classes]
                else:
                    # Otherwise, use class_names directly if it has the right length
                    legend_labels = class_names if len(class_names) == len(unique_classes) else [str(c) for c in unique_classes]
                
                ax.legend(handles, legend_labels, title="Classes", loc="best")
            else:
                fig.colorbar(scatter, ax=ax, label='Class')
        else:
            # Simple scatter plot if no classes
            ax.scatter(
                X_transformed[:, comp1],
                X_transformed[:, comp2],
                X_transformed[:, comp3],
                alpha=0.7,
                edgecolors='k',
                s=50
            )
        
        # Get explained variance for the selected components
        explained_variance_ratio = self.get_explained_variance_ratio()
        variance_comp1 = explained_variance_ratio[comp1]
        variance_comp2 = explained_variance_ratio[comp2]
        variance_comp3 = explained_variance_ratio[comp3]
        
        # Set axis labels with explained variance
        ax.set_xlabel(f'PC{comp1+1} ({variance_comp1:.1%})')
        ax.set_ylabel(f'PC{comp2+1} ({variance_comp2:.1%})')
        ax.set_zlabel(f'PC{comp3+1} ({variance_comp3:.1%})')
        
        plt.title('Principal Component Analysis (PCA) - 3D Projection')
        plt.tight_layout()
        plt.show()
    
    def plot_feature_contributions(self, n_components=2, feature_names=None):
        """
        Plot the contribution of each feature to the principal components.
        
        Args:
            n_components: Number of components to show
            feature_names: Names of the original features
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        # Get the principal components
        components = self.get_components()
        
        # Limit to the requested number of components
        n_components = min(n_components, components.shape[0])
        components = components[:n_components]
        
        # Use feature_names if provided, otherwise use the one stored during training
        # or generate generic names
        if feature_names is None:
            if self.original_feature_names is not None:
                feature_names = self.original_feature_names
            else:
                feature_names = [f"Feature {i+1}" for i in range(components.shape[1])]
        
        # Create a DataFrame for better plotting
        df = pd.DataFrame(
            components.T,
            index=feature_names,
            columns=[f"PC{i+1}" for i in range(n_components)]
        )
        
        # Create a heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            df,
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f',
            cbar_kws={'label': 'Component weight'}
        )
        plt.title('Feature Contributions to Principal Components')
        plt.tight_layout()
        plt.show()
        
        # Plot the feature contributions as a bar chart for each component
        fig, axes = plt.subplots(n_components, 1, figsize=(12, 4 * n_components), sharex=True)
        
        # Make axes iterable if there's only one
        if n_components == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            # Sort the features by absolute contribution for this component
            component_abs = np.abs(components[i])
            sorted_indices = np.argsort(component_abs)[::-1]
            
            # Plot the top features for this component
            max_features = min(10, len(feature_names))  # Show at most 10 features
            top_indices = sorted_indices[:max_features]
            
            # Create a color map based on whether contribution is positive or negative
            colors = ['red' if components[i, j] < 0 else 'blue' for j in top_indices]
            
            ax.barh(
                [feature_names[j] for j in top_indices],
                [components[i, j] for j in top_indices],
                color=colors
            )
            ax.set_title(f'Feature Contributions to PC{i+1}')
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('Contribution')
        
        plt.tight_layout()
        plt.show()
    
    def plot_biplot(self, X, feature_names=None, components=(0, 1), scale=1.0):
        """
        Create a biplot of the principal components and feature loadings.
        
        Args:
            X: Original data (not transformed)
            feature_names: Names of the original features
            components: Tuple of two component indices to plot
            scale: Scaling factor for feature vectors
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        # Transform the data
        X_transformed = self.transform(X)
        
        # Get the components to plot
        comp1, comp2 = components
        
        # Get the principal components (loadings)
        pcs = self.get_components()
        
        # Use feature_names if provided, otherwise use the ones stored during training
        # or generate generic names
        if feature_names is None:
            if self.original_feature_names is not None:
                feature_names = self.original_feature_names
            else:
                feature_names = [f"Feature {i+1}" for i in range(pcs.shape[1])]
        
        # Get explained variance
        explained_variance_ratio = self.get_explained_variance_ratio()
        
        # Create the biplot
        plt.figure(figsize=(12, 10))
        
        # Plot the transformed data points
        plt.scatter(
            X_transformed[:, comp1],
            X_transformed[:, comp2],
            alpha=0.2,
            edgecolors='k'
        )
        
        # Plot the feature vectors
        for i, feature in enumerate(feature_names):
            plt.arrow(
                0, 0,  # Start at origin
                pcs[comp1, i] * scale,  # x component
                pcs[comp2, i] * scale,  # y component
                color='r',
                width=0.002,
                head_width=0.05
            )
            
            # Add feature names at the end of the vectors
            plt.text(
                pcs[comp1, i] * scale * 1.1,
                pcs[comp2, i] * scale * 1.1,
                feature,
                color='r',
                ha='center',
                va='center'
            )
        
        # Set axis labels with explained variance
        plt.xlabel(f'Principal Component {comp1+1} ({explained_variance_ratio[comp1]:.1%} variance)')
        plt.ylabel(f'Principal Component {comp2+1} ({explained_variance_ratio[comp2]:.1%} variance)')
        
        # Add grid and equal aspect ratio
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        plt.axis('equal')
        
        plt.title('PCA Biplot: Feature Contributions and Transformed Data')
        plt.tight_layout()
        plt.show()
    
    def plot_reconstruction_error(self, X, max_components=None, n_points=20):
        """
        Plot the reconstruction error for different numbers of components.
        
        Args:
            X: Original data
            max_components: Maximum number of components to consider
            n_points: Number of points to evaluate
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        # Apply scaling if enabled
        if self._apply_scaling:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Determine the maximum number of components to consider
        if max_components is None:
            max_components = min(X.shape[1], self.model.n_components_)
        else:
            max_components = min(max_components, X.shape[1], self.model.n_components_)
        
        # Generate a range of component numbers to evaluate
        if n_points <= max_components:
            components_range = np.linspace(1, max_components, n_points, dtype=int)
        else:
            components_range = np.arange(1, max_components + 1)
        
        # Calculate reconstruction error for each number of components
        errors = []
        
        for n_comp in components_range:
            # Take only the first n_comp components
            components = self.model.components_[:n_comp]
            
            # Project data onto the components
            X_projected = X_scaled.dot(components.T)
            
            # Reconstruct data from the projection
            X_reconstructed = X_projected.dot(components)
            
            # Calculate mean squared error
            mse = np.mean(np.sum((X_scaled - X_reconstructed) ** 2, axis=1))
            errors.append(mse)
        
        # Plot the reconstruction error
        plt.figure(figsize=(10, 6))
        plt.plot(components_range, errors, 'o-', markersize=8)
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Reconstruction Error (MSE)')
        plt.title('Reconstruction Error vs. Number of Components')
        plt.grid(True)
        
        # Find elbow point
        from kneed import KneeLocator
        try:
            kneedle = KneeLocator(components_range, errors, curve='convex', direction='decreasing')
            elbow = kneedle.elbow
            
            if elbow:
                plt.axvline(x=elbow, color='r', linestyle='--', alpha=0.7)
                plt.text(elbow + 0.2, min(errors) + (max(errors) - min(errors)) * 0.5, 
                        f'Elbow point: {elbow} components', color='r')
        except:
            # Skip if kneed package is not available or fails
            pass
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        # Save both the PCA model and the scaler
        joblib.dump({
            'pca_model': self.model,
            'scaler': self.scaler,
            'apply_scaling': self._apply_scaling,
            'original_feature_names': self.original_feature_names
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            self: The loaded model instance
        """
        saved_data = joblib.load(filepath)
        
        self.model = saved_data['pca_model']
        self.scaler = saved_data['scaler']
        self._apply_scaling = saved_data['apply_scaling']
        self.original_feature_names = saved_data.get('original_feature_names', None)
        
        self.is_trained = True
        print(f"Model loaded from {filepath}")
        return self 