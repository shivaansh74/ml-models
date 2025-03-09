"""
Random Forest Model Implementation

This module provides a Random Forest model implementation that follows
the standardized template structure. Random Forests are ensemble learning methods
that operate by constructing multiple decision trees and outputting the mean prediction
(regression) or mode prediction (classification) of the individual trees.
"""

import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.model_template import MLModel


class RandomForestModel(MLModel):
    """
    Random Forest model for classification and regression tasks.
    
    Use cases:
    - Classification: Customer churn prediction, fraud detection
    - Regression: Sales forecasting, house price prediction
    """
    
    def __init__(self, task="classification", n_estimators=100, max_depth=None, 
                 min_samples_split=2, min_samples_leaf=1, max_features="auto", 
                 bootstrap=True, random_state=None, n_jobs=-1, **kwargs):
        """
        Initialize the Random Forest model.
        
        Args:
            task: Type of task, either "classification" or "regression"
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            min_samples_split: Minimum number of samples required to split an internal node
            min_samples_leaf: Minimum number of samples required to be at a leaf node
            max_features: Number of features to consider for the best split
            bootstrap: Whether bootstrap samples are used when building trees
            random_state: Seed for random number generation
            n_jobs: Number of jobs to run in parallel (-1 means using all processors)
            **kwargs: Additional parameters for the sklearn RandomForest models
        """
        super().__init__(**kwargs)
        self.model_type = "Random Forest"
        self.task = task
        self.model_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'bootstrap': bootstrap,
            'random_state': random_state,
            'n_jobs': n_jobs,
            **kwargs
        }
        
        if task == "classification":
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                bootstrap=bootstrap,
                random_state=random_state,
                n_jobs=n_jobs,
                **kwargs
            )
        elif task == "regression":
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                bootstrap=bootstrap,
                random_state=random_state,
                n_jobs=n_jobs,
                **kwargs
            )
        else:
            raise ValueError("Task must be either 'classification' or 'regression'")
        
        self.fitted = False

    def train(self, X_train, y_train, **kwargs):
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training target values
            **kwargs: Additional parameters for fit method
            
        Returns:
            self: The trained model instance
        """
        self.model.fit(X_train, y_train, **kwargs)
        self.fitted = True
        self._log_training_complete()
        return self

    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted values
        """
        if not self.fitted:
            raise ValueError("Model has not been trained yet. Call 'train' first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        Only available for classification tasks.
        
        Args:
            X: Features to predict on
            
        Returns:
            Class probabilities
        """
        if not self.fitted:
            raise ValueError("Model has not been trained yet. Call 'train' first.")
        
        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification tasks")
        
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test, task_type=None):
        """
        Evaluate the model's performance.
        
        Args:
            X_test: Test features
            y_test: Test target values
            task_type: Type of task (classification/regression)
                       If None, uses self.task
        
        Returns:
            Dictionary of evaluation metrics
        """
        if task_type is None:
            task_type = self.task
            
        return super().evaluate(X_test, y_test, task_type)

    def get_feature_importance(self, feature_names=None):
        """
        Get feature importances from the trained model.
        
        Args:
            feature_names: List of feature names. If None, indices will be used.
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.fitted:
            raise ValueError("Model has not been trained yet. Call 'train' first.")
        
        importances = self.model.feature_importances_
        
        # Use feature names if provided, otherwise use indices
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        # Create dictionary of feature importances
        feature_importance_dict = dict(zip(feature_names, importances))
        
        # Sort by importance
        return dict(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True))

    def plot_feature_importance(self, feature_names=None, top_n=10):
        """
        Plot feature importances.
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to plot
            
        Returns:
            Matplotlib figure
        """
        if not self.fitted:
            raise ValueError("Model has not been trained yet. Call 'train' first.")
        
        importance_dict = self.get_feature_importance(feature_names)
        
        # Limit to top_n features
        importance_dict = dict(list(importance_dict.items())[:top_n])
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.barh(list(importance_dict.keys()), list(importance_dict.values()))
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importances - {self.model_type}')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_tree(self, tree_index=0, feature_names=None, class_names=None, filled=True, max_depth=3):
        """
        Visualize a specific tree from the random forest.
        
        Args:
            tree_index: Index of the tree to visualize
            feature_names: Names of the features
            class_names: Names of the classes (for classification)
            filled: Whether to fill nodes with colors
            max_depth: Maximum depth of the tree to plot
            
        Returns:
            Matplotlib figure
        """
        if not self.fitted:
            raise ValueError("Model has not been trained yet. Call 'train' first.")
        
        from sklearn.tree import plot_tree
        
        if tree_index >= self.model.n_estimators:
            raise ValueError(f"Tree index must be less than the number of estimators ({self.model.n_estimators})")
        
        plt.figure(figsize=(20, 10))
        plot_tree(self.model.estimators_[tree_index], 
                  max_depth=max_depth, 
                  feature_names=feature_names,
                  class_names=class_names, 
                  filled=filled, 
                  rounded=True, 
                  proportion=True)
        plt.title(f'Random Forest - Tree #{tree_index} Visualization')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_oob_error(self, X, y, max_estimators=100, step=10):
        """
        Plot the Out-of-Bag (OOB) error as a function of the number of estimators.
        Only available if bootstrap=True was used during initialization.
        
        Args:
            X: Features
            y: Target values
            max_estimators: Maximum number of estimators to evaluate
            step: Step size for the number of estimators
            
        Returns:
            Matplotlib figure
        """
        if not hasattr(self.model, 'oob_score_') or not self.model.bootstrap:
            raise ValueError("OOB error can only be plotted if bootstrap=True and oob_score=True")
        
        # Create a range of estimator values
        estimator_range = range(1, min(max_estimators, self.model.n_estimators) + 1, step)
        oob_scores = []
        
        # For each estimator count, fit a model and get OOB score
        for n_estimators in estimator_range:
            if self.task == "classification":
                rf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    oob_score=True,
                    **{k: v for k, v in self.model_params.items() if k != 'n_estimators' and k != 'n_jobs'}
                )
            else:
                rf = RandomForestRegressor(
                    n_estimators=n_estimators,
                    oob_score=True,
                    **{k: v for k, v in self.model_params.items() if k != 'n_estimators' and k != 'n_jobs'}
                )
            
            rf.fit(X, y)
            oob_scores.append(rf.oob_score_)
        
        # Plot the OOB error
        plt.figure(figsize=(10, 6))
        plt.plot(list(estimator_range), oob_scores, 'b-', marker='o')
        plt.xlabel('Number of Estimators')
        plt.ylabel('OOB Score')
        plt.title('Random Forest - OOB Score vs. Number of Estimators')
        plt.grid(True)
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