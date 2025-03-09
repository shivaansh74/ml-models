"""
Logistic Regression Model Implementation

This module provides a Logistic Regression model implementation that follows
the standardized template structure. Logistic Regression is used for binary
classification tasks, such as spam email classification.
"""

import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.model_template import MLModel


class LogisticRegressionModel(MLModel):
    """
    Logistic Regression model for binary classification tasks.
    
    Use case: Spam email classification, fraud detection, medical diagnosis, etc.
    """
    
    def __init__(self, penalty='l2', C=1.0, solver='lbfgs', max_iter=100, **kwargs):
        """
        Initialize the Logistic Regression model.
        
        Args:
            penalty: Regularization penalty ('l1', 'l2', 'elasticnet', 'none')
            C: Inverse of regularization strength
            solver: Algorithm to use ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')
            max_iter: Maximum number of iterations
            **kwargs: Additional parameters for the sklearn LogisticRegression model
        """
        super().__init__(**kwargs)
        self.model_type = "Logistic Regression"
        self.model_params = {
            'penalty': penalty,
            'C': C,
            'solver': solver,
            'max_iter': max_iter,
            **kwargs
        }
        
        self.model = SklearnLogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            max_iter=max_iter,
            **kwargs
        )
    
    def train(self, X_train, y_train, **kwargs):
        """
        Train the Logistic Regression model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional training parameters
            
        Returns:
            self: The trained model instance
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Store model coefficients and intercept for interpretation
        self.coefficients = self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_
        self.intercept = self.model.intercept_[0] if len(self.model.intercept_.shape) > 0 else self.model.intercept_
        
        # Store classes
        self.classes = self.model.classes_
        
        return self
    
    def predict(self, X):
        """
        Make class predictions using the trained model.
        
        Args:
            X: Features to make predictions on
            
        Returns:
            Class predictions from the model
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Make probability predictions using the trained model.
        
        Args:
            X: Features to make predictions on
            
        Returns:
            Probability predictions from the model
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test, task_type="classification"):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: True labels for test data
            task_type: Must be "classification" for Logistic Regression
            
        Returns:
            Dictionary of evaluation metrics
        """
        if task_type != "classification":
            raise ValueError("Logistic Regression is a classification model. Task type must be 'classification'.")
        
        return super().evaluate(X_test, y_test, task_type="classification")
    
    def get_feature_importance(self, feature_names=None):
        """
        Get the feature importance (coefficients) from the model.
        
        Args:
            feature_names: Names of the features (optional)
            
        Returns:
            DataFrame with feature names and their coefficients
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        import pandas as pd
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.coefficients))]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': self.coefficients
        })
        
        # Sort by absolute coefficient value for better visualization
        importance_df['Abs_Coefficient'] = importance_df['Coefficient'].abs()
        importance_df = importance_df.sort_values('Abs_Coefficient', ascending=False)
        importance_df = importance_df.drop(columns=['Abs_Coefficient'])
        
        return importance_df
    
    def plot_feature_importance(self, feature_names=None, top_n=10):
        """
        Plot the feature importance (coefficients) from the model.
        
        Args:
            feature_names: Names of the features (optional)
            top_n: Number of top features to display
            
        Returns:
            matplotlib plot of feature importance
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        importance_df = self.get_feature_importance(feature_names)
        
        # Take top N features
        importance_df = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 6))
        colors = ['green' if c > 0 else 'red' for c in importance_df['Coefficient']]
        plt.barh(importance_df['Feature'], importance_df['Coefficient'], color=colors)
        plt.xlabel('Coefficient (Green = Positive Impact, Red = Negative Impact)')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Coefficients')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, X, y_true):
        """
        Plot the ROC curve and calculate AUC.
        
        Args:
            X: Features
            y_true: True labels
            
        Returns:
            matplotlib plot of ROC curve
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        # Get probability predictions for positive class
        y_prob = self.predict_proba(X)[:, 1]
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()
        
        return roc_auc
    
    def plot_confusion_matrix(self, X, y_true, normalize=False):
        """
        Plot the confusion matrix for the model predictions.
        
        Args:
            X: Features
            y_true: True labels
            normalize: Whether to normalize the confusion matrix
            
        Returns:
            matplotlib plot of confusion matrix
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        y_pred = self.predict(X)
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Define class labels
        if hasattr(self, 'classes'):
            class_labels = self.classes
        else:
            class_labels = ['Class 0', 'Class 1']
        
        tick_marks = np.arange(len(class_labels))
        plt.xticks(tick_marks, class_labels, rotation=45)
        plt.yticks(tick_marks, class_labels)
        
        # Add text annotations
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
    
    def plot_results(self, X_test, y_test, task_type="classification"):
        """
        Plot the results of the model.
        
        Args:
            X_test: Test features
            y_test: True labels for test data
            task_type: Must be "classification" for Logistic Regression
        """
        if task_type != "classification":
            raise ValueError("Logistic Regression is a classification model. Task type must be 'classification'.")
        
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(X_test, y_test)
        
        # Plot ROC curve
        self.plot_roc_curve(X_test, y_test)
    
    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath):
        """
        Load a trained model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            self: The loaded model instance
        """
        self.model = joblib.load(filepath)
        self.is_trained = True
        
        # Update model attributes
        self.coefficients = self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_
        self.intercept = self.model.intercept_[0] if len(self.model.intercept_.shape) > 0 else self.model.intercept_
        self.classes = self.model.classes_
        
        return self


# Example usage
if __name__ == "__main__":
    # Sample data generation for binary classification
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate binary target with noise
    # Linear combination of features with some weights
    w = np.array([1.5, -2.0, 0.5, -1.0, 2.5])
    z = np.dot(X, w) + np.random.randn(n_samples) * 0.5
    y = (z > 0).astype(int)  # Binary outcome
    
    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature names
    feature_names = [f'Feature_{i+1}' for i in range(n_features)]
    
    # Create and train the model
    model = LogisticRegressionModel(C=1.0, max_iter=1000)
    model.train(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    print("Performance metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Feature importance
    print("\nFeature Importance:")
    importance_df = model.get_feature_importance(feature_names)
    print(importance_df)
    
    # Plot results
    model.plot_results(X_test, y_test)
    
    # Plot feature importance
    model.plot_feature_importance(feature_names) 