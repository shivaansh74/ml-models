"""
Naïve Bayes Model Implementation

This module provides a Naïve Bayes model implementation that follows
the standardized template structure. Naïve Bayes is a probabilistic classifier
commonly used for text classification tasks such as spam filtering, sentiment analysis,
and document categorization.
"""

import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.model_template import MLModel


class NaiveBayesModel(MLModel):
    """
    Naïve Bayes model for classification tasks.
    
    Use case: Text classification tasks such as spam detection, sentiment analysis,
    and document categorization.
    """
    
    def __init__(self, variant='multinomial', alpha=1.0, fit_prior=True, class_prior=None, **kwargs):
        """
        Initialize the Naïve Bayes model.
        
        Args:
            variant: Type of Naive Bayes variant to use ('multinomial', 'gaussian', or 'bernoulli')
            alpha: Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing)
            fit_prior: Whether to learn class prior probabilities or use a uniform prior
            class_prior: Prior probabilities of the classes. If specified, the priors are not adjusted according to the data
            **kwargs: Additional parameters for the sklearn Naive Bayes model
        """
        super().__init__(**kwargs)
        self.model_type = f"{variant.capitalize()} Naïve Bayes"
        self.variant = variant
        self.model_params = {
            'variant': variant,
            'alpha': alpha,
            'fit_prior': fit_prior,
            'class_prior': class_prior,
            **kwargs
        }
        
        # Initialize the appropriate Naive Bayes variant
        if variant.lower() == 'multinomial':
            self.model = MultinomialNB(
                alpha=alpha,
                fit_prior=fit_prior,
                class_prior=class_prior
            )
        elif variant.lower() == 'gaussian':
            self.model = GaussianNB(
                priors=class_prior,
                **kwargs
            )
        elif variant.lower() == 'bernoulli':
            self.model = BernoulliNB(
                alpha=alpha,
                fit_prior=fit_prior,
                class_prior=class_prior,
                **kwargs
            )
        else:
            raise ValueError("variant must be one of: 'multinomial', 'gaussian', or 'bernoulli'")
    
    def train(self, X_train, y_train, **kwargs):
        """
        Train the Naïve Bayes model on the provided data.
        
        Args:
            X_train: Training features (document-term matrix for text)
            y_train: Training labels
            **kwargs: Additional training parameters
            
        Returns:
            self: The trained model instance
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Features to make predictions on (document-term matrix for text)
            
        Returns:
            Predictions from the model
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for the input samples.
        
        Args:
            X: Features to make predictions on
            
        Returns:
            Class probabilities for each sample
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
            task_type: Should be "classification" for Naive Bayes
            
        Returns:
            Dictionary of evaluation metrics
        """
        if task_type != "classification":
            raise ValueError("Naive Bayes is a classification model. task_type must be 'classification'")
            
        return super().evaluate(X_test, y_test, task_type="classification")
    
    def plot_confusion_matrix(self, X_test, y_test, class_names=None):
        """
        Plot the confusion matrix for the test data.
        
        Args:
            X_test: Test features
            y_test: True labels for test data
            class_names: Names of the classes for the axis labels
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names if class_names else "auto",
                   yticklabels=class_names if class_names else "auto")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{self.model_type} - Confusion Matrix')
        plt.show()
        
    def plot_class_probabilities(self, X, top_n=5):
        """
        Plot the class probabilities for the given samples.
        
        Args:
            X: Features to make predictions on
            top_n: Number of top samples to plot
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        # Get class probabilities
        probs = self.predict_proba(X)
        
        # Plot for a subset of samples
        n_samples = min(top_n, X.shape[0])
        
        plt.figure(figsize=(12, 8))
        plt.bar(np.arange(len(self.model.classes_)), probs[0])
        plt.xticks(np.arange(len(self.model.classes_)), self.model.classes_)
        plt.xlabel('Classes')
        plt.ylabel('Probability')
        plt.title(f'{self.model_type} - Class Probabilities for Sample 1')
        plt.show()
    
    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
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
        print(f"Model loaded from {filepath}")
        return self 