"""
Logistic Regression Example

This script demonstrates how to use the LogisticRegressionModel class
for spam email classification.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Add the parent directory to the path to import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.data_utils import load_data, split_data, preprocess_features
from models.supervised.logistic_regression import LogisticRegressionModel


def create_synthetic_spam_data(n_samples=1000, random_state=42):
    """Create a synthetic spam classification dataset."""
    # Create synthetic email features and labels
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=3,
        weights=[0.7, 0.3],  # 70% ham, 30% spam
        random_state=random_state
    )
    
    # Create DataFrame with feature names
    feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['is_spam'] = y
    
    return df


def run_synthetic_spam_example():
    """Run a logistic regression example with synthetic spam data."""
    print("\n" + "="*50)
    print("LOGISTIC REGRESSION WITH SYNTHETIC SPAM DATA")
    print("="*50)
    
    # Create synthetic data
    df = create_synthetic_spam_data(n_samples=2000)
    
    print("Synthetic spam dataset created:")
    print(df.head())
    
    # Class distribution
    spam_count = df['is_spam'].sum()
    ham_count = len(df) - spam_count
    print(f"\nClass distribution: {ham_count} ham emails (not spam), {spam_count} spam emails")
    
    # Split data
    X = df.drop('is_spam', axis=1)
    y = df['is_spam']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the model
    print("\nTraining Logistic Regression model...")
    model = LogisticRegressionModel(C=1.0, max_iter=1000)
    model.train(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate
    metrics = model.evaluate(X_test_scaled, y_test)
    print("\nPerformance metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Feature importance
    print("\nFeature Importance (top 10):")
    importance_df = model.get_feature_importance(X.columns)
    print(importance_df.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    colors = ['green' if c > 0 else 'red' for c in importance_df.head(10)['Coefficient']]
    plt.barh(importance_df.head(10)['Feature'], importance_df.head(10)['Coefficient'], color=colors)
    plt.xlabel('Coefficient (Green = Positive Impact, Red = Negative Impact)')
    plt.ylabel('Feature')
    plt.title('Top 10 Features for Spam Classification')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.savefig('examples/spam_feature_importance.png')
    print("\nSaved feature importance plot to 'examples/spam_feature_importance.png'")
    
    # Plot confusion matrix
    y_pred = model.predict(X_test_scaled)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    class_labels = ['Ham', 'Spam']
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('examples/spam_confusion_matrix.png')
    print("Saved confusion matrix plot to 'examples/spam_confusion_matrix.png'")
    
    # Plot ROC curve
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
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
    plt.savefig('examples/spam_roc_curve.png')
    print("Saved ROC curve plot to 'examples/spam_roc_curve.png'")
    
    return model


def generate_text_data():
    """Generate synthetic email text data for demonstration."""
    # Common words in spam and ham emails
    spam_phrases = [
        "buy now", "free offer", "limited time", "act now", "discount",
        "special promotion", "click here", "exclusive deal", "money back",
        "guarantee", "best price", "free gift", "winner", "risk-free"
    ]
    
    ham_phrases = [
        "meeting tomorrow", "project update", "please review", "attached file",
        "team discussion", "weekly report", "schedule change", "budget review",
        "client feedback", "conference call", "deadline reminder"
    ]
    
    # Generate 1000 emails (70% ham, 30% spam)
    n_samples = 1000
    n_ham = int(0.7 * n_samples)
    n_spam = n_samples - n_ham
    
    emails = []
    labels = []
    
    # Generate ham emails
    for i in range(n_ham):
        # Pick 2-4 random ham phrases
        n_phrases = np.random.randint(2, 5)
        phrases = np.random.choice(ham_phrases, n_phrases, replace=False)
        
        # Add some random text
        email = ' '.join(phrases)
        email += " The meeting is at " + str(np.random.randint(8, 18)) + ":00."
        email += " Please let me know if you have any questions."
        
        emails.append(email)
        labels.append(0)  # 0 = ham
    
    # Generate spam emails
    for i in range(n_spam):
        # Pick 2-4 random spam phrases
        n_phrases = np.random.randint(2, 5)
        phrases = np.random.choice(spam_phrases, n_phrases, replace=False)
        
        # Add some random text
        email = ' '.join(phrases)
        email += " Save $" + str(np.random.randint(50, 500)) + " today!"
        email += " This offer expires soon."
        
        emails.append(email)
        labels.append(1)  # 1 = spam
    
    # Shuffle the data
    indices = np.random.permutation(n_samples)
    emails = [emails[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    return pd.DataFrame({
        'email_text': emails,
        'is_spam': labels
    })


def run_text_spam_example():
    """Run a logistic regression example with text-based spam data."""
    print("\n" + "="*50)
    print("LOGISTIC REGRESSION WITH TEXT-BASED SPAM DATA")
    print("="*50)
    
    # Generate text data
    df = generate_text_data()
    
    print("Text-based spam dataset created:")
    print(df.head())
    
    # Class distribution
    spam_count = df['is_spam'].sum()
    ham_count = len(df) - spam_count
    print(f"\nClass distribution: {ham_count} ham emails (not spam), {spam_count} spam emails")
    
    # Split data
    X_text = df['email_text']
    y = df['is_spam']
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Use TF-IDF to convert text to features
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    
    # Get feature names for later
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"\nText converted to {X_train.shape[1]} TF-IDF features")
    
    # Create and train the model
    print("\nTraining Logistic Regression model...")
    model = LogisticRegressionModel(C=1.0, max_iter=1000)
    model.train(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    print("\nPerformance metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Feature importance
    print("\nTop spam-indicating words:")
    importance_df = model.get_feature_importance(feature_names)
    # Display top 10 positive coefficients (spam indicators)
    spam_indicators = importance_df.sort_values('Coefficient', ascending=False).head(10)
    print(spam_indicators)
    
    print("\nTop ham-indicating words:")
    # Display top 10 negative coefficients (ham indicators)
    ham_indicators = importance_df.sort_values('Coefficient', ascending=True).head(10)
    print(ham_indicators)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    
    # Subplot for spam indicators
    plt.subplot(1, 2, 1)
    plt.barh(spam_indicators['Feature'], spam_indicators['Coefficient'], color='red')
    plt.xlabel('Coefficient')
    plt.ylabel('Word')
    plt.title('Top 10 Spam-Indicating Words')
    
    # Subplot for ham indicators
    plt.subplot(1, 2, 2)
    plt.barh(ham_indicators['Feature'], ham_indicators['Coefficient'], color='green')
    plt.xlabel('Coefficient')
    plt.ylabel('Word')
    plt.title('Top 10 Ham-Indicating Words')
    
    plt.tight_layout()
    plt.savefig('examples/text_spam_feature_importance.png')
    print("\nSaved text feature importance plot to 'examples/text_spam_feature_importance.png'")
    
    # Plot ROC curve
    y_prob = model.predict_proba(X_test)[:, 1]
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Text Spam Classification')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('examples/text_spam_roc_curve.png')
    print("Saved ROC curve plot to 'examples/text_spam_roc_curve.png'")
    
    return model


if __name__ == "__main__":
    # Run both examples
    synthetic_model = run_synthetic_spam_example()
    text_model = run_text_spam_example()
    
    print("\n" + "="*50)
    print("Examples completed! Check the generated plots.")
    print("="*50) 