"""
Naïve Bayes Example: Spam Detection

This example demonstrates how to use the Naïve Bayes model for text classification,
specifically for spam detection. The example uses a dataset of emails
labeled as spam or not spam (ham) and shows how to:

1. Preprocess the text data using TF-IDF vectorization
2. Train a Multinomial Naïve Bayes model
3. Evaluate the model's performance
4. Visualize the results with a confusion matrix
5. Analyze feature importance to understand what words are most indicative of spam
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from models.supervised.naive_bayes import NaiveBayesModel


def load_sample_data():
    """
    Load sample spam data or create synthetic data if not available.
    
    Returns:
        DataFrame with 'text' and 'label' columns
    """
    try:
        # Try to load the SMS Spam Collection Dataset
        # This dataset is available at: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
        # If you don't have the dataset, this will generate synthetic data instead
        data_path = "../data/spam.csv"
        df = pd.read_csv(data_path, encoding='latin-1')
        df = df.rename(columns={'v1': 'label', 'v2': 'text'})
        df = df[['text', 'label']]
        print(f"Loaded real spam dataset with {len(df)} examples.")
        return df
    except:
        print("Real dataset not found. Creating synthetic spam data...")
        
        # Create synthetic data for demonstration purposes
        np.random.seed(42)
        spam_texts = [
            "URGENT: You have won a free prize. Call now to claim!",
            "Congratulations! You've been selected for a free iPhone. Click here!",
            "URGENT: Your account will be suspended. Verify your information now.",
            "You have won $5000! Send your details to claim the prize.",
            "Free entry to win a new car. Reply with your details.",
            "Buy cheap medications online! Discount prices!",
            "Get rich quick! Invest now and double your money.",
            "Lose weight fast with our new miracle pill. Order now!",
            "ALERT: Your payment is due. Click here to avoid penalties.",
            "Limited time offer: 90% off designer watches. Buy now!"
        ]
        
        ham_texts = [
            "Hi, can we meet tomorrow at 2pm?",
            "Don't forget to bring the documents for the meeting.",
            "I'll be home around 6. Do you need anything from the store?",
            "The report is ready. I'll email it to you soon.",
            "Happy birthday! Hope you have a great day.",
            "The weather forecast says it might rain tomorrow.",
            "Can you please call me when you have a moment?",
            "I finished the project. Let's discuss it next week.",
            "Thanks for your help with the presentation yesterday.",
            "I'm running late. Will be there in 15 minutes."
        ]
        
        # Create more synthetic examples
        more_spam = [
            f"URGENT offer: 50% off on {item}. Limited time!" for item in 
            ["watches", "phones", "laptops", "vacations", "cars"]
        ]
        more_ham = [
            f"Can we discuss the {topic} project tomorrow?" for topic in 
            ["marketing", "sales", "development", "design", "research"]
        ]
        
        spam_texts.extend(more_spam)
        ham_texts.extend(more_ham)
        
        # Create a dataframe
        texts = spam_texts + ham_texts
        labels = ["spam"] * len(spam_texts) + ["ham"] * len(ham_texts)
        
        df = pd.DataFrame({
            'text': texts,
            'label': labels
        })
        
        print(f"Created synthetic dataset with {len(df)} examples.")
        return df


def main():
    print("=" * 80)
    print("Naïve Bayes Example: Spam Detection")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading and exploring the dataset...")
    df = load_sample_data()
    
    # Display data statistics
    print("\nDataset preview:")
    print(df.head())
    
    print("\nClass distribution:")
    print(df['label'].value_counts())
    
    # Convert labels to binary format (1 for spam, 0 for ham)
    df['label_binary'] = df['label'].apply(lambda x: 1 if x.lower() == 'spam' else 0)
    
    # Split the data
    print("\n2. Splitting data into training and testing sets...")
    X = df['text']
    y = df['label_binary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # Create a TF-IDF vectorizer to convert text to numerical features
    print("\n3. Creating TF-IDF features from text...")
    tfidf = TfidfVectorizer(
        max_features=1000,  # Use top 1000 words only
        min_df=2,           # Ignore terms that appear in less than 2 documents
        max_df=0.8,         # Ignore terms that appear in more than 80% of documents
        stop_words='english' # Remove common English stop words
    )
    
    # Transform the text data
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    print(f"Vocabulary size: {len(tfidf.vocabulary_)}")
    print(f"Feature matrix shape: {X_train_tfidf.shape}")
    
    # Create and train the Naïve Bayes model
    print("\n4. Training the Multinomial Naïve Bayes model...")
    nb_model = NaiveBayesModel(variant='multinomial', alpha=1.0)
    nb_model.train(X_train_tfidf, y_train)
    print("Model training completed.")
    
    # Evaluate the model
    print("\n5. Evaluating the model...")
    y_pred = nb_model.predict(X_test_tfidf)
    
    # Display evaluation metrics
    metrics = nb_model.evaluate(X_test_tfidf, y_test)
    print("\nClassification Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Display classification report
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])
    print(report)
    
    # Confusion Matrix Visualization
    print("\n6. Visualizing confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Ham', 'Spam'],
               yticklabels=['Ham', 'Spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for Spam Detection')
    plt.show()
    
    # Feature Importance: Analyze the most informative features (words)
    print("\n7. Analyzing feature importance (most informative words)...")
    
    # Get feature names
    feature_names = tfidf.get_feature_names_out()
    
    # Get coefficients from the Naive Bayes model
    # For Multinomial NB, the feature log probability is stored in feature_log_prob_
    feature_importance = nb_model.model.feature_log_prob_[1] - nb_model.model.feature_log_prob_[0]
    
    # Sort the coefficients
    sorted_indices = feature_importance.argsort()
    
    # Print the most spammy words (most indicative of spam)
    n_top_features = 20
    print("\nTop spam-indicating words:")
    for idx in sorted_indices[-n_top_features:]:
        print(f"{feature_names[idx]}: {feature_importance[idx]:.4f}")
    
    # Print the most ham-like words (least indicative of spam)
    print("\nTop ham-indicating words:")
    for idx in sorted_indices[:n_top_features]:
        print(f"{feature_names[idx]}: {feature_importance[idx]:.4f}")
    
    # Visualize the most important features
    plt.figure(figsize=(12, 8))
    
    # Plot top spam-indicating words
    plt.subplot(1, 2, 1)
    top_spam_words = [feature_names[idx] for idx in sorted_indices[-10:]]
    top_spam_values = [feature_importance[idx] for idx in sorted_indices[-10:]]
    plt.barh(range(10), top_spam_values, color='red')
    plt.yticks(range(10), top_spam_words)
    plt.xlabel('Log Probability Ratio')
    plt.title('Top Spam-Indicating Words')
    
    # Plot top ham-indicating words
    plt.subplot(1, 2, 2)
    top_ham_words = [feature_names[idx] for idx in sorted_indices[:10]]
    top_ham_values = [feature_importance[idx] for idx in sorted_indices[:10]]
    plt.barh(range(10), top_ham_values, color='green')
    plt.yticks(range(10), top_ham_words)
    plt.xlabel('Log Probability Ratio')
    plt.title('Top Ham-Indicating Words')
    
    plt.tight_layout()
    plt.show()
    
    # Try with new examples
    print("\n8. Making predictions on new examples...")
    new_texts = [
        "URGENT: You've won a $1000 Walmart gift card. Call now to claim your prize!",
        "Meeting tomorrow at 10 AM to discuss the project timeline.",
        "Get 80% off on luxury watches. Limited time offer!",
        "Can you pick up some groceries on your way home?"
    ]
    
    # Transform the new texts
    new_texts_tfidf = tfidf.transform(new_texts)
    
    # Predict and get probabilities
    predictions = nb_model.predict(new_texts_tfidf)
    probabilities = nb_model.predict_proba(new_texts_tfidf)
    
    # Display results
    print("\nPredictions for new messages:")
    for i, text in enumerate(new_texts):
        spam_prob = probabilities[i][1]
        print(f"Text: {text}")
        print(f"Prediction: {'Spam' if predictions[i] == 1 else 'Ham'}")
        print(f"Spam probability: {spam_prob:.4f}")
        print("---")
    
    # Save the model
    print("\n9. Saving the model...")
    model_path = "../models/naive_bayes_spam_model.joblib"
    nb_model.save_model(model_path)
    
    print("\nExample complete! The Naïve Bayes model has been trained for spam detection.")


if __name__ == "__main__":
    main() 