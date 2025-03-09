"""
Natural Language Processing (NLP) Models Implementation

This module provides NLP model implementations that follow the standardized
template structure. It includes models for chatbots and text summarization
using various techniques from rule-based approaches to transformer-based models.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import os
import re
import string
import random
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

# Import for transformer-based models
try:
    import torch
    from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.model_template import MLModel


class NLPBaseModel(MLModel):
    """
    Base class for NLP models with common utility functions.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the NLP base model.
        
        Args:
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.model_type = "NLP Base Model"
        self.nlp_task = "base"
        self.model_params = kwargs
        
        # Download NLTK resources if not already available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
    
    def preprocess_text(self, text, lowercase=True, remove_punctuation=True, 
                       remove_stopwords=True, lemmatize=False, stem=False):
        """
        Preprocess text with common NLP techniques.
        
        Args:
            text: Input text to preprocess
            lowercase: Whether to convert to lowercase
            remove_punctuation: Whether to remove punctuation
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize words
            stem: Whether to stem words
            
        Returns:
            Preprocessed text
        """
        if lowercase:
            text = text.lower()
            
        if remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
            
        # Tokenize
        tokens = word_tokenize(text)
        
        if remove_stopwords:
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]
            
        if lemmatize:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
            
        if stem:
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(token) for token in tokens]
            
        return ' '.join(tokens)
    
    def tokenize_sentences(self, text):
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        return sent_tokenize(text)
    
    def get_word_frequency(self, text):
        """
        Get word frequency distribution.
        
        Args:
            text: Input text
            
        Returns:
            Counter object with word frequencies
        """
        tokens = word_tokenize(self.preprocess_text(text))
        return Counter(tokens)
    
    def plot_word_frequency(self, text, top_n=20):
        """
        Plot the top N most frequent words.
        
        Args:
            text: Input text
            top_n: Number of top words to show
            
        Returns:
            Matplotlib figure
        """
        word_freq = self.get_word_frequency(text)
        top_words = dict(word_freq.most_common(top_n))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=list(top_words.keys()), y=list(top_words.values()), ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title(f'Top {top_n} Most Frequent Words')
        ax.set_xlabel('Words')
        ax.set_ylabel('Frequency')
        plt.tight_layout()
        
        return fig


class ChatbotModel(NLPBaseModel):
    """
    Chatbot model implementing both rule-based and retrieval-based approaches.
    
    Use case: Customer service automation, information retrieval, interactive assistants,
    and FAQ response systems.
    """
    
    def __init__(self, chatbot_type="rule_based", use_transformers=False, 
                model_name=None, **kwargs):
        """
        Initialize the chatbot model.
        
        Args:
            chatbot_type: Type of chatbot ("rule_based", "retrieval_based", or "transformer")
            use_transformers: Whether to use transformer models (requires transformers package)
            model_name: Name of transformer model to use (e.g., "facebook/blenderbot-400M-distill")
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.model_type = f"Chatbot ({chatbot_type})"
        self.nlp_task = "chatbot"
        self.chatbot_type = chatbot_type
        self.use_transformers = use_transformers
        self.model_name = model_name
        
        self.patterns = {}
        self.responses = {}
        self.documents = []
        self.vectorizer = None
        self.document_vectors = None
        
        # For transformer-based chatbots
        self.transformer_model = None
        self.tokenizer = None
        
        if use_transformers and not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformer models require the 'transformers' package. Please install it.")
            
        if use_transformers and model_name is None:
            self.model_name = "facebook/blenderbot-400M-distill"
            
        self.model_params.update({
            'chatbot_type': chatbot_type,
            'use_transformers': use_transformers,
            'model_name': self.model_name
        })
        
        if use_transformers and TRANSFORMERS_AVAILABLE:
            self._initialize_transformer_model()
    
    def _initialize_transformer_model(self):
        """Initialize the transformer model for chatbot."""
        if self.use_transformers and TRANSFORMERS_AVAILABLE:
            try:
                self.transformer_model = pipeline("conversational", model=self.model_name)
                self.is_trained = True
            except Exception as e:
                print(f"Error loading transformer model: {e}")
                print("Falling back to rule-based or retrieval-based methods.")
                self.use_transformers = False
    
    def add_pattern(self, pattern, responses, pattern_type="exact"):
        """
        Add a pattern-response pair for rule-based chatbot.
        
        Args:
            pattern: Pattern to match (string or regex)
            responses: List of possible responses
            pattern_type: Type of pattern matching ("exact", "contains", or "regex")
        """
        if isinstance(responses, str):
            responses = [responses]
            
        self.patterns[pattern] = {
            'responses': responses,
            'type': pattern_type
        }
    
    def add_document(self, document, metadata=None):
        """
        Add a document for retrieval-based chatbot.
        
        Args:
            document: Text document to add
            metadata: Additional information about the document
        """
        if metadata is None:
            metadata = {}
            
        self.documents.append({
            'text': document,
            'metadata': metadata
        })
        
        # Reset document vectors
        self.document_vectors = None
        self.is_trained = False
    
    def train(self, documents=None, **kwargs):
        """
        Train the chatbot model on documents (for retrieval-based).
        
        Args:
            documents: List of documents for retrieval-based chatbot
            **kwargs: Additional training parameters
            
        Returns:
            self: The trained model instance
        """
        if self.chatbot_type == "rule_based":
            # Rule-based chatbots don't require training
            self.is_trained = True
            return self
            
        if self.chatbot_type == "transformer" and self.use_transformers:
            # Transformer models are pre-trained
            self._initialize_transformer_model()
            return self
        
        # Retrieval-based training
        if documents is not None:
            for doc in documents:
                if isinstance(doc, dict) and 'text' in doc:
                    self.add_document(doc['text'], doc.get('metadata'))
                else:
                    self.add_document(doc)
        
        if not self.documents:
            raise ValueError("No documents provided for retrieval-based chatbot training")
            
        # Create vectorizer
        self.vectorizer = TfidfVectorizer()
        
        # Extract text from documents
        texts = [doc['text'] for doc in self.documents]
        
        # Create document vectors
        self.document_vectors = self.vectorizer.fit_transform(texts)
        
        self.is_trained = True
        return self
    
    def predict(self, query):
        """
        Generate a response to the user query.
        
        Args:
            query: User input text
            
        Returns:
            Chatbot response
        """
        if not self.is_trained and self.chatbot_type != "rule_based":
            raise ValueError("Model has not been trained yet. Call train() first.")
            
        if self.chatbot_type == "rule_based":
            return self._rule_based_response(query)
            
        elif self.chatbot_type == "retrieval_based":
            return self._retrieval_based_response(query)
            
        elif self.chatbot_type == "transformer" and self.use_transformers:
            return self._transformer_based_response(query)
            
        else:
            # Default to rule-based if no other method works
            return self._rule_based_response(query)
    
    def _rule_based_response(self, query):
        """Generate response using rule-based approach."""
        # Preprocess query
        processed_query = self.preprocess_text(query, lemmatize=True)
        
        # If no patterns defined, use default responses
        if not self.patterns:
            return random.choice([
                "I'm not sure how to respond to that.",
                "Could you please rephrase that?",
                "Interesting. Tell me more.",
                "I don't have enough information to respond properly."
            ])
        
        # Check each pattern
        for pattern, config in self.patterns.items():
            pattern_type = config['type']
            responses = config['responses']
            
            if pattern_type == "exact" and processed_query == pattern:
                return random.choice(responses)
                
            elif pattern_type == "contains" and pattern in processed_query:
                return random.choice(responses)
                
            elif pattern_type == "regex" and re.search(pattern, processed_query):
                return random.choice(responses)
                
        # No match found
        return random.choice([
            "I'm not sure I understand.",
            "Could you please provide more information?",
            "I don't know how to respond to that yet.",
            "I'm still learning. Could you try a different query?"
        ])
    
    def _retrieval_based_response(self, query):
        """Generate response using retrieval-based approach."""
        if not self.document_vectors is not None:
            raise ValueError("No document vectors available. Train the model first.")
            
        # Vectorize the query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarity
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        
        # Get the most similar document
        best_match_idx = similarities.argmax()
        best_match_score = similarities[best_match_idx]
        
        # If similarity is too low, return a default response
        if best_match_score < 0.1:
            return "I couldn't find a good match for your query. Could you please rephrase it?"
            
        # Return the best matching document
        return self.documents[best_match_idx]['text']
    
    def _transformer_based_response(self, query):
        """Generate response using transformer model."""
        if self.transformer_model is None:
            raise ValueError("Transformer model not initialized.")
            
        try:
            response = self.transformer_model(query)
            if isinstance(response, list) and len(response) > 0:
                return response[0]['generated_text']
            return response.generated_text
        except Exception as e:
            print(f"Error generating response with transformer: {e}")
            return "I'm having trouble generating a response right now."
    
    def chat(self, max_turns=10):
        """
        Interactive chat session with the chatbot.
        
        Args:
            max_turns: Maximum number of conversation turns
        """
        print(f"Starting chat with {self.model_type}. Type 'quit' to exit.")
        
        for _ in range(max_turns):
            user_input = input("\nYou: ")
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nChatbot: Goodbye!")
                break
                
            response = self.predict(user_input)
            print(f"\nChatbot: {response}")
    
    def save_model(self, filepath):
        """
        Save the chatbot model to disk.
        
        Args:
            filepath: Path to save the model
        """
        # Can't save transformer models
        if self.chatbot_type == "transformer" and self.use_transformers:
            print("Warning: Transformer model configuration is saved, but not the model itself.")
            
        model_data = {
            'model_params': self.model_params,
            'patterns': self.patterns,
            'documents': self.documents,
            'is_trained': self.is_trained,
            'chatbot_type': self.chatbot_type
        }
        
        # Save vectorizer if available
        if self.vectorizer is not None:
            model_data['vectorizer'] = self.vectorizer
            
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """
        Load a chatbot model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            self: The loaded model instance
        """
        model_data = joblib.load(filepath)
        
        # Load model parameters
        self.model_params = model_data['model_params']
        self.patterns = model_data['patterns']
        self.documents = model_data['documents']
        self.is_trained = model_data['is_trained']
        self.chatbot_type = model_data['chatbot_type']
        
        # Set attributes from model_params
        for key, value in self.model_params.items():
            setattr(self, key, value)
            
        # Load vectorizer if available
        if 'vectorizer' in model_data:
            self.vectorizer = model_data['vectorizer']
            
            # Recreate document vectors if needed
            if self.documents and self.vectorizer:
                texts = [doc['text'] for doc in self.documents]
                self.document_vectors = self.vectorizer.transform(texts)
                
        # Initialize transformer model if needed
        if self.chatbot_type == "transformer" and self.use_transformers and TRANSFORMERS_AVAILABLE:
            self._initialize_transformer_model()
            
        return self


class TextSummarizerModel(NLPBaseModel):
    """
    Text summarization model implementing extractive and abstractive approaches.
    
    Use case: Document summarization, news headline generation, content digestion,
    and information extraction from large texts.
    """
    
    def __init__(self, summarizer_type="extractive", use_transformers=False,
                model_name=None, max_length=100, min_length=30, **kwargs):
        """
        Initialize the text summarizer model.
        
        Args:
            summarizer_type: Type of summarization ("extractive" or "abstractive")
            use_transformers: Whether to use transformer models
            model_name: Name of transformer model to use (e.g., "facebook/bart-large-cnn")
            max_length: Maximum length of generated summary
            min_length: Minimum length of generated summary
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.model_type = f"Text Summarizer ({summarizer_type})"
        self.nlp_task = "summarization"
        self.summarizer_type = summarizer_type
        self.use_transformers = use_transformers
        self.model_name = model_name
        self.max_length = max_length
        self.min_length = min_length
        
        # For extractive summarization
        self.vectorizer = None
        
        # For transformer-based abstractive summarization
        self.transformer_model = None
        self.tokenizer = None
        
        if use_transformers and not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformer models require the 'transformers' package. Please install it.")
            
        if use_transformers and model_name is None:
            self.model_name = "facebook/bart-large-cnn" if summarizer_type == "abstractive" else "distilbert-base-uncased"
            
        self.model_params.update({
            'summarizer_type': summarizer_type,
            'use_transformers': use_transformers,
            'model_name': self.model_name,
            'max_length': max_length,
            'min_length': min_length
        })
        
        if use_transformers and TRANSFORMERS_AVAILABLE:
            self._initialize_transformer_model()
    
    def _initialize_transformer_model(self):
        """Initialize the transformer model for summarization."""
        if self.use_transformers and TRANSFORMERS_AVAILABLE:
            try:
                if self.summarizer_type == "abstractive":
                    self.transformer_model = pipeline("summarization", model=self.model_name)
                else:  # extractive
                    # For extractive, we'll use a general pipeline
                    self.transformer_model = pipeline("feature-extraction", model=self.model_name)
                self.is_trained = True
            except Exception as e:
                print(f"Error loading transformer model: {e}")
                print("Falling back to traditional summarization methods.")
                self.use_transformers = False
    
    def train(self, X_train=None, y_train=None, **kwargs):
        """
        Train the summarizer model if needed (mainly for extractive).
        
        Args:
            X_train: Training texts
            y_train: Target summaries
            **kwargs: Additional training parameters
            
        Returns:
            self: The trained model instance
        """
        if self.summarizer_type == "abstractive" and self.use_transformers:
            # Abstractive transformer models are pre-trained
            self._initialize_transformer_model()
            self.is_trained = True
            return self
            
        # For extractive summarization, we'll create a TF-IDF vectorizer
        if self.summarizer_type == "extractive":
            self.vectorizer = TfidfVectorizer(stop_words='english')
            # No actual training needed for our extractive approach
            self.is_trained = True
            
        return self
    
    def predict(self, text):
        """
        Generate a summary for the given text.
        
        Args:
            text: Input text to summarize
            
        Returns:
            Summary of the input text
        """
        if self.summarizer_type == "extractive":
            if self.use_transformers and self.transformer_model is not None:
                return self._transformer_extractive_summarize(text)
            else:
                return self._extractive_summarize(text)
                
        elif self.summarizer_type == "abstractive":
            if self.use_transformers and self.transformer_model is not None:
                return self._transformer_abstractive_summarize(text)
            else:
                # Fallback to extractive if no transformer available
                return self._extractive_summarize(text)
    
    def _extractive_summarize(self, text):
        """
        Perform extractive summarization using TF-IDF and sentence similarity.
        
        Args:
            text: Input text to summarize
            
        Returns:
            Extractive summary
        """
        # Split text into sentences
        sentences = self.tokenize_sentences(text)
        
        if len(sentences) <= 1:
            return text
            
        # Create vectorizer if not already done
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(stop_words='english')
            
        # Create sentence vectors
        try:
            sentence_vectors = self.vectorizer.fit_transform(sentences)
        except ValueError:
            # If an empty vocabulary is found, return first few sentences
            summary_length = min(3, len(sentences))
            return ' '.join(sentences[:summary_length])
            
        # Calculate sentence similarity matrix
        similarity_matrix = cosine_similarity(sentence_vectors)
        
        # Calculate sentence scores based on similarity
        sentence_scores = similarity_matrix.sum(axis=1)
        
        # Determine number of sentences for summary
        summary_length = max(1, min(int(len(sentences) * 0.3), int(self.max_length / 20)))
        
        # Get top sentences indices
        top_indices = sentence_scores.argsort()[-summary_length:]
        
        # Sort indices to preserve original order
        top_indices = sorted(top_indices)
        
        # Create summary by joining top sentences
        summary = ' '.join([sentences[i] for i in top_indices])
        
        return summary
    
    def _transformer_extractive_summarize(self, text):
        """
        Perform extractive summarization using transformer embeddings.
        
        Args:
            text: Input text to summarize
            
        Returns:
            Extractive summary
        """
        # Split text into sentences
        sentences = self.tokenize_sentences(text)
        
        if len(sentences) <= 1:
            return text
            
        # Get sentence embeddings
        try:
            embeddings = self.transformer_model(sentences)
            
            # For each sentence, get a single embedding by averaging
            sentence_vectors = np.array([np.mean(emb, axis=0) for emb in embeddings])
            
            # Calculate sentence similarity matrix
            similarity_matrix = cosine_similarity(sentence_vectors)
            
            # Calculate sentence scores based on similarity
            sentence_scores = similarity_matrix.sum(axis=1)
            
            # Determine number of sentences for summary
            summary_length = max(1, min(int(len(sentences) * 0.3), int(self.max_length / 20)))
            
            # Get top sentences indices
            top_indices = sentence_scores.argsort()[-summary_length:]
            
            # Sort indices to preserve original order
            top_indices = sorted(top_indices)
            
            # Create summary by joining top sentences
            summary = ' '.join([sentences[i] for i in top_indices])
            
            return summary
            
        except Exception as e:
            print(f"Error in transformer extractive summarization: {e}")
            return self._extractive_summarize(text)
    
    def _transformer_abstractive_summarize(self, text):
        """
        Perform abstractive summarization using transformer models.
        
        Args:
            text: Input text to summarize
            
        Returns:
            Abstractive summary
        """
        try:
            summary = self.transformer_model(text, 
                                           max_length=self.max_length, 
                                           min_length=self.min_length, 
                                           do_sample=False)
            
            if isinstance(summary, list) and len(summary) > 0:
                return summary[0]['summary_text']
            
            return summary[0]['summary_text']
            
        except Exception as e:
            print(f"Error in transformer abstractive summarization: {e}")
            return self._extractive_summarize(text)
    
    def evaluate(self, texts, reference_summaries):
        """
        Evaluate the summarizer's performance.
        
        Args:
            texts: List of input texts
            reference_summaries: List of reference summaries
            
        Returns:
            Dictionary with evaluation metrics
        """
        if len(texts) != len(reference_summaries):
            raise ValueError("Number of texts and reference summaries must match")
            
        # Import evaluation metrics
        try:
            from rouge import Rouge
            rouge = Rouge()
            have_rouge = True
        except ImportError:
            print("Rouge package not available. Install with: pip install rouge")
            have_rouge = False
            
        predicted_summaries = [self.predict(text) for text in texts]
        
        results = {}
        
        # Calculate ROUGE scores if available
        if have_rouge:
            try:
                rouge_scores = rouge.get_scores(predicted_summaries, reference_summaries, avg=True)
                results.update(rouge_scores)
            except Exception as e:
                print(f"Error calculating ROUGE scores: {e}")
                
        # Calculate average length ratio
        length_ratios = [len(pred) / len(ref) for pred, ref in zip(predicted_summaries, reference_summaries)]
        results['avg_length_ratio'] = sum(length_ratios) / len(length_ratios)
        
        return results
    
    def save_model(self, filepath):
        """
        Save the summarizer model to disk.
        
        Args:
            filepath: Path to save the model
        """
        # Can't save transformer models
        if self.use_transformers:
            print("Warning: Transformer model configuration is saved, but not the model itself.")
            
        model_data = {
            'model_params': self.model_params,
            'is_trained': self.is_trained,
            'summarizer_type': self.summarizer_type
        }
        
        # Save vectorizer if available
        if self.vectorizer is not None:
            model_data['vectorizer'] = self.vectorizer
            
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """
        Load a summarizer model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            self: The loaded model instance
        """
        model_data = joblib.load(filepath)
        
        # Load model parameters
        self.model_params = model_data['model_params']
        self.is_trained = model_data['is_trained']
        self.summarizer_type = model_data['summarizer_type']
        
        # Set attributes from model_params
        for key, value in self.model_params.items():
            setattr(self, key, value)
            
        # Load vectorizer if available
        if 'vectorizer' in model_data:
            self.vectorizer = model_data['vectorizer']
                
        # Initialize transformer model if needed
        if self.use_transformers and TRANSFORMERS_AVAILABLE:
            self._initialize_transformer_model()
            
        return self 