"""
NLP Chatbot and Text Summarization Example

This example demonstrates how to use the NLP models for chatbot functionality 
and text summarization. It shows both rule-based and retrieval-based chatbots,
as well as extractive and abstractive text summarization.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.nlp import ChatbotModel, TextSummarizerModel

# Sample text for demonstration
LONG_TEXT = """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural 
intelligence displayed by animals including humans. AI research has been defined as the field of 
study of intelligent agents, which refers to any system that perceives its environment and takes 
actions that maximize its chance of achieving its goals.

The term "artificial intelligence" had previously been used to describe machines that mimic and 
display "human" cognitive skills that are associated with the human mind, such as "learning" and 
"problem-solving". This definition has since been rejected by major AI researchers who now describe 
AI in terms of rationality and acting rationally, which does not limit how intelligence can be 
articulated.

AI applications include advanced web search engines (e.g., Google), recommendation systems 
(used by YouTube, Amazon and Netflix), understanding human speech (such as Siri and Alexa), 
self-driving cars (e.g., Waymo), generative or creative tools (ChatGPT and AI art), 
automated decision-making and competing at the highest level in strategic game systems 
(such as chess and Go).

As machines become increasingly capable, tasks considered to require "intelligence" are often 
removed from the definition of AI, a phenomenon known as the AI effect. For instance, optical 
character recognition is frequently excluded from things considered to be AI, having become a 
routine technology.

The various sub-fields of AI research are centered around particular goals and the use of 
particular tools. The traditional goals of AI research include reasoning, knowledge representation, 
planning, learning, natural language processing, perception, and the ability to move and 
manipulate objects. General intelligence (the ability to solve an arbitrary problem) is among 
the field's long-term goals. To solve these problems, AI researchers have adapted and integrated 
a wide range of problem-solving techniques, including search and mathematical optimization, 
formal logic, artificial neural networks, and methods based on statistics, probability and economics.
"""

def main():
    """Run the NLP examples."""
    print("NLP Chatbot and Text Summarization Example")
    print("==========================================\n")
    
    # Part 1: Chatbot Example
    print("\n--- CHATBOT EXAMPLE ---\n")
    chatbot_example()
    
    # Part 2: Text Summarization Example
    print("\n--- TEXT SUMMARIZATION EXAMPLE ---\n")
    summarization_example()

def chatbot_example():
    """Demonstrate chatbot functionality."""
    print("Creating a rule-based chatbot...")
    
    # Create a rule-based chatbot
    rule_based_bot = ChatbotModel(chatbot_type="rule_based")
    
    # Add some pattern-response pairs
    rule_based_bot.add_pattern("hello", ["Hi there!", "Hello!", "Hey!"], pattern_type="contains")
    rule_based_bot.add_pattern("how are you", ["I'm doing well, thanks for asking!", "I'm great! How are you?"], pattern_type="contains")
    rule_based_bot.add_pattern("bye", ["Goodbye!", "See you later!", "Have a nice day!"], pattern_type="contains")
    rule_based_bot.add_pattern("name", ["My name is AI Assistant.", "I'm an AI Assistant."], pattern_type="contains")
    rule_based_bot.add_pattern("help", ["I can answer simple questions and have a basic conversation."], pattern_type="contains")
    
    # Patterns related to AI
    rule_based_bot.add_pattern("what is ai", ["AI stands for Artificial Intelligence. It involves creating machines that can perform tasks requiring human intelligence."], pattern_type="contains")
    rule_based_bot.add_pattern("machine learning", ["Machine Learning is a subset of AI that enables computers to learn from data without being explicitly programmed."], pattern_type="contains")
    
    # Train the chatbot (for rule-based, this just sets is_trained to True)
    rule_based_bot.train()
    
    # Demonstrate rule-based chatbot
    print("\nRule-based Chatbot Responses:")
    demo_queries = [
        "Hello there!",
        "What is your name?",
        "What is AI?",
        "Tell me about machine learning",
        "Can you help me?",
        "How are you doing today?",
        "I don't understand quantum physics",
        "Goodbye!"
    ]
    
    for query in demo_queries:
        response = rule_based_bot.predict(query)
        print(f"User: {query}")
        print(f"Chatbot: {response}\n")
    
    # Create a retrieval-based chatbot
    print("\nCreating a retrieval-based chatbot...")
    
    # Use some simple documents as the knowledge base
    documents = [
        "Artificial Intelligence (AI) is the simulation of human intelligence in machines.",
        "Machine Learning is a subset of AI that focuses on data and algorithms.",
        "Deep Learning is a type of machine learning inspired by the structure of the human brain.",
        "Natural Language Processing (NLP) enables computers to understand and process human language.",
        "Computer Vision is the field of AI that trains computers to interpret visual information.",
        "Reinforcement Learning is learning what to do to maximize a reward signal.",
        "Neural Networks are computing systems inspired by biological neural networks.",
        "The Turing Test, developed by Alan Turing in 1950, is a test of a machine's ability to exhibit intelligent behavior.",
        "Expert Systems were among the first truly successful forms of AI software.",
        "AI Ethics concerns the moral behaviors of humans as they design, construct, use, and treat AI.",
    ]
    
    retrieval_bot = ChatbotModel(chatbot_type="retrieval_based")
    retrieval_bot.train(documents=documents)
    
    # Demonstrate retrieval-based chatbot
    print("\nRetrieval-based Chatbot Responses:")
    retrieval_queries = [
        "What is artificial intelligence?",
        "Tell me about machine learning",
        "How does deep learning work?",
        "What is NLP?",
        "Explain computer vision",
        "What is the Turing test?",
        "Are there ethical concerns with AI?"
    ]
    
    for query in retrieval_queries:
        response = retrieval_bot.predict(query)
        print(f"User: {query}")
        print(f"Chatbot: {response}\n")
    
    # Save the trained chatbot
    print("\nSaving the rule-based chatbot model...")
    rule_based_bot.save_model("../data/rule_based_chatbot_model")
    
    # Interactive mode (commented out by default)
    # print("\nEntering interactive mode with the rule-based chatbot (type 'exit' to quit)...")
    # while True:
    #     user_input = input("You: ")
    #     if user_input.lower() == 'exit':
    #         print("Chatbot: Goodbye!")
    #         break
    #     response = rule_based_bot.predict(user_input)
    #     print(f"Chatbot: {response}")

def summarization_example():
    """Demonstrate text summarization functionality."""
    print("Demonstrating extractive text summarization...")
    
    # Create an extractive summarizer
    extractive_summarizer = TextSummarizerModel(
        summarizer_type="extractive",
        max_length=150,
        min_length=50
    )
    
    # Train the summarizer (for extractive, this just initializes the vectorizer)
    extractive_summarizer.train()
    
    # Generate an extractive summary
    extractive_summary = extractive_summarizer.predict(LONG_TEXT)
    
    print("\nOriginal Text:")
    print(LONG_TEXT)
    
    print("\nExtractive Summary:")
    print(extractive_summary)
    
    # Compare word counts
    original_word_count = len(LONG_TEXT.split())
    summary_word_count = len(extractive_summary.split())
    
    print(f"\nOriginal text: {original_word_count} words")
    print(f"Extractive summary: {summary_word_count} words")
    print(f"Compression ratio: {summary_word_count / original_word_count:.2f}")
    
    # Try abstractive summarization if transformers are available
    try:
        from transformers import pipeline
        
        print("\nDemonstrating abstractive text summarization...")
        
        # Create an abstractive summarizer
        abstractive_summarizer = TextSummarizerModel(
            summarizer_type="abstractive",
            use_transformers=True,
            max_length=75,
            min_length=30
        )
        
        # Train the summarizer (for abstractive, this initializes the transformer model)
        try:
            abstractive_summarizer.train()
            
            # Generate an abstractive summary
            abstractive_summary = abstractive_summarizer.predict(LONG_TEXT)
            
            print("\nAbstractive Summary:")
            print(abstractive_summary)
            
            # Compare word counts
            abstractive_word_count = len(abstractive_summary.split())
            print(f"Abstractive summary: {abstractive_word_count} words")
            print(f"Compression ratio: {abstractive_word_count / original_word_count:.2f}")
            
            # Save the trained summarizer
            print("\nSaving the summarizer models...")
            extractive_summarizer.save_model("../data/extractive_summarizer_model")
            abstractive_summarizer.save_model("../data/abstractive_summarizer_model")
            
        except Exception as e:
            print(f"Error initializing transformer model: {e}")
            print("Falling back to extractive summarization only.")
            
    except ImportError:
        print("\nTransformers package not available. Abstractive summarization example skipped.")
        print("Install with: pip install transformers")
    
    # Demonstrate summarization on a collection of documents
    print("\nSummarizing multiple documents...")
    
    # Sample articles
    sample_articles = [
        """
        The International Monetary Fund (IMF) has projected that the global economy will grow by 3.5% in 2023, 
        a slight increase from earlier forecasts. This improved outlook is attributed to resilient consumer 
        spending, reduced supply chain disruptions, and effective monetary policies by central banks worldwide. 
        However, the IMF warns that inflation remains a concern, with many countries still experiencing price 
        pressures above target levels. The report highlighted that advanced economies are expected to see slower 
        growth compared to emerging markets and developing economies, which continue to show strong momentum 
        despite global challenges. China's economic reopening after strict COVID-19 lockdowns is expected to 
        contribute significantly to global growth, although the recovery may be uneven across sectors.
        """,
        
        """
        Scientists have discovered a new species of deep-sea fish that can withstand extreme pressure at depths 
        of over 8,000 meters in the Mariana Trench. The fish, named Pseudoliparis swirei, has several unique 
        adaptations including specialized cell membranes and high levels of trimethylamine N-oxide (TMAO), a 
        chemical that stabilizes proteins under pressure. Researchers from the Woods Hole Oceanographic Institution 
        observed the fish during a recent expedition using advanced deep-sea submersibles equipped with high-definition 
        cameras. The discovery provides new insights into how life can adapt to extreme environments and may have 
        implications for various fields including biotechnology and medicine. This finding expands our understanding 
        of biodiversity in one of Earth's least explored ecosystems.
        """
    ]
    
    for i, article in enumerate(sample_articles):
        print(f"\nSample Article {i+1} (Original length: {len(article.split())} words):")
        print(article[:200] + "...")  # Show just the beginning
        
        summary = extractive_summarizer.predict(article)
        print(f"\nSummary (length: {len(summary.split())} words):")
        print(summary)

if __name__ == "__main__":
    main() 