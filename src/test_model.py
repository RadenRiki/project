# test_model.py

import pandas as pd
import numpy as np
from preprocessing import prepare_data
from model import train_model, predict_sentiment
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

def plot_confusion_matrix(confusion_matrix, labels):
    """Plot confusion matrix using seaborn"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Create plots directory if it doesn't exist
    os.makedirs('../plots', exist_ok=True)
    plt.savefig('../plots/confusion_matrix.png')
    plt.close()

def plot_feature_importance(feature_importance_df):
    """Plot feature importance"""
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', 
                y='feature',
                data=feature_importance_df.head(20))
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    
    # Save plot
    os.makedirs('../plots', exist_ok=True)
    plt.savefig('../plots/feature_importance.png')
    plt.close()

def test_model():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = pd.read_csv('../data/all_combined.csv')
    
    # Take a sample for testing (increased from 10000 to get better representation)
    sample_size = 20000
    df_sample = df.sample(n=sample_size, random_state=42)
    print(f"Using {len(df_sample)} samples for testing...")
    
    # Print initial class distribution
    print("\nInitial class distribution:")
    print(df_sample['score'].value_counts().sort_index())
    
    # Preprocess data
    df_clean = prepare_data(df_sample)
    
    print("\nAfter preprocessing, sentiment distribution:")
    print(df_clean['sentiment'].value_counts())
    
    # Train model
    print("\nTraining model...")
    model, vectorizer, performance = train_model(df_clean)
    
    # Load n_features
    try:
        n_features = joblib.load('../models/n_features.joblib')
    except Exception as e:
        print(f"Error loading n_features: {str(e)}")
        return
    
    # Print performance metrics
    print("\n=== MODEL PERFORMANCE ===")
    print("\nClassification Report:")
    print(performance['classification_report'])
    
    # Plot confusion matrix
    plot_confusion_matrix(
        performance['confusion_matrix'],
        ['negative', 'neutral', 'positive']
    )
    print("\nConfusion matrix plot saved to ../plots/confusion_matrix.png")
    
    # Plot and print feature importance
    print("\nTop 20 Important Features:")
    print(performance['feature_importance'])
    plot_feature_importance(performance['feature_importance'])
    print("Feature importance plot saved to ../plots/feature_importance.png")
    
    # Test predictions with diverse examples
    print("\n=== TESTING PREDICTIONS ===")
    test_texts = [
        # Positive examples
        "I absolutely love this app! It has changed my life for the better 😊👍",
        "This app is amazing, works perfectly! 😊",
        "Love this app! Best one I've ever used. Super smooth and helpful",
        "Great improvements in the latest update, much faster now",
        
        # Negative examples
        "Terrible app, keeps crashing and losing my data 😡",
        "Worst app ever! Full of bugs and advertisements",
        "Can't login, very frustrating experience. Don't waste your time",
        
        # Neutral examples
        "It's okay, has some good features but needs improvement",
        "Average app, does the job but nothing special",
        "Updated to latest version. Some things better, others worse",
        
        # Mixed signals
        "Good app but has some annoying bugs",
        "Great concept but needs better execution",
        "Amazing features but terrible battery drain"
    ]
    
    print("\nTesting various types of reviews:")
    for text in test_texts:
        sentiment, probs, is_uncertain = predict_sentiment(text, model, vectorizer, n_features, confidence_threshold=0.6)
        if is_uncertain:
            sentiment_display = f"UNCERTAIN ({sentiment.upper()})"
            uncertainty_note = " - Model is uncertain about this prediction."
        else:
            sentiment_display = sentiment.upper()
            uncertainty_note = ""
        print(f"\nText: {text}")
        print(f"Predicted Sentiment: {sentiment_display}{uncertainty_note}")
        print(f"Confidence Scores: Negative={probs[0]:.2f}, Neutral={probs[1]:.2f}, Positive={probs[2]:.2f}")
        
    print("\nModel and vectorizer saved in ../models/")
    print("Visualization plots saved in ../plots/")

if __name__ == "__main__":
    test_model()
