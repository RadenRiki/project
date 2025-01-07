import pandas as pd
import numpy as np
from preprocessing import prepare_data
from model import train_model, predict_sentiment
import matplotlib.pyplot as plt
import seaborn as sns

def test_model():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = pd.read_csv('../data/all_combined.csv')
    
    # Take a smaller sample first (misalnya 10000 rows)
    df_sample = df.sample(n=10000, random_state=42)
    print(f"Using {len(df_sample)} samples for testing...")
    
    df_clean = prepare_data(df_sample)
    
    # Train model
    print("\nTraining model...")
    model, vectorizer, performance = train_model(df_clean)
    
    # Print performance metrics
    print("\n=== MODEL PERFORMANCE ===")
    print("\nClassification Report:")
    print(performance['classification_report'])
    
    print("\nTop 20 Important Features:")
    print(performance['feature_importance'])
    
    # Test predictions
    print("\n=== TESTING PREDICTIONS ===")
    test_texts = [
        "This app is amazing, works perfectly!",
        "Terrible app, keeps crashing and losing my data",
        "It's okay, has some good features but needs improvement",
        "Love this app! Best one I've ever used 😊",
        "Can't login, very frustrating experience 😡"
    ]
    
    for text in test_texts:
        sentiment, probs = predict_sentiment(text, model, vectorizer)
        print(f"\nText: {text}")
        print(f"Predicted sentiment: {sentiment}")
        print(f"Confidence scores: negative={probs[0]:.2f}, neutral={probs[1]:.2f}, positive={probs[2]:.2f}")

if __name__ == "__main__":
    test_model()