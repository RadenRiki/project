import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(df):
    """
    Train the sentiment analysis model
    
    Parameters:
    df (pandas.DataFrame): Preprocessed dataframe with 'cleaned_content' and 'sentiment' columns
    
    Returns:
    tuple: (trained model, vectorizer, performance metrics)
    """
    # Prepare features and target
    X = df['cleaned_content']
    y = df['sentiment']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # Use both unigrams and bigrams
        stop_words='english'
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    
    model.fit(X_train_vec, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_vec)
    
    # Calculate performance metrics
    performance = {
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'test_samples': len(y_test),
        'feature_importance': pd.DataFrame({
            'feature': vectorizer.get_feature_names_out(),
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)
    }
    
    # Save model and vectorizer
    joblib.dump(model, '../models/sentiment_model.joblib')
    joblib.dump(vectorizer, '../models/vectorizer.joblib')
    
    return model, vectorizer, performance

def predict_sentiment(text, model, vectorizer):
    """
    Predict sentiment for new text
    
    Parameters:
    text (str): Input text
    model: Trained model
    vectorizer: Fitted vectorizer
    
    Returns:
    tuple: (predicted sentiment, probability scores)
    """
    # Vectorize the text
    text_vec = vectorizer.transform([text])
    
    # Get prediction and probabilities
    sentiment = model.predict(text_vec)[0]
    probabilities = model.predict_proba(text_vec)[0]
    
    return sentiment, probabilities