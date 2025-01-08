import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
from scipy import sparse
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def extract_features(text):
    """
    Extract additional features from text
    """
    features = {}
    
    # Convert to string if not already
    text = str(text).lower()
    
    # Extended sentiment words lists
    positive_words = [
        'good', 'great', 'awesome', 'excellent', 'love', 'perfect', 'best', 'amazing',
        'wonderful', 'fantastic', 'super', 'brilliant', 'outstanding', 'superb',
        'happy', 'pleased', 'satisfied', 'smooth', 'helpful', 'impressive', 'easy',
        'fast', 'reliable', 'efficient', 'improved', 'recommended', 'worth'
    ]
    negative_words = [
        'bad', 'terrible', 'worst', 'poor', 'horrible', 'waste', 'crash', 'bug',
        'problem', 'awful', 'useless', 'annoying', 'disappointing', 'frustrating',
        'slow', 'broken', 'error', 'issues', 'fail', 'garbage', 'rubbish', 'hate',
        'unusable', 'expensive', 'spam', 'ads', 'scam', 'freeze', 'stuck'
    ]
    neutral_words = [
        'okay', 'average', 'moderate', 'decent', 'normal', 'fine', 'alright',
        'fair', 'regular', 'basic', 'simple', 'standard', 'typical', 'expected',
        'usual', 'common', 'middle', 'medium', 'neither', 'mixed', 'partial'
    ]
    
    # Sentiment word counts
    features['positive_words'] = sum(word in text.split() for word in positive_words)
    features['negative_words'] = sum(word in text.split() for word in negative_words)
    features['neutral_words'] = sum(word in text.split() for word in neutral_words)
    
    # Contradiction features
    contradictions = ['but', 'however', 'although', 'though', 'despite', 'except']
    features['has_contradiction'] = int(any(word in text.split() for word in contradictions))
    
    # Text statistics
    features['text_length'] = len(text)
    features['word_count'] = len(text.split())
    features['avg_word_length'] = features['text_length'] / (features['word_count'] + 1)
    
    # Special characters
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['emoji_count'] = sum(1 for char in text if char in '😊😃😄😅😂🤣😭😡💕❤️👍')
    
    # Emphasis features
    features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
    features['repeated_chars'] = len(re.findall(r'(.)\1{2,}', text))
    
    return features

def create_ensemble_model():
    """
    Create an ensemble model combining RF, LogisticRegression, and LinearSVC
    """
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    lr = LogisticRegression(
        class_weight='balanced',
        max_iter=2000,  # Increase from 1000
        C=0.1,         # Add regularization
        random_state=42,
        n_jobs=-1
    )
    
    svm = LinearSVC(
        class_weight='balanced',
        max_iter=2000,
        random_state=42
    )
    
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('lr', lr)
        ],
        voting='soft'
    )
    
    return ensemble

def train_model(df):
    """
    Train the sentiment analysis model with ensemble approach
    """
    print("Balancing dataset...")
    # Separate by sentiment
    df_positive = df[df['sentiment'] == 'positive']
    df_negative = df[df['sentiment'] == 'negative']
    df_neutral = df[df['sentiment'] == 'neutral']
    
    # Find the minimum class size
    min_class_size = max(
        min(len(df_positive), len(df_negative), len(df_neutral)),
        1000
    )
    
    # Balance classes
    df_positive_balanced = resample(df_positive, 
                                  replace=len(df_positive) < min_class_size,
                                  n_samples=min_class_size,
                                  random_state=42)
    df_negative_balanced = resample(df_negative,
                                  replace=len(df_negative) < min_class_size,
                                  n_samples=min_class_size,
                                  random_state=42)
    df_neutral_balanced = resample(df_neutral,
                                 replace=len(df_neutral) < min_class_size,
                                 n_samples=min_class_size,
                                 random_state=42)
    
    # Combine balanced dataset
    df_balanced = pd.concat([
        df_positive_balanced, 
        df_negative_balanced,
        df_neutral_balanced
    ])
    
    print(f"Balanced class distribution:\n{df_balanced['sentiment'].value_counts()}")
    
    # Prepare features and target
    X = df_balanced['cleaned_content']
    y = df_balanced['sentiment']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=5
    )
    
    # Transform text data
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Save feature names
    joblib.dump(vectorizer.get_feature_names_out(), '../models/feature_names.joblib')
    
    # Extract additional features
    print("Extracting additional features...")
    train_features = pd.DataFrame([extract_features(text) for text in X_train])
    test_features = pd.DataFrame([extract_features(text) for text in X_test])
    
    # Save total number of features
    n_features = X_train_vec.shape[1] + len(train_features.columns)
    joblib.dump(n_features, '../models/n_features.joblib')
    
    # Combine TF-IDF and additional features
    X_train_combined = sparse.hstack([X_train_vec, train_features])
    X_test_combined = sparse.hstack([X_test_vec, test_features])
    
    # Create and train ensemble model
    print("Training ensemble model...")
    model = create_ensemble_model()
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train_combined, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
    
    # Train final model
    model.fit(X_train_combined, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_combined)
    
    # Calculate performance metrics
    feature_names = (list(vectorizer.get_feature_names_out()) + 
                    list(train_features.columns))
    
    performance = {
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'test_samples': len(y_test),
        'feature_importance': pd.DataFrame({
            'feature': feature_names,
            'importance': [0.01] * len(feature_names)  # Placeholder since ensemble doesn't have feature_importances_
        }).sort_values('importance', ascending=False).head(20),
        'cross_val_scores': cv_scores
    }
    
    # Create models directory if it doesn't exist
    os.makedirs('../models', exist_ok=True)
    
    # Save model and vectorizer
    joblib.dump(model, '../models/sentiment_model.joblib')
    joblib.dump(vectorizer, '../models/vectorizer.joblib')
    
    return model, vectorizer, performance

def predict_sentiment(text, model, vectorizer):
    """
    Predict sentiment for new text with confidence threshold
    """
    # Vectorize the text
    text_vec = vectorizer.transform([text])
    
    # Extract additional features
    additional_features = pd.DataFrame([extract_features(text)])
    
    # Combine features
    combined_features = sparse.hstack([text_vec, additional_features])
    
    # Get prediction and probabilities
    sentiment = model.predict(combined_features)[0]
    probabilities = model.predict_proba(combined_features)[0]
    
    # Get confidence scores
    max_prob = max(probabilities)
    
    # If confidence is too low, return neutral
    if max_prob < 0.4:  # Confidence threshold
        return 'neutral', probabilities
        
    return sentiment, probabilities