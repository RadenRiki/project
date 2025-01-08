# model.py

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
from sklearn.preprocessing import StandardScaler
from scipy import sparse
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import preprocess_text
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Pastikan bahwa NLTK data sudah diunduh
nltk.download('vader_lexicon', quiet=True)


def extract_features(text, sia):
    """
    Extract additional features from text, termasuk skor sentimen dari VADER
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
    
    # # Sentiment lexicon scores (contoh sederhana)
    # positive_lexicon = ['amazing', 'perfect', 'excellent']
    # negative_lexicon = ['terrible', 'worst', 'horrible']
    # features['positive_lexicon'] = sum(word in text.split() for word in positive_lexicon)
    # features['negative_lexicon'] = sum(word in text.split() for word in negative_lexicon)
    
    # SentimentIntensityAnalyzer scores
    sentiment_scores = sia.polarity_scores(text)
    features['vader_neg'] = sentiment_scores['neg']
    features['vader_neu'] = sentiment_scores['neu']
    features['vader_pos'] = sentiment_scores['pos']
    features['vader_compound'] = sentiment_scores['compound']
    
    return features


def create_ensemble_model():
    """
    Create an ensemble model combining RandomForest and LogisticRegression
    """
    # Initialize SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    
    # RandomForest hyperparameters
    rf = RandomForestClassifier(
        n_estimators=300,      # Increased from 200
        max_depth=40,          # Increased from 30
        min_samples_split=3,   # Decreased from 5
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # LogisticRegression hyperparameters
    lr = LogisticRegression(
        class_weight='balanced',
        max_iter=10000,         # Increased from 5000
        C=0.1,                  # Added regularization
        solver='saga',          # Changed solver to 'saga'
        random_state=42,
        n_jobs=-1
    )
    
    # Optional: Add SVM or other classifiers if desired
    svm = LinearSVC(
        class_weight='balanced',
        max_iter=2000,
        random_state=42
    )
    
    # Ensemble (Voting) Classifier
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('lr', lr),
            # ('svm', svm)  # Uncomment if you want to include SVM
        ],
        voting='soft',
        n_jobs=-1
    )
    
    return ensemble, sia


def train_model(df):
    """
    Train the sentiment analysis model with ensemble approach, 
    then use RandomForest feature importance instead of permutation importance.
    """
    print("Balancing dataset...")
    # Pisahkan data berdasarkan sentimen
    df_positive = df[df['sentiment'] == 'positive']
    df_negative = df[df['sentiment'] == 'negative']
    df_neutral = df[df['sentiment'] == 'neutral']
    
    # Oversample kelas minor agar seimbang
    max_class_size = max(len(df_positive), len(df_negative), len(df_neutral))
    df_positive_balanced = resample(df_positive, replace=True, n_samples=max_class_size, random_state=42)
    df_negative_balanced = resample(df_negative, replace=True, n_samples=max_class_size, random_state=42)
    df_neutral_balanced = resample(df_neutral, replace=True, n_samples=max_class_size, random_state=42)
    
    # Gabungkan kembali
    df_balanced = pd.concat([df_positive_balanced, df_negative_balanced, df_neutral_balanced])
    print(f"Balanced class distribution:\n{df_balanced['sentiment'].value_counts()}")
    
    # Pisahkan features (X) dan target (y)
    X = df_balanced['cleaned_content']
    y = df_balanced['sentiment']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=5
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Simpan nama fitur TF-IDF
    joblib.dump(vectorizer.get_feature_names_out(), '../models/feature_names.joblib')
    
    # Inisialisasi SIA
    sia = SentimentIntensityAnalyzer()
    
    # Buat fitur tambahan
    print("Extracting additional features...")
    train_features = pd.DataFrame([extract_features(text, sia) for text in X_train])
    test_features = pd.DataFrame([extract_features(text, sia) for text in X_test])
    
    # Scaling fitur tambahan
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    # Simpan scaler
    joblib.dump(scaler, '../models/scaler.joblib')
    
    # Kembalikan ke DataFrame agar kolomnya masih bisa diakses
    train_features_scaled = pd.DataFrame(train_features_scaled, columns=train_features.columns)
    test_features_scaled = pd.DataFrame(test_features_scaled, columns=test_features.columns)
    
    # Jumlah total fitur
    n_features = X_train_vec.shape[1] + len(train_features_scaled.columns)
    joblib.dump(n_features, '../models/n_features.joblib')
    
    # Gabungkan TF-IDF + fitur tambahan
    X_train_combined = sparse.hstack([X_train_vec, train_features_scaled])
    X_test_combined = sparse.hstack([X_test_vec, test_features_scaled])
    
    # Buat model ensemble
    print("Training ensemble model...")
    ensemble_model, sia = create_ensemble_model()
    
    # Cross-validation
    cv_scores = cross_val_score(
        ensemble_model, 
        X_train_combined, 
        y_train, 
        cv=5, 
        scoring='accuracy', 
        n_jobs=-1
    )
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
    
    # Latih final model
    ensemble_model.fit(X_train_combined, y_train)
    
    # Prediksi di test set
    y_pred = ensemble_model.predict(X_test_combined)
    
    # Kumpulkan metrik performa
    performance = {
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'test_samples': len(y_test),
        'cross_val_scores': cv_scores
    }
    
    # ------------------------------
    #  GUNAKAN RANDOM FOREST FEATURE IMPORTANCE
    # ------------------------------
    print("Extracting feature importances from Random Forest...")
    # Ambil model Random Forest dari ensemble
    rf_model = ensemble_model.named_estimators_['rf']
    
    # Dapatkan nama semua fitur
    feature_names = list(vectorizer.get_feature_names_out()) + list(train_features_scaled.columns)
    
    # Dapatkan importance
    rf_importances = rf_model.feature_importances_
    
    # Buat DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_importances
    }).sort_values('importance', ascending=False).head(20)
    
    performance['feature_importance'] = feature_importance_df
    
    # Buat folder models jika belum ada
    os.makedirs('../models', exist_ok=True)
    
    # Simpan model dan vectorizer
    joblib.dump(ensemble_model, '../models/sentiment_model.joblib')
    joblib.dump(vectorizer, '../models/vectorizer.joblib')
    
    return ensemble_model, vectorizer, performance


def predict_sentiment(text, model, vectorizer, n_features, confidence_threshold=0.6):
    """
    Predict sentiment for new text with confidence threshold
    """
    # Initialize SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    
    # Preprocess text
    processed_text, _ = preprocess_text(text)
    
    # Vectorize the text
    text_vec = vectorizer.transform([processed_text])
    
    # Extract additional features (pakai SIA)
    additional_features = pd.DataFrame([extract_features(processed_text, sia)])
    
    # Load scaler agar consistent
    scaler = joblib.load('../models/scaler.joblib')
    additional_features_scaled = scaler.transform(additional_features)
    additional_features_scaled = pd.DataFrame(additional_features_scaled, columns=additional_features.columns)
    
    # Gabungkan semua fitur
    combined_features = sparse.hstack([text_vec, additional_features_scaled])
    
    # Cek kesesuaian jumlah fitur
    actual_n_features = combined_features.shape[1]
    if actual_n_features != n_features:
        print(f"Feature dimensionality mismatch: expected {n_features}, got {actual_n_features}")
        return 'neutral', [0.0, 1.0, 0.0], False
    
    # Prediksi
    sentiment = model.predict(combined_features)[0]
    probabilities = model.predict_proba(combined_features)[0]
    
    # Confidence threshold
    max_prob = max(probabilities)
    if max_prob < confidence_threshold:
        # Fallback ke word counts
        if additional_features['positive_words'].iloc[0] > additional_features['negative_words'].iloc[0]:
            sentiment = 'positive'
        elif additional_features['negative_words'].iloc[0] > additional_features['positive_words'].iloc[0]:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        is_uncertain = True
    else:
        is_uncertain = False
    
    return sentiment, probabilities, is_uncertain


# Example usage (optional):
# if __name__ == "__main__":
#     df = pd.read_csv('../data/all_combined.csv')
#     model, vectorizer, performance = train_model(df)
#     print(performance['classification_report'])
