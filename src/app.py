# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from preprocessing import preprocess_text
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import sparse
import re

def extract_features(text):
    """Extract additional features from text"""
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
    
    return pd.DataFrame([features])

def analyze_sentiment(text, model, vectorizer, n_features):
    """Analyze sentiment of a single text"""
    # Preprocess text
    processed_text, _ = preprocess_text(text)
    
    # Vectorize the text
    text_vec = vectorizer.transform([processed_text])
    
    # Extract additional features
    extra_features = extract_features(processed_text)
    
    # Combine features
    features = sparse.hstack([text_vec, extra_features])
    
    # Validate feature dimensionality
    actual_n_features = features.shape[1]
    if actual_n_features != n_features:
        st.error(f"Feature dimensionality mismatch: expected {n_features}, got {actual_n_features}")
        return 'neutral', [0.0, 1.0, 0.0]  # Default to neutral if mismatch
    
    # Get prediction and probabilities
    sentiment = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    # Tambah confidence threshold
    max_prob = max(probabilities)
    if max_prob < 0.6:  # Jika confidence rendah
        # Ambil data features dari extra_features (yang sudah dalam bentuk DataFrame)
        if extra_features['positive_words'].iloc[0] > extra_features['negative_words'].iloc[0]:
            return 'positive', probabilities
        elif extra_features['negative_words'].iloc[0] > extra_features['positive_words'].iloc[0]:
            return 'negative', probabilities
        else:
            return 'neutral', probabilities
    
    return sentiment, probabilities

# Load model and vectorizer
@st.cache_resource
def load_model():
    try:
        model = joblib.load('../models/sentiment_model.joblib')
        vectorizer = joblib.load('../models/vectorizer.joblib')
        feature_names = joblib.load('../models/feature_names.joblib')
        n_features = joblib.load('../models/n_features.joblib')
        return model, vectorizer, feature_names, n_features
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

def plot_sentiment_distribution(df):
    """Plot sentiment distribution"""
    fig = px.pie(
        names=df['sentiment'].value_counts().index,
        values=df['sentiment'].value_counts().values,
        title='Sentiment Distribution',
        color_discrete_sequence=['#FF4B4B', '#A0A0A0', '#45B6FE']
    )
    return fig

def main():
    st.title("App Review Analyzer for Developers")
    
    # Sidebar
    st.sidebar.header("Options")
    analysis_mode = st.sidebar.radio(
        "Choose Analysis Mode",
        ["Single Review Analysis", "Bulk Review Analysis"]
    )
    
    try:
        model, vectorizer, feature_names, n_features = load_model()
        
        if model is None or vectorizer is None or feature_names is None or n_features is None:
            st.error("Failed to load model. Please check if model files exist.")
            return
        
        if analysis_mode == "Single Review Analysis":
            st.subheader("Analyze Single Review")
            
            # Text input
            review_text = st.text_area(
                "Enter review text:",
                height=100,
                placeholder="Enter the review text here..."
            )
            
            if st.button("Analyze") and review_text:
                with st.spinner("Analyzing..."):
                    # Get sentiment and probabilities
                    sentiment, probs = analyze_sentiment(review_text, model, vectorizer, n_features)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Sentiment")
                        sentiment_color = {
                            'positive': 'green',
                            'neutral': 'gray',
                            'negative': 'red'
                        }
                        st.markdown(
                            f'<h1 style="color: {sentiment_color[sentiment]};">{sentiment.upper()}</h1>',
                            unsafe_allow_html=True
                        )
                    
                    with col2:
                        st.markdown("### Confidence Scores")
                        st.progress(probs[0], text=f"Negative: {probs[0]:.2%}")
                        st.progress(probs[1], text=f"Neutral: {probs[1]:.2%}")
                        st.progress(probs[2], text=f"Positive: {probs[2]:.2%}")
        
        else:  # Bulk Analysis
            st.subheader("Analyze Multiple Reviews")
            
            uploaded_file = st.file_uploader(
                "Upload CSV file with reviews",
                type=['csv'],
                help="CSV should have a 'content' column containing review text"
            )
            
            if uploaded_file:
                with st.spinner("Processing reviews..."):
                    # Read CSV
                    df = pd.read_csv(uploaded_file)
                    
                    if 'content' not in df.columns:
                        st.error("CSV must contain a 'content' column!")
                        return
                    
                    # Process each review
                    results = []
                    for text in df['content']:
                        sentiment, probs = analyze_sentiment(text, model, vectorizer, n_features)
                        results.append({
                            'text': text,
                            'sentiment': sentiment,
                            'negative_prob': probs[0],
                            'neutral_prob': probs[1],
                            'positive_prob': probs[2]
                        })
                    
                    results_df = pd.DataFrame(results)
                    
                    # Display results
                    st.markdown("### Analysis Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Sentiment Distribution")
                        fig = plot_sentiment_distribution(results_df)
                        st.plotly_chart(fig)
                    
                    with col2:
                        st.markdown("#### Summary Statistics")
                        sentiment_counts = results_df['sentiment'].value_counts()
                        st.write(sentiment_counts)
                    
                    # Show detailed results
                    st.markdown("### Detailed Results")
                    st.dataframe(results_df)
                    
                    # Download results
                    st.download_button(
                        "Download Results",
                        results_df.to_csv(index=False),
                        "analyzed_reviews.csv",
                        "text/csv",
                        key='download-csv'
                    )
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.error("Please make sure model files are present in the 'models' directory")

if __name__ == "__main__":
    main()
