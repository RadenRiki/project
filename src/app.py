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

# Tambahkan import untuk VADER
from nltk.sentiment import SentimentIntensityAnalyzer

def extract_features(text):
    """
    Extract additional features from text
    (selaras dengan yang di model.py, tanpa positive_lexicon/negative_lexicon)
    """
    # Inisialisasi SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

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
    
    # --- Sentiment word counts ---
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
    
    # VADER scores
    sentiment_scores = sia.polarity_scores(text)
    features['vader_neg'] = sentiment_scores['neg']
    features['vader_neu'] = sentiment_scores['neu']
    features['vader_pos'] = sentiment_scores['pos']
    features['vader_compound'] = sentiment_scores['compound']
    
    return pd.DataFrame([features])

def analyze_sentiment(text, model, vectorizer, n_features, confidence_threshold=0.6):
    """
    Analyze sentiment of a single text with confidence threshold
    """
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
        return 'neutral', [0.0, 1.0, 0.0], False  # Default to neutral if mismatch
    
    # Get prediction and probabilities
    sentiment = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    # Check confidence threshold
    max_prob = max(probabilities)
    if max_prob < confidence_threshold:
        # Determine sentiment based on word counts as fallback
        if extra_features['positive_words'].iloc[0] > extra_features['negative_words'].iloc[0]:
            sentiment = 'positive'
        elif extra_features['negative_words'].iloc[0] > extra_features['positive_words'].iloc[0]:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        is_uncertain = True
    else:
        is_uncertain = False
    
    return sentiment, probabilities, is_uncertain

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
    
    # Add app description
    st.markdown("""
    Analyze user reviews sentiment quickly and accurately. Upload single reviews or bulk analyze multiple reviews at once.
    The model achieves 85% accuracy across positive, negative, and neutral sentiments.
    """)
    
    # Sidebar
    st.sidebar.header("Options")
    
    # Add How to Use guide in sidebar
    st.sidebar.markdown("### How to Use")
    st.sidebar.markdown("""
    **Single Review Analysis:**
    - Paste your review text
    - Adjust confidence threshold if needed
    - Click Analyze
    
    **Bulk Analysis:**
    - Prepare CSV with 'content' column
    - Upload CSV file
    - Get instant analysis of all reviews
    
    **Understanding Results:**
    - Confidence scores show model's certainty
    - UNCERTAIN tag means low confidence
    - Always verify critical reviews manually
    """)
    
    # Mode selection
    analysis_mode = st.sidebar.radio(
        "Choose Analysis Mode",
        ["Single Review Analysis", "Bulk Review Analysis"]
    )
    
    # Model performance metrics
    if st.sidebar.checkbox("Show Model Performance"):
        st.sidebar.info("""
        **Model Performance Metrics:**
        - Accuracy: 85%
        - Precision: 86%
        - Recall: 85%
        - F1-Score: 85%
        """)
    
    try:
        model, vectorizer, feature_names, n_features = load_model()
        
        if model is None or vectorizer is None or feature_names is None or n_features is None:
            st.error("⚠️ Failed to load model. Please check if model files exist in the models directory.")
            return
        
        if analysis_mode == "Single Review Analysis":
            st.subheader("Analyze Single Review")
            
            # Text input
            review_text = st.text_area(
                "Enter review text:",
                height=150,
                placeholder="Enter the review text here (e.g., 'This app is amazing, works perfectly! 😊')"
            )
            
            # Confidence threshold slider
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.05,
                help="Set the confidence threshold below which the sentiment is marked as UNCERTAIN."
            )
            
            if st.button("Analyze") and review_text:
                with st.spinner("Analyzing review... (this may take a few seconds)"):
                    # Get sentiment and probabilities
                    sentiment, probs, is_uncertain = analyze_sentiment(
                        text=review_text, 
                        model=model, 
                        vectorizer=vectorizer, 
                        n_features=n_features, 
                        confidence_threshold=confidence_threshold
                    )
                    
                    # Determine display sentiment
                    if is_uncertain:
                        sentiment_display = f"UNCERTAIN ({sentiment.upper()})"
                        sentiment_color = 'orange'
                    else:
                        sentiment_display = sentiment.upper()
                        sentiment_color = {
                            'positive': 'green',
                            'neutral': 'gray',
                            'negative': 'red'
                        }.get(sentiment, 'gray')
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Sentiment")
                        st.markdown(
                            f'<h1 style="color: {sentiment_color};">{sentiment_display}</h1>',
                            unsafe_allow_html=True
                        )
                    
                    with col2:
                        st.markdown("### Confidence Scores")
                        st.progress(probs[0], text=f"Negative: {probs[0]:.2%}")
                        st.progress(probs[1], text=f"Neutral: {probs[1]:.2%}")
                        st.progress(probs[2], text=f"Positive: {probs[2]:.2%}")
                    
                    if is_uncertain:
                        st.warning("⚠️ The model is uncertain about this review's sentiment. Please review manually.")
        
        else:  # Bulk Analysis
            st.subheader("Analyze Multiple Reviews")
            
            # File upload with better instructions
            uploaded_file = st.file_uploader(
                "Upload CSV file with reviews",
                type=['csv'],
                help="Upload a CSV file with a 'content' column containing the review texts. Max file size: 200MB"
            )
            
            if uploaded_file:
                with st.spinner("Processing reviews... This might take a while for large files."):
                    try:
                        # Read CSV
                        df = pd.read_csv(uploaded_file)
                        
                        if 'content' not in df.columns:
                            st.error("⚠️ Your CSV file must have a 'content' column containing the review texts. Please check your file format.")
                            return
                        
                        # Confidence threshold for bulk analysis
                        bulk_confidence_threshold = st.sidebar.slider(
                            "Bulk Analysis Confidence Threshold",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.6,
                            step=0.05,
                            help="Set the confidence threshold for bulk analysis."
                        )
                        
                        # Process each review with progress bar
                        results = []
                        total_reviews = len(df['content'])
                        
                        # Create progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for idx, text in enumerate(df['content']):
                            sentiment, probs, is_uncertain = analyze_sentiment(
                                text=text, 
                                model=model, 
                                vectorizer=vectorizer, 
                                n_features=n_features, 
                                confidence_threshold=bulk_confidence_threshold
                            )
                            if is_uncertain:
                                sentiment = f"UNCERTAIN ({sentiment})"
                            results.append({
                                'text': text,
                                'sentiment': sentiment,
                                'negative_prob': probs[0],
                                'neutral_prob': probs[1],
                                'positive_prob': probs[2]
                            })
                            
                            # Update progress
                            progress = (idx + 1) / total_reviews
                            progress_bar.progress(progress)
                            status_text.text(f"Processed {idx+1} of {total_reviews} reviews...")
                        
                        results_df = pd.DataFrame(results)
                        
                        # Clear progress bar and status
                        progress_bar.empty()
                        status_text.empty()
                        
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
                            
                            # Add percentage distribution
                            st.markdown("#### Percentage Distribution")
                            sentiment_percentages = (sentiment_counts / len(results_df) * 100).round(1)
                            for sentiment, percentage in sentiment_percentages.items():
                                st.write(f"{sentiment}: {percentage}%")
                        
                        # Show detailed results
                        st.markdown("### Detailed Results")
                        st.dataframe(results_df)
                        
                        # Download results
                        st.download_button(
                            "📥 Download Results",
                            results_df.to_csv(index=False),
                            "analyzed_reviews.csv",
                            "text/csv",
                            key='download-csv'
                        )
                        
                    except Exception as e:
                        st.error(f"Error processing file: {str(e)}")
                        st.error("Please make sure your CSV file is properly formatted and not corrupted.")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please make sure all required files are present and try again.")

if __name__ == "__main__":
    main()
