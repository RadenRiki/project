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

def extract_features(text):
    """Extract additional features from text"""
    features = {}
    
    # Convert to string if not already
    text = str(text).lower()
    
    # Sentiment word counts
    positive_words = ['good', 'great', 'awesome', 'excellent', 'love', 'perfect']
    negative_words = ['bad', 'terrible', 'worst', 'poor', 'horrible', 'waste']
    neutral_words = ['okay', 'average', 'moderate', 'decent', 'normal', 'fine']
    
    features['positive_words'] = sum(word in text for word in positive_words)
    features['negative_words'] = sum(word in text for word in negative_words)
    features['neutral_words'] = sum(word in text for word in neutral_words)
    
    # Text statistics
    features['text_length'] = len(text)
    features['word_count'] = len(text.split())
    features['avg_word_length'] = features['text_length'] / (features['word_count'] + 1)
    
    return pd.DataFrame([features])

def analyze_sentiment(text, model, vectorizer):
    """Analyze sentiment of a single text"""
    # Preprocess text
    processed_text, _ = preprocess_text(text)
    
    # Vectorize the text
    text_vec = vectorizer.transform([processed_text])
    
    # Extract additional features
    extra_features = extract_features(processed_text)
    
    # Combine features
    features = sparse.hstack([text_vec, extra_features])
    
    # Get prediction and probabilities
    sentiment = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    return sentiment, probabilities

# Load model and vectorizer
@st.cache_resource
def load_model():
    try:
        model = joblib.load('../models/sentiment_model.joblib')
        vectorizer = joblib.load('../models/vectorizer.joblib')
        feature_names = joblib.load('../models/feature_names.joblib')
        return model, vectorizer, feature_names
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

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
        model, vectorizer = load_model()
        
        if model is None or vectorizer is None:
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
                    sentiment, probs = analyze_sentiment(review_text, model, vectorizer)
                    
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
                        sentiment, probs = analyze_sentiment(text, model, vectorizer)
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