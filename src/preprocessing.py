import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import emoji
from langdetect import detect

# Download required NLTK data
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def detect_language(text):
    try:
        # Clean text first
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = emoji.demojize(text)
        text = text.strip()
        
        # Handle short text differently
        if len(text.split()) < 3:
            # Extended positive/negative words for short text
            positive_words = {'good', 'great', 'nice', 'super', 'perfect', 'awesome', 'excellent', 
                            'amazing', 'love', 'best', 'fantastic', 'wonderful', 'brilliant'}
            negative_words = {'bad', 'poor', 'terrible', 'worst', 'horrible', 'awful', 'useless', 
                            'waste', 'garbage', 'rubbish', 'hate'}
            
            if text.lower() in positive_words or text.lower() in negative_words:
                return 'en'
            
            # Check for specific character sets
            if re.search(r'[\u0600-\u06FF]', text):  # Arabic
                return 'ar'
            elif re.search(r'[\u0980-\u09FF]', text):  # Bengali
                return 'bn'
                
        # For longer text
        # Strong English indicators
        english_patterns = [
            r'\b(facebook|instagram|whatsapp|app)\b',
            r'\b(account|login|password)\b',
            r'\b(the|and|or|is|are|was|were)\b',
            r'\b(i|you|he|she|it|we|they)\b'
        ]
        
        text_lower = text.lower()
        if any(re.search(pattern, text_lower) for pattern in english_patterns):
            return 'en'
            
        try:
            detected = detect(text)
            # Verify detection for common misclassifications
            if detected in ['da', 'so', 'pl'] and re.search(r'[a-zA-Z]', text):
                # If text contains English characters, double check
                if any(re.search(pattern, text_lower) for pattern in english_patterns):
                    return 'en'
            return detected
        except:
            if re.search(r'[\u0600-\u06FF]', text):
                return 'ar'
            elif re.search(r'[\u0980-\u09FF]', text):
                return 'bn'
            return 'unknown'
            
    except Exception as e:
        print(f"Language detection error: {str(e)}")
        return 'unknown'

def preprocess_text(text):
    try:
        # Convert to string if not already
        text = str(text)
        
        # Detect language
        lang = detect_language(text)
        
        # Convert emojis to text
        text = emoji.demojize(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        if lang == 'en':  # Process English text
            # Remove punctuation and special characters
            text = re.sub(r'[^\w\s]', ' ', text)
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove numbers
            text = re.sub(r'\d+', '', text)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            # Lemmatization for English text
            lemmatizer = WordNetLemmatizer()
            words = text.split()
            text = ' '.join([lemmatizer.lemmatize(word) for word in words])
        
        else:  # For non-English text, do minimal processing
            # Just remove extra whitespace and special characters
            text = re.sub(r'[^\w\s]', ' ', text)
            text = ' '.join(text.split())
        
        return text, lang
    except:
        return "", "unknown"

def score_to_sentiment(row):
    """
    Convert score to sentiment using both score and text content
    """
    try:
        score = row['score']
        text = str(row['content']).lower()
        
        # Strong sentiment indicators
        strong_positive = ['amazing', 'perfect', 'excellent', 'love', 'best', 'fantastic',
                         'awesome', 'brilliant', 'outstanding', 'superb', 'wonderful']
        strong_negative = ['terrible', 'worst', 'horrible', 'waste', 'useless', 'garbage',
                         'awful', 'pathetic', 'rubbish', 'hate', 'scam']
        neutral_indicators = ['okay', 'average', 'decent', 'alright', 'fair', 'moderate']
        
        # Count sentiment words
        positive_count = sum(word in text for word in strong_positive)
        negative_count = sum(word in text for word in strong_negative)
        neutral_count = sum(word in text for word in neutral_indicators)
        
        # Detect contradictions
        contradictions = ['but', 'however', 'although', 'though', 'despite', 'except']
        has_contradiction = any(word in text for word in contradictions)
        
        # Rule-based classification
        if score >= 4 and any(word in text for word in strong_negative):
            return 'negative'
        elif score <= 2 and any(word in text for word in strong_positive):
            return 'positive'
        elif has_contradiction:
            # For contradictory statements, rely more on text than score
            if positive_count > negative_count:
                return 'positive'
            elif negative_count > positive_count:
                return 'negative'
            else:
                return 'neutral'
        elif score >= 4:
            return 'positive'
        elif score <= 2:
            return 'negative'
        elif neutral_count > 0 or (positive_count == negative_count):
            return 'neutral'
        else:
            return 'neutral'  # Default case
            
    except Exception as e:
        print(f"Error in score_to_sentiment: {str(e)}")
        if pd.isna(score):
            return 'neutral'
        elif score >= 4:
            return 'positive'
        elif score <= 2:
            return 'negative'
        else:
            return 'neutral'

def prepare_data(df):
    """
    Prepare data for sentiment analysis
    """
    # Copy the dataframe
    df_clean = df.copy()
    
    # Clean the text content and get language
    processed_data = df_clean['content'].apply(preprocess_text)
    df_clean['cleaned_content'] = processed_data.apply(lambda x: x[0])
    df_clean['language'] = processed_data.apply(lambda x: x[1])
    
    # Convert scores to sentiment using improved logic
    df_clean['sentiment'] = df_clean.apply(score_to_sentiment, axis=1)
    
    # Remove empty content after cleaning
    df_clean = df_clean[df_clean['cleaned_content'] != ""]
    
    # Add additional features
    df_clean['text_length'] = df_clean['cleaned_content'].str.len()
    df_clean['word_count'] = df_clean['cleaned_content'].str.split().str.len()
    
    return df_clean