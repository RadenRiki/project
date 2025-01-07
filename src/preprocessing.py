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
            # Check for positive/negative words
            positive_words = {'good', 'great', 'nice', 'super', 'perfect', 'awesome'}
            negative_words = {'bad', 'poor', 'terrible', 'worst'}
            
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

def validate_sentiment(text, score):
    """
    Validate if the sentiment score matches the content
    Returns corrected score if necessary
    """
    # List of negative keywords
    negative_keywords = ['not', 'cannot', 'cant', 'can\'t', 'problem', 'issue', 
                        'bad', 'poor', 'terrible', 'horrible', 'fail', 'bug', 
                        'crash', 'error', 'not working', 'doesn\'t work']
    
    text_lower = text.lower()
    
    # Check if high score (4-5) contains negative keywords
    if score >= 4 and any(keyword in text_lower for keyword in negative_keywords):
        return 2  # Adjust to negative sentiment
    
    return score

def prepare_data(df):
    # Copy the dataframe
    df_clean = df.copy()
    
    # Clean the text content and get language
    processed_data = df_clean['content'].apply(preprocess_text)
    df_clean['cleaned_content'] = processed_data.apply(lambda x: x[0])
    df_clean['language'] = processed_data.apply(lambda x: x[1])
    
    # Validate and correct sentiment scores
    df_clean['corrected_score'] = df_clean.apply(
        lambda x: validate_sentiment(x['content'], x['score']), axis=1
    )
    
    # Convert scores to sentiment categories using corrected scores
    df_clean['sentiment'] = pd.cut(df_clean['corrected_score'], 
                                 bins=[0, 2, 3, 5], 
                                 labels=['negative', 'neutral', 'positive'])
    
    # Remove empty content after cleaning
    df_clean = df_clean[df_clean['cleaned_content'] != ""]
    
    return df_clean