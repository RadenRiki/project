import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import emoji

# Download required NLTK data
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    try:
        # Convert to string if not already
        text = str(text)
        
        # Convert emojis to text
        text = emoji.demojize(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    except:
        return ""

def prepare_data(df):
    # Copy the dataframe
    df_clean = df.copy()
    
    # Clean the text content
    df_clean['cleaned_content'] = df_clean['content'].apply(preprocess_text)
    
    # Convert scores to sentiment categories
    df_clean['sentiment'] = pd.cut(df_clean['score'], 
                                 bins=[0, 2, 3, 5], 
                                 labels=['negative', 'neutral', 'positive'])
    
    # Remove empty content after cleaning
    df_clean = df_clean[df_clean['cleaned_content'] != ""]
    
    return df_clean