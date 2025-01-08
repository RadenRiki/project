import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import emoji
from langdetect import detect
from fuzzywuzzy import fuzz
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Multi-language sentiment dictionaries
SENTIMENT_WORDS = {
    'en': {
        'positive': [
            'amazing', 'perfect', 'excellent', 'love', 'best', 'fantastic',
            'awesome', 'brilliant', 'outstanding', 'superb', 'wonderful',
            'enjoying', 'enjoyable', 'enjoyed', 'great', 'fun', 'good',
            'happy', 'helpful', 'nice', 'beautiful', 'impressive'
        ],
        'negative': [
            'terrible', 'worst', 'horrible', 'waste', 'useless', 'garbage',
            'awful', 'pathetic', 'rubbish', 'hate', 'scam', 'bad',
            'poor', 'disappointing', 'frustrated', 'annoying', 'broken'
        ],
        'neutral': [
            'okay', 'average', 'decent', 'alright', 'fair', 'moderate',
            'ok', 'meh', 'standard', 'normal', 'simple'
        ]
    },
    'id': {
        'positive': [
            'bagus', 'baik', 'mantap', 'keren', 'suka', 'enak', 'sempurna',
            'hebat', 'jos', 'oke', 'mantab', 'recommended', 'berguna'
        ],
        'negative': [
            'buruk', 'jelek', 'hancur', 'rusak', 'payah', 'buang', 'sampah',
            'gagal', 'parah', 'lambat', 'error', 'lemot', 'benci'
        ],
        'neutral': [
            'biasa', 'lumayan', 'cukup', 'standar', 'sedang', 'netral'
        ]
    },
    'ar': {
        'positive': [
            'جيد', 'ممتاز', 'رائع', 'عظيم', 'حلو', 'جميل', 'احب', 'افضل'
        ],
        'negative': [
            'سيء', 'رديء', 'خطأ', 'مشكلة', 'بطيء', 'فشل', 'كره', 'اسوأ'
        ],
        'neutral': [
            'معتدل', 'متوسط', 'عادي', 'مقبول'
        ]
    }
}

# Common typos and variants dictionary
TYPOS_DICT = {
    'gud': 'good',
    'gd': 'good',
    'gr8': 'great',
    'supar': 'super',
    'owsum': 'awesome',
    'awesum': 'awesome',
    'oke': 'okay',
    'k': 'okay',
    'oky': 'okay',
    'vry': 'very',
    'prblm': 'problem',
    'prob': 'problem',
    'gud': 'good',
    'gj': 'good job',
    'gg': 'good game',
}

def fuzzy_match_word(word, word_list, threshold=85):
    """Menggunakan fuzzy matching untuk menemukan kata yang mirip"""
    matches = []
    for known_word in word_list:
        ratio = fuzz.ratio(word.lower(), known_word.lower())
        if ratio > threshold:
            matches.append((known_word, ratio))
    return max(matches, key=lambda x: x[1])[0] if matches else word

def handle_typos_and_variants(text):
    """Menangani typo dan variasi kata"""
    words = text.lower().split()
    corrected_words = []
    
    for word in words:
        # Check common typos first
        if word in TYPOS_DICT:
            corrected_words.append(TYPOS_DICT[word])
            continue
            
        # If not in typos dict, try fuzzy matching with sentiment words
        all_sentiment_words = set()
        for sentiment_lists in SENTIMENT_WORDS['en'].values():  # Using English as base
            all_sentiment_words.update(sentiment_lists)
            
        if len(word) > 2:  # Only try to match words longer than 2 characters
            matched = fuzzy_match_word(word, all_sentiment_words)
            corrected_words.append(matched)
        else:
            corrected_words.append(word)
            
    return ' '.join(corrected_words)

def detect_language(text):
    try:
        # Clean text first
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = emoji.demojize(text)
        text = text.strip()
        
        # Handle short text differently
        if len(text.split()) < 3:
            # Check in all language dictionaries
            text_lower = text.lower()
            for lang, sent_dict in SENTIMENT_WORDS.items():
                for word_list in sent_dict.values():
                    if any(word in text_lower for word in word_list):
                        return lang
            
            # Check for specific character sets
            if re.search(r'[\u0600-\u06FF]', text):  # Arabic
                return 'ar'
            elif re.search(r'[\u0980-\u09FF]', text):  # Bengali
                return 'bn'
                
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
            if detected in ['da', 'so', 'pl'] and re.search(r'[a-zA-Z]', text):
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
        logger.error(f"Language detection error: {str(e)}")
        return 'unknown'

def handle_emojis(text):
    """Menangani emoji dengan mapping yang lebih lengkap"""
    # Convert emoji to text equivalent
    text = emoji.demojize(text)
    
    # Extended emoji mapping
    emoji_mapping = {
        # Positive emojis
        ':thumbs_up:': 'good',
        ':smiling_face:': 'happy',
        ':grinning_face:': 'happy',
        ':beaming_face_with_smiling_eyes:': 'very happy',
        ':star_struck:': 'amazing',
        ':heart:': 'love',
        ':red_heart:': 'love',
        ':folded_hands:': 'thank',
        ':clapping_hands:': 'good',
        
        # Negative emojis
        ':thumbs_down:': 'bad',
        ':crying_face:': 'sad',
        ':angry_face:': 'angry',
        ':face_with_symbols_on_mouth:': 'angry',
        ':pouting_face:': 'angry',
        ':confused_face:': 'confused',
        ':broken_heart:': 'bad',
        ':face_with_steam_from_nose:': 'angry',
        ':skull:': 'terrible',
        
        # Neutral emojis
        ':neutral_face:': 'okay',
        ':thinking_face:': 'thinking',
        ':face_without_mouth:': 'neutral',
    }
    
    # Replace emojis with their text equivalents
    for emoji_text, replacement in emoji_mapping.items():
        text = text.replace(emoji_text, f' {replacement} ')
    
    return text

def analyze_context(text):
    """Analisis konteks untuk sentiment yang lebih akurat"""
    context = {
        'has_intensifier': False,
        'has_negation': False,
        'has_contradiction': False,
        'sentiment_shifts': 0
    }
    
    # Identify sentence parts
    sentences = text.split('.')
    for sentence in sentences:
        words = sentence.split()
        
        # Check for intensifiers
        intensifiers = ['very', 'really', 'so', 'too', 'extremely', 'quite']
        context['has_intensifier'] = any(word in intensifiers for word in words)
        
        # Check for negations
        negations = ['not', 'no', 'never', "n't", 'neither', 'nor']
        context['has_negation'] = any(word in negations for word in words)
        
        # Check for contradictions
        contradictions = ['but', 'however', 'although', 'though', 'despite', 'yet']
        context['has_contradiction'] = any(word in contradictions for word in words)
        
    return context

def preprocess_text(text):
    try:
        # Convert to string if not already
        text = str(text)
        
        # Handle typos and variants first
        text = handle_typos_and_variants(text)
        
        # Handle short text
        if len(text.split()) <= 3:
            text = handle_short_text(text)
        
        # Detect language
        lang = detect_language(text)
        
        # Handle emojis
        text = handle_emojis(text)
        
        # Remove URLs and usernames
        text = re.sub(r'http\S+|www\S+|https\S+|@\w+', '', text, flags=re.MULTILINE)
        
        # Language specific processing
        if lang in SENTIMENT_WORDS:  # If we have sentiment words for this language
            # Remove punctuation and special characters
            text = re.sub(r'[^\w\s]', ' ', text)
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove numbers but keep them if they're part of common expressions (e.g., "5star")
            text = re.sub(r'\b\d+\b', '', text)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            if lang == 'en':
                # Lemmatization for English text
                lemmatizer = WordNetLemmatizer()
                words = text.split()
                text = ' '.join([lemmatizer.lemmatize(word) for word in words])
        
        else:  # For languages without specific processing rules
            # Minimal processing
            text = re.sub(r'[^\w\s]', ' ', text)
            text = ' '.join(text.split())
        
        return text, lang
    except Exception as e:
        logger.error(f"Text preprocessing error: {str(e)}")
        return "", "unknown"

def handle_short_text(text):
    """Menangani teks pendek dengan lebih baik"""
    text = text.lower()
    
    # Extended short text mapping
    short_text_mapping = {
        # Positive
        'gj': 'good job',
        'gg': 'good game',
        'nice': 'good',
        'cool': 'good',
        'gud': 'good',
        'thx': 'thank you',
        'ty': 'thank you',
        '👍': 'good',
        '❤️': 'love',
        
        # Negative
        'meh': 'bad',
        'nah': 'bad',
        'ugh': 'bad',
        'wtf': 'bad',
        'bs': 'bad',
        '👎': 'bad',
        
        # Neutral
        'k': 'okay',
        'kk': 'okay',
        'hmm': 'neutral',
        'eh': 'neutral'
    }
    
    # Check if text matches any mapping
    if text in short_text_mapping:
        return short_text_mapping[text]
        
    # If no exact match, try fuzzy matching
    for short_form, full_form in short_text_mapping.items():
        if fuzz.ratio(text, short_form) > 85:
            return full_form
            
    return text

def get_sentiment_from_text(text, lang):
    """Get sentiment based purely on text analysis"""
    if lang not in SENTIMENT_WORDS:
        lang = 'en'  # fallback to English
        
    sentiment_dict = SENTIMENT_WORDS[lang]
    
    # Count sentiment words
    positive_count = sum(word in text.split() for word in sentiment_dict['positive'])
    negative_count = sum(word in text.split() for word in sentiment_dict['negative'])
    neutral_count = sum(word in text.split() for word in sentiment_dict['neutral'])
    
    # Get context
    context = analyze_context(text)
    
    # Adjust counts based on context
    if context['has_negation']:
        # Swap positive and negative counts
        positive_count, negative_count = negative_count, positive_count
        
    if context['has_intensifier']:
        # Intensify the dominant sentiment
        if positive_count > negative_count:
            positive_count *= 1.5
        elif negative_count > positive_count:
            negative_count *= 1.5
            
    # Calculate sentiment
    total_words = len(text.split())
    positive_ratio = positive_count / total_words if total_words > 0 else 0
    negative_ratio = negative_count / total_words if total_words > 0 else 0
    neutral_ratio = neutral_count / total_words if total_words > 0 else 0
    
    # Return sentiment with confidence
    max_ratio = max(positive_ratio, negative_ratio, neutral_ratio)
    if max_ratio < 0.1:  # If no strong sentiment indicators
        return 'neutral', 0.5
    elif positive_ratio == max_ratio:
        return 'positive', positive_ratio
    elif negative_ratio == max_ratio:
        return 'negative', negative_ratio
    else:
        return 'neutral', neutral_ratio

def score_to_sentiment(row):
    try:
        text = str(row['content']).lower()
        score = row['score']
        # Get language
        _, lang = preprocess_text(text)
        
        # Get text-based sentiment first
        text_sentiment, confidence = get_sentiment_from_text(text, lang)
        
        # If we have high confidence in text analysis, use that
        if confidence > 0.7:
            return text_sentiment
            
        # Deteksi negasi
        negation_words = ['not', 'no', "n't", 'never', 'none', 'nothing', 'neither', 'nowhere', 'cannot']
        contains_negation = any(neg in text.split() for neg in negation_words)
        
        # Intensifiers
        intensifiers = ['so', 'very', 'really', 'quite', 'too', 'super']
        
        # Deteksi problem indicators
        problem_indicators = ['problem', 'issue', 'bug', 'error', 'crash', 'broken', 'stuck', 'slow', 
                            'fail', 'failed', 'failing', 'wrong', 'bad', 'poor', 'terrible', 
                            'not working', 'doesn\'t work', 'won\'t work', 'cant', 'cannot']
        
        # Get sentiment dictionaries for the detected language
        if lang in SENTIMENT_WORDS:
            sentiment_dict = SENTIMENT_WORDS[lang]
            strong_positive = sentiment_dict['positive']
            strong_negative = sentiment_dict['negative']
            neutral_indicators = sentiment_dict['neutral']
        else:
            # Fallback to English
            sentiment_dict = SENTIMENT_WORDS['en']
            strong_positive = sentiment_dict['positive']
            strong_negative = sentiment_dict['negative']
            neutral_indicators = sentiment_dict['neutral']
        
        # Count words with context
        positive_count = sum(word in text for word in strong_positive)
        negative_count = sum(word in text for word in strong_negative)
        neutral_count = sum(word in text for word in neutral_indicators)
        
        # Check for problems/issues first
        has_problems = any(indicator in text for indicator in problem_indicators)
        if has_problems:
            return 'negative'
            
        # Handle negation
        if contains_negation:
            # Check specific negated phrases
            if any(f"not {pos}" in text for pos in strong_positive):
                return 'negative'
            # Check for other negated contexts
            if positive_count > 0:  # If there are positive words but they're negated
                return 'negative'
        
        # Check for intensified sentiments
        has_intensifier = any(word in text.split() for word in intensifiers)
        if has_intensifier:
            words = text.split()
            for i, word in enumerate(words[:-1]):
                if word in intensifiers:
                    if i+1 < len(words):
                        next_word = words[i+1]
                        if next_word in strong_positive:
                            if not contains_negation:  # Make sure it's not negated
                                return 'positive'
                        if next_word in strong_negative:
                            return 'negative'
        
        # Detect contradictions with context
        contradictions = ['but', 'however', 'although', 'though', 'despite', 'except']
        if any(word in text.split() for word in contradictions):
            for contra in contradictions:
                if contra in text:
                    parts = text.split(contra)
                    if len(parts) > 1:
                        after_contra = parts[1]
                        pos_after = sum(word in after_contra for word in strong_positive)
                        neg_after = sum(word in after_contra for word in strong_negative)
                        if pos_after > neg_after:
                            return 'positive'
                        elif neg_after > pos_after:
                            return 'negative'
        
        # Use score as a final factor
        if score >= 4:
            if negative_count > positive_count * 2 or has_problems:  # Strong negative override
                return 'negative'
            if contains_negation and not any(pos in text for pos in strong_positive):
                return 'negative'
            return 'positive'
        elif score <= 2:
            if positive_count > negative_count * 2:  # Strong positive override
                return 'positive'
            return 'negative'
        
        # If we get here, use the counts
        if neutral_count > max(positive_count, negative_count):
            return 'neutral'
        elif positive_count == negative_count:
            return 'neutral'
        elif positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
            
    except Exception as e:
        logger.error(f"Error in score_to_sentiment: {str(e)}")
        return 'neutral'  # Default case for errors
        
def prepare_data(df):
    """
    Prepare data for sentiment analysis with improved logging and error handling
    """
    try:
        # Copy the dataframe
        df_clean = df.copy()
        
        # Clean the text content and get language
        logger.info("Processing text and detecting languages...")
        processed_data = df_clean['content'].apply(preprocess_text)
        df_clean['cleaned_content'] = processed_data.apply(lambda x: x[0])
        df_clean['language'] = processed_data.apply(lambda x: x[1])
        
        # Log language distribution
        lang_dist = df_clean['language'].value_counts()
        logger.info(f"Language distribution:\n{lang_dist}")
        
        # Convert scores to sentiment using improved logic
        logger.info("Converting scores to sentiment...")
        df_clean['sentiment'] = df_clean.apply(score_to_sentiment, axis=1)
        
        # Log sentiment distribution
        sent_dist = df_clean['sentiment'].value_counts()
        logger.info(f"Sentiment distribution:\n{sent_dist}")
        
        # Remove empty content after cleaning
        df_clean = df_clean[df_clean['cleaned_content'] != ""]
        
        # Add additional features
        df_clean['text_length'] = df_clean['cleaned_content'].str.len()
        df_clean['word_count'] = df_clean['cleaned_content'].str.split().str.len()
        
        # Calculate confidence scores
        logger.info("Calculating confidence scores...")
        confidences = []
        for _, row in df_clean.iterrows():
            _, confidence = get_sentiment_from_text(row['cleaned_content'], row['language'])
            confidences.append(confidence)
        df_clean['sentiment_confidence'] = confidences
        
        return df_clean
        
    except Exception as e:
        logger.error(f"Error in prepare_data: {str(e)}")
        raise