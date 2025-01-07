import pandas as pd
from preprocessing import prepare_data
import pandas as pd
from collections import Counter

def test_preprocessing():
    # Baca dataset
    print("Loading dataset...")
    df = pd.read_csv('../data/all_combined.csv')
    
    # Ambil sample yang lebih beragam
    print("\nSelecting diverse samples...")
    sample_size = 3
    df_sample = pd.concat([
        # Reviews dengan berbagai score
        df[df['score'] == 1].head(sample_size),  # Negative
        df[df['score'] == 3].head(sample_size),  # Neutral
        df[df['score'] == 5].head(sample_size),  # Positive
        
        # Reviews dengan emoji
        df[df['content'].str.contains('😡|😊|👍', na=False)].head(sample_size),
        
        # Reviews dengan URL
        df[df['content'].str.contains('http|www', na=False)].head(sample_size),
        
        # Reviews dengan multiple languages (if available)
        df[df['content'].str.contains('[\u0600-\u06FF]', na=False)].head(sample_size),  # Arabic
        df[df['content'].str.contains('[\u0980-\u09FF]', na=False)].head(sample_size)   # Bengali
    ])

    # Remove duplicates
    df_sample = df_sample.drop_duplicates(subset=['content'])

    # Process data
    print("\nProcessing samples...")
    df_clean = prepare_data(df_sample)

    # Print detailed results
    print("\n=== PREPROCESSING RESULTS ===")
    for _, row in df_clean.iterrows():
        print("\nORIGINAL TEXT:")
        print(row['content'])
        print("\nPROCESSED:")
        print(f"Language: {row['language']}")
        print(f"Cleaned text: {row['cleaned_content']}")
        print(f"Original Score: {row['score']}")
        print(f"Corrected Score: {row['corrected_score']}")
        print(f"Final Sentiment: {row['sentiment']}")
        print("-" * 70)

    # Print statistics
    print("\n=== STATISTICS ===")
    print(f"Total samples processed: {len(df_clean)}")
    print("\nLanguage distribution:")
    lang_dist = Counter(df_clean['language'])
    for lang, count in lang_dist.items():
        print(f"{lang}: {count} ({count/len(df_clean)*100:.1f}%)")

    print("\nSentiment distribution:")
    sent_dist = Counter(df_clean['sentiment'])
    for sent, count in sent_dist.items():
        print(f"{sent}: {count} ({count/len(df_clean)*100:.1f}%)")

    print("\nScore correction stats:")
    score_changes = (df_clean['score'] != df_clean['corrected_score']).sum()
    print(f"Reviews with corrected scores: {score_changes} ({score_changes/len(df_clean)*100:.1f}%)")

if __name__ == "__main__":
    test_preprocessing()