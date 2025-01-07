import pandas as pd
from preprocessing import prepare_data

# Baca dataset
df = pd.read_csv('../data/all_combined.csv')

# Ambil sample kecil dulu untuk testing (misalnya 5 baris)
df_sample = df.head()

# Print data original
print("=== DATA ORIGINAL ===")
for _, row in df_sample.iterrows():
    print(f"Content: {row['content']}")
    print(f"Score: {row['score']}")
    print("-" * 50)

# Preprocess data
df_clean = prepare_data(df_sample)

# Print hasil preprocessing
print("\n=== HASIL PREPROCESSING ===")
for _, row in df_clean.iterrows():
    print(f"Original: {row['content']}")
    print(f"Cleaned: {row['cleaned_content']}")
    print(f"Sentiment: {row['sentiment']}")
    print("-" * 50)