import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import os

print("Loading Tariff.xlsx...")
try:
    df = pd.read_excel('Tariff.xlsx')
except Exception as e:
    print(f"Error loading Excel: {e}")
    exit(1)

# Keep only necessary columns
cols_to_keep = ['رمز النظام المنسق \n Harmonized Code', 'الصنف باللغة العربية \n Item Arabic Name', 'الصنف باللغة الانجليزية \n Item English Name']
df = df[cols_to_keep].copy()

# Rename columns for easier access
df.columns = ['hs_code', 'arabic_name', 'english_name']

print(f"Original rows: {len(df)}")

# Clean HS Codes (they were loaded as floats, e.g., 1.010000e+10)
def clean_hs_code(code):
    if pd.isna(code):
        return ""
    try:
        # Convert float to int, then to string to avoid scientific notation
        return str(int(code))
    except:
        return str(code)

df['hs_code'] = df['hs_code'].apply(clean_hs_code)

# Clean text: remove leading dashes, extra spaces, handle NaNs
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    # Remove leading dashes and spaces
    text = re.sub(r'^[-\s]+', '', text)
    # Remove trailing colons/spaces
    text = re.sub(r'[:\s]+$', '', text)
    return text.strip()

print("Cleaning text data...")
df['arabic_name'] = df['arabic_name'].apply(clean_text)
df['english_name'] = df['english_name'].apply(clean_text)

# Drop rows where both names are empty
df = df[(df['arabic_name'] != '') | (df['english_name'] != '')].copy()

print(f"Rows after cleaning: {len(df)}")

# Create combined text for embedding
df['combined_text'] = df['english_name'] + " | " + df['arabic_name']

# Save cleaned dataframe
df.to_pickle('cleaned_tariff.pkl')
print("Saved cleaned_tariff.pkl")

# Load model and compute embeddings
model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
print(f"Loading model: {model_name}...")
model = SentenceTransformer(model_name)

print("Computing embeddings (this may take a few minutes depending on your hardware)...")
# Convert to list for the model
sentences = df['combined_text'].tolist()

# Compute embeddings
embeddings = model.encode(sentences, show_progress_bar=True, batch_size=64)

# Save embeddings
np.save('embeddings.npy', embeddings)
print("Saved embeddings.npy")
print("Data preparation complete!")
