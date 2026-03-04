import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Page Config
st.set_page_config(page_title="HS Code Classifier POC", page_icon="🌍", layout="wide")

st.title("🌍 AI-Powered HS Code Classifier (POC)")
st.markdown("""
This application classifies products into Harmonized System (HS) Codes.
It supports **Arabic** and **English** product descriptions, or you can enter a **GTIN barcode**.
""")

# Cache data loading so it's instantaneous after the first run
@st.cache_data
def load_data():
    try:
        df = pd.read_pickle('cleaned_tariff.pkl')
        embeddings = np.load('embeddings.npy')
        return df, embeddings
    except FileNotFoundError:
        st.error("Data files missing! Please ensure `cleaned_tariff.pkl` and `embeddings.npy` are uploaded to the Space.")
        st.stop()

# Cache model loading to avoid reloading on every interaction
@st.cache_resource
def load_model():
    model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        st.error(f"Failed to load sentence-transformer model: {e}")
        st.stop()

# 1. Initialization
df, tariff_embeddings = load_data()
model = load_model()

# 2. GTIN Resolution Logic
def resolve_gtin(gtin_code):
    """Fetches product name from OpenFoodFacts given a valid GTIN."""
    url = f"https://world.openfoodfacts.org/api/v0/product/{gtin_code}.json"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 1:
                # Prioritize English, then generic product_name
                product = data['product']
                name = product.get('product_name_en') or product.get('product_name')
                if name:
                    return name
        return None # Failed to find or parse name
    except requests.exceptions.RequestException:
        return None # API error

# 3. Main Search/Classification Logic
def search_hs_codes(query, top_k=5):
    """Embeds the query, calculates cosine similarity, and returns top K results."""
    # Compute query embedding
    query_emb = model.encode([query])
    
    # Calculate similarities (1 x N array)
    similarities = cosine_similarity(query_emb, tariff_embeddings)[0]
    
    # Get top K indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Retrieve top records and scores
    results = []
    for idx in top_indices:
        score = similarities[idx]
        record = df.iloc[idx]
        results.append({
            'HS Code': record['hs_code'],
            'English Description': record['english_name'],
            'Arabic Description': record['arabic_name'],
            'Confidence': float(score) # Convert to standard Python float for Streamlit formatting
        })
    return results

# 4. User Interface
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        user_input = st.text_input("Enter Product Description (Ar/En) or GTIN Barcode", placeholder="e.g., Roasted coffee or 5449000000996")
    with col2:
        st.write("") # Spacing
        st.write("")
        search_button = st.button("Classify Product", type="primary", use_container_width=True)

if search_button and user_input:
    query_text = user_input.strip()
    
    # Check if input looks like a GTIN (numeric, 8 to 14 digits)
    is_gtin = query_text.isdigit() and (8 <= len(query_text) <= 14)
    
    if is_gtin:
        with st.spinner("Looking up GTIN in public database..."):
            resolved_name = resolve_gtin(query_text)
            
            if resolved_name:
                st.success(f"Barcode found! Product identified as: **{resolved_name}**")
                query_text = resolved_name # Update query_text to the resolved name for the AI model
            else:
                st.error("GTIN not found in public database. Please enter a product description manually.")
                st.stop() # Stop execution if GTIN fails

    # Perform Classification
    with st.spinner("Classifying product..."):
        results = search_hs_codes(query_text)
        
        st.subheader(f"Top Candidate HS Codes for '{query_text}'")
        
        # Display results nicely
        for i, res in enumerate(results):
            # Convert raw score (e.g., 0.85) to percentage (85%)
            conf_percent = res['Confidence'] * 100
            
            with st.expander(f"**{res['HS Code']}** - {res['English Description']} ({conf_percent:.1f}% Confidence)", expanded=(i==0)):
                st.markdown(f"**HS Code:** `{res['HS Code']}`")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**English Description:**")
                    st.info(res['English Description'])
                with c2:
                    st.markdown("**Arabic Description (الوصف بالعربية):**")
                    st.success(res['Arabic Description'] if res['Arabic Description'] else "N/A")
                
                # Visual confidence bar
                st.progress(max(0.0, min(1.0, res['Confidence'])), text="Match Probability")

