
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from textblob import TextBlob

# Project folders
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"

# Load saved data
corpus_df = pd.read_pickle(DATA_DIR / "corpus.pkl")

with open(DATA_DIR / "documents.json", "r", encoding="utf-8") as file:
    documents = json.load(file)

with open(MODELS_DIR / "bm25_model.pkl", "rb") as file:
    bm25_model = pickle.load(file)

embeddings = np.load(MODELS_DIR / "document_embeddings.npy")

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# Abbreviations
abbreviations = {
    "ai": "artificial intelligence",
    "cv": "computer vision",
    "cnn": "convolutional neural network",
    "gan": "generative adversarial network",
    "vit": "vision transformer",
    "yolo": "you only look once"
}

def process_query(query):
    phrases = re.findall(r'"([^"]+)"', query)
    query = query.lower()
    words = query.split()
    expanded_words = []

    for word in words:
        if word in abbreviations:
            expanded_words.append(abbreviations[word])
        else:
            expanded_words.append(word)

    query = " ".join(expanded_words)
    query = str(TextBlob(query).correct())

    return query, phrases

def search_papers(query, method="Hybrid", top_k=10):
    processed_query, phrases = process_query(query)

    query_tokens = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        processed_query.lower()
    )

    bm25_scores = bm25_model.get_scores(query_tokens)

    query_embedding = embedding_model.encode(
        [processed_query],
        normalize_embeddings=True
    )

    dense_scores = np.dot(embeddings, query_embedding[0])

    if method == "BM25":
        final_scores = bm25_scores

    elif method == "Dense":
        final_scores = dense_scores

    else:
        bm25_ranking = np.argsort(bm25_scores)[::-1]
        dense_ranking = np.argsort(dense_scores)[::-1]

        rrf_scores = {}
        k = 60

        for rank, doc_index in enumerate(bm25_ranking):
            rrf_scores[doc_index] = 1 / (k + rank + 1)

        for rank, doc_index in enumerate(dense_ranking):
            rrf_scores[doc_index] = (
                rrf_scores.get(doc_index, 0) +
                1 / (k + rank + 1)
            )

        final_scores = np.zeros(len(documents))

        for doc_index, score in rrf_scores.items():
            final_scores[doc_index] = score

    # Phrase matching boost
    for doc_index, document in enumerate(documents):
        for phrase in phrases:
            if phrase.lower() in document.lower():
                final_scores[doc_index] += 0.01

    final_ranking = np.argsort(final_scores)[::-1]

    results = []

    for doc_index in final_ranking[:top_k]:
        paper = corpus_df.iloc[doc_index]

        results.append({
            "title": paper["title"],
            "arxiv_id": paper["arxiv_id"],
            "link": paper["link"],
            "subtopic": paper["subtopic"],
            "score": round(float(final_scores[doc_index]), 5),
            "snippet": documents[doc_index][:200]
        })

    return pd.DataFrame(results)

# Streamlit interface
st.set_page_config(
    page_title="arXiv Hybrid Search",
    page_icon="📚",
    layout="wide"
)

st.title("arXiv Research Paper Search")
st.write("Search research papers using BM25, Dense Retrieval, or Hybrid Search.")

query = st.text_input("Enter your search query:")

method = st.selectbox(
    "Choose search method:",
    ["BM25", "Dense", "Hybrid"]
)

if st.button("Search"):
    if query.strip():

        with st.spinner("Searching papers..."):
            results = search_papers(query, method, 10)

        st.subheader("Top 10 Results")

        for index, paper in results.iterrows():
            st.markdown(f"### {index + 1}. {paper['title']}")
            st.write("Sub-topic:", paper["subtopic"])
            st.write("arXiv ID:", paper["arxiv_id"])
            st.write("Score:", paper["score"])
            st.write(paper["snippet"] + "...")
            st.markdown(f"[Open paper on arXiv]({paper['link']})")
            st.divider()

    else:
        st.warning("Please enter a search query.")
