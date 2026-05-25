from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Global model holder (lazy loaded)
_model = None


def get_model():
    """
    Load the SentenceTransformer model only once.
    This prevents FastAPI startup from blocking,
    especially when using uvicorn --reload on Windows.
    """
    global _model
    if _model is None:
        print("⏳ Loading SentenceTransformer model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ SentenceTransformer model loaded")
    return _model


def embed_text(texts):
    """
    Generate embeddings for a list of texts.
    """
    model = get_model()
    return model.encode(texts)


def search_vectors(query_embedding, product_embeddings, top_k=5):
    """
    Perform cosine similarity search and return top matches.
    """
    similarities = cosine_similarity([query_embedding], product_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]
    return top_indices, similarities[top_indices]
