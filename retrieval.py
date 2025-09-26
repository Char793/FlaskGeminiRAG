import os
import threading
import numpy as np

USE_KEYWORD_RETRIEVAL = os.environ.get("USE_KEYWORD_RETRIEVAL", "0") == "1"

embed_model = None
doc_embeddings = None

def load_embeddings(documents):
    """Load embeddings for documents"""
    global embed_model, doc_embeddings
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = embed_model.encode(documents, convert_to_numpy=True)

def preload_embeddings(documents):
    """Background preload to avoid slow first request"""
    if not USE_KEYWORD_RETRIEVAL:
        threading.Thread(target=load_embeddings, args=(documents,), daemon=True).start()
    else:
        print("ℹ️ Keyword retrieval only (no embeddings)")

def retrieve_docs(query, documents, top_k=5):
    global embed_model, doc_embeddings

    if USE_KEYWORD_RETRIEVAL:
        q_words = set(query.lower().split())
        scores = [len(set(d.lower().split()) & q_words) for d in documents]
        idx = np.argsort(scores)[::-1][:top_k]
        return [documents[i] for i in idx if scores[i] > 0] or [documents[0]]

    else:
        if embed_model is None or doc_embeddings is None:
            load_embeddings(documents)

        from sklearn.metrics.pairwise import cosine_similarity
        q_emb = embed_model.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(q_emb, doc_embeddings)[0]
        idx = sims.argsort()[::-1][:top_k]
        return [documents[i] for i in idx]
