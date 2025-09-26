import os
import traceback
from flask import Flask, request, jsonify
from google import genai
from google.genai.types import HttpOptions
from flask_cors import CORS
import numpy as np

# Optional: sentence-transformers embedding (may download weights)
USE_KEYWORD_RETRIEVAL = os.environ.get("USE_KEYWORD_RETRIEVAL", "0") == "1"
try:
    if not USE_KEYWORD_RETRIEVAL:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
except Exception as e:
    print("Could not import sentence-transformers or sklearn. Falling back to keyword retrieval.")
    print(e)
    USE_KEYWORD_RETRIEVAL = True

app = Flask(__name__)
CORS(app, origins="*")  # in production restrict origins to your GitHub Pages URL

# Required env var
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set. See README.")

# Model selection (default to the working one you found)
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Initialize client (force v1 API)
client = genai.Client(api_key=GEMINI_API_KEY, http_options=HttpOptions(api_version="v1"))

# Small example knowledge base (replace/expand in production)
documents = [
    "Product X warranty lasts 2 years.",
    "Product Y is waterproof and can be used outdoors.",
    "Support is available 24/7 via email support@example.com",
]

# Load embed model unless using keyword retrieval
if not USE_KEYWORD_RETRIEVAL:
    print("Loading embedding model (sentence-transformers). This may take a bit...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    # returns numpy array
    doc_embeddings = embed_model.encode(documents, convert_to_numpy=True)
else:
    embed_model = None
    doc_embeddings = None
    print("Using lightweight keyword retrieval (USE_KEYWORD_RETRIEVAL=1)")

def retrieve_docs(query, top_k=2):
    """Return top_k most relevant documents."""
    if USE_KEYWORD_RETRIEVAL:
        # simple keyword overlap scoring (fast, no downloads)
        q_words = set(query.lower().split())
        scores = []
        for d in documents:
            d_words = set(d.lower().split())
            scores.append(len(q_words & d_words))
        idx = np.argsort(scores)[::-1][:top_k]
        return [documents[i] for i in idx if scores[i] > 0] or [documents[0]]
    else:
        q_emb = embed_model.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(q_emb, doc_embeddings)[0]
        idx = sims.argsort()[::-1][:top_k]
        return [documents[i] for i in idx]

@app.route("/")
def index():
    return "✅ Service is running"

@app.route("/test-gemini", methods=["GET"])
def test_gemini():
    """Quick endpoint to confirm Gemini is reachable and returns text."""
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents="Hello! Please reply with a one-line confirmation and include the model name."
        )
        # prefer .text if available, else try the structured path
        text = getattr(resp, "text", None)
        if not text:
            try:
                text = resp.candidates[0].content.parts[0].text
            except Exception:
                text = str(resp)
        return jsonify({"ok": True, "model": MODEL, "response": text})
    except Exception as e:
        traceback_str = traceback.format_exc()
        print("Error calling Gemini:\n", traceback_str)
        return jsonify({"ok": False, "error": str(e), "trace": traceback_str}), 500

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "no message provided"}), 400

        docs = retrieve_docs(user_message, top_k=3)
        context = "\n\n".join(docs)
        prompt = (
            "You are a helpful assistant. Use the following documents to answer the question.\n\n"
            f"Documents:\n{context}\n\nQuestion: {user_message}\nAnswer:"
        )

        resp = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )

        text = getattr(resp, "text", None)
        if not text:
            try:
                text = resp.candidates[0].content.parts[0].text
            except Exception:
                text = str(resp)

        return jsonify({"response": text})
    except Exception as e:
        traceback_str = traceback.format_exc()
        print("Error in /chat handler:\n", traceback_str)
        return jsonify({"error": str(e), "trace": traceback_str}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting app on 0.0.0.0:{port}, GEMINI_MODEL={MODEL}, USE_KEYWORD_RETRIEVAL={USE_KEYWORD_RETRIEVAL}")
    app.run(host="0.0.0.0", port=port)
