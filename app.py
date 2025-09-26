import os
import csv
import traceback
import datetime
import threading
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS

from google import genai
from google.genai.types import HttpOptions
from google.cloud import storage

# --- Flask App ---
app = Flask(__name__)
CORS(app, origins="*")  # in production, restrict to your frontend domain

# --- Env Vars ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set.")

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
LOG_BUCKET = os.environ.get("LOG_BUCKET", "my-chat-logs")

USE_KEYWORD_RETRIEVAL = os.environ.get("USE_KEYWORD_RETRIEVAL", "0") == "1"

# --- Gemini Client ---
client = genai.Client(api_key=GEMINI_API_KEY, http_options=HttpOptions(api_version="v1"))

# --- Load Knowledge Base (CSV) ---
documents = []
with open("knowledge_base.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["content"].strip():
            documents.append(row["content"].strip())

print(f"✅ Loaded {len(documents)} documents from knowledge_base.csv")

# --- Embeddings ---
embed_model = None
doc_embeddings = None

def load_embeddings():
    """Preload embedding model + compute doc embeddings in background"""
    global embed_model, doc_embeddings
    try:
        if not USE_KEYWORD_RETRIEVAL:
            print("⚡ Preloading embedding model...")
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            doc_embeddings = embed_model.encode(documents, convert_to_numpy=True)
            print("✅ Embeddings ready")
        else:
            print("ℹ️ Using keyword retrieval (no embeddings)")
    except Exception as e:
        print("❌ Failed to load embeddings:", e)
        USE_KEYWORD_RETRIEVAL = True

# Start background preload
threading.Thread(target=load_embeddings, daemon=True).start()

# --- Retrieval ---
def retrieve_docs(query, top_k=5):
    global embed_model, doc_embeddings

    if USE_KEYWORD_RETRIEVAL:
        q_words = set(query.lower().split())
        scores = []
        for d in documents:
            d_words = set(d.lower().split())
            scores.append(len(q_words & d_words))
        idx = np.argsort(scores)[::-1][:top_k]
        return [documents[i] for i in idx if scores[i] > 0] or [documents[0]]
    else:
        if embed_model is None or doc_embeddings is None:
            load_embeddings()  # lazy fallback
        from sklearn.metrics.pairwise import cosine_similarity
        q_emb = embed_model.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(q_emb, doc_embeddings)[0]
        idx = sims.argsort()[::-1][:top_k]
        return [documents[i] for i in idx]

# --- Cloud Storage Logging ---
storage_client = storage.Client()

def log_chat(user_message, response):
    ts = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
    filename = f"logs/chat-{ts}.txt"

    bucket = storage_client.bucket(LOG_BUCKET)
    blob = bucket.blob(filename)

    log_text = f"Q: {user_message}\nA: {response}\n"
    blob.upload_from_string(log_text, content_type="text/plain")

    print(f"✅ Logged chat to {filename}")

# --- Routes ---
@app.route("/")
def index():
    return "✅ Service is running"

@app.route("/test-gemini", methods=["GET"])
def test_gemini():
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents="Hello! Please reply with a one-line confirmation and include the model name."
        )
        text = getattr(resp, "text", None)
        if not text:
            text = resp.candidates[0].content.parts[0].text
        return jsonify({"ok": True, "model": MODEL, "response": text})
    except Exception as e:
        traceback_str = traceback.format_exc()
        return jsonify({"ok": False, "error": str(e), "trace": traceback_str}), 500

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "no message provided"}), 400

        docs = retrieve_docs(user_message, top_k=5)
        context = "\n\n".join(docs)
        prompt = (
            "You are a helpful assistant. Use the following documents to answer the question.\n\n"
            f"Documents:\n{context}\n\nQuestion: {user_message}\nAnswer:"
        )

        resp = client.models.generate_content(model=MODEL, contents=prompt)
        text = getattr(resp, "text", None)
        if not text:
            text = resp.candidates[0].content.parts[0].text

        # Log to Cloud Storage
        log_chat(user_message, text)

        return jsonify({"response": text})
    except Exception as e:
        traceback_str = traceback.format_exc()
        return jsonify({"error": str(e), "trace": traceback_str}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 Starting app on 0.0.0.0:{port}, GEMINI_MODEL={MODEL}, USE_KEYWORD_RETRIEVAL={USE_KEYWORD_RETRIEVAL}")
    app.run(host="0.0.0.0", port=port)
