import os
import traceback
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai.types import HttpOptions

# --- Flask App ---
app = Flask(__name__)
CORS(app, origins="*")  # restrict in production

# --- Env Vars ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set.")

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# --- Gemini Client ---
client = genai.Client(api_key=GEMINI_API_KEY, http_options=HttpOptions(api_version="v1"))

# --- Load Knowledge Base (CSV) ---
documents = []
df = pd.read_csv("knowledge_base.csv")
for _, row in df.iterrows():
    row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])
    documents.append(row_text)

print(f"✅ Loaded {len(documents)} documents from knowledge_base.csv")

# --- Embeddings ---
EMBED_FILE = "knowledge_base_embedded.npy"
doc_embeddings = None

def embed_texts(texts, batch_size=50):
    """Embed a list of texts in batches using text-embedding-004."""
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.models.embed_content(
            model="text-embedding-004",
            contents=batch
        )
        vectors.extend([e.values for e in resp.embeddings])
    return np.array(vectors)

def load_or_create_embeddings():
    global doc_embeddings
    if os.path.exists(EMBED_FILE):
        print("🔄 Loading cached embeddings...")
        doc_embeddings = np.load(EMBED_FILE)
    else:
        print("⚡ Precomputing embeddings with text-embedding-004...")
        doc_embeddings = embed_texts(documents)
        np.save(EMBED_FILE, doc_embeddings)
        print(f"✅ Embeddings saved to {EMBED_FILE}")

load_or_create_embeddings()

# --- Retrieval ---
def retrieve_docs(query, top_k=5):
    global doc_embeddings
    if doc_embeddings is None:
        load_or_create_embeddings()

    # Embed user query
    q_emb = embed_texts([query])[0]
    sims = np.dot(doc_embeddings, q_emb) / (np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-8)
    idx = sims.argsort()[::-1][:top_k]
    return [documents[i] for i in idx]

# --- Routes ---
@app.route("/")
def index():
    return "✅ Service is running"

@app.route("/test-gemini", methods=["GET"])
def test_gemini():
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents="Hello! Please reply with a one-line confirmation."
        )
        text = getattr(resp, "text", None)
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

        docs = retrieve_docs(user_message, top_k=5)
        context = "\n\n".join(docs)
        prompt = (
            "You are a helpful assistant. Use the following documents to answer the question.\n\n"
            f"Documents:\n{context}\n\nQuestion: {user_message}\nAnswer:"
        )

        resp = client.models.generate_content(model=MODEL, contents=prompt)
        text = getattr(resp, "text", None)
        return jsonify({"response": text})
    except Exception as e:
        traceback_str = traceback.format_exc()
        print("Error in /chat handler:\n", traceback_str)
        return jsonify({"error": str(e), "trace": traceback_str}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting app on 0.0.0.0:{port}, GEMINI_MODEL={MODEL}")
    app.run(host="0.0.0.0", port=port)
