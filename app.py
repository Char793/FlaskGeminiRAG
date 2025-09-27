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

# --- Load precomputed embeddings ---
EMBED_FILE = "knowledge_base_embedded.npy"
if not os.path.exists(EMBED_FILE):
    raise RuntimeError(f"{EMBED_FILE} not found. You must precompute embeddings locally first.")

doc_embeddings = np.load(EMBED_FILE)
print(f"✅ Loaded embeddings from {EMBED_FILE}")

# --- Retrieval ---
def retrieve_docs(query, top_k=5):
    """Retrieve top_k most similar documents using precomputed embeddings."""
    # Embed query on-the-fly
    resp = client.models.embed_content(
        model="text-embedding-004",
        contents=[query]
    )
    q_emb = np.array(resp.embeddings[0].values)
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

