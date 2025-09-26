import os
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai.types import HttpOptions

from retrieval import retrieve_docs, preload_embeddings
from storage_logger import log_chat
from knowledge_base import load_documents

# --- Flask app ---
app = Flask(__name__)
CORS(app, origins="*")

# --- Env vars ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set.")

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# --- Gemini Client ---
client = genai.Client(api_key=GEMINI_API_KEY, http_options=HttpOptions(api_version="v1"))

# --- Knowledge base ---
documents = load_documents("knowledge_base.csv")

# --- Start background preload (embeddings) ---
preload_embeddings(documents)

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
        text = getattr(resp, "text", None) or resp.candidates[0].content.parts[0].text
        return jsonify({"ok": True, "model": MODEL, "response": text})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "no message provided"}), 400

        docs = retrieve_docs(user_message, documents, top_k=5)
        context = "\n\n".join(docs)

        prompt = (
            "You are a helpful assistant. Use the following documents to answer the question.\n\n"
            f"Documents:\n{context}\n\nQuestion: {user_message}\nAnswer:"
        )

        resp = client.models.generate_content(model=MODEL, contents=prompt)
        text = getattr(resp, "text", None) or resp.candidates[0].content.parts[0].text

        log_chat(user_message, text)
        return jsonify({"response": text})
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
