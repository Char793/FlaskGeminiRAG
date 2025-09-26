import os
from flask import Flask, request, jsonify
from google import genai   # pip package google-genai
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
CORS(app)  # allow cross-origin requests from GitHub Pages (restrict in prod)

# read secret from env
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.3B")  # change as needed

# setup genai client (it reads GEMINI_API_KEY from env automatically)
client = genai.Client(api_key=GEMINI_API_KEY)

# simple static KB (replace with vector DB in prod)
documents = ["Product X warranty lasts 2 years.", ...]
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embed_model.encode(documents)

def retrieve_docs(query, top_k=1):
    q_emb = embed_model.encode([query])
    sims = cosine_similarity(q_emb, doc_embeddings)[0]
    idx = sims.argsort()[::-1][:top_k]
    return [documents[i] for i in idx]

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    docs = retrieve_docs(user_message, top_k=3)
    prompt = "Use these documents to answer:\n\n" + "\n\n".join(docs) + f"\n\nQuestion: {user_message}"
    resp = client.responses.generate(
        model=MODEL,
        input=prompt,
        temperature=0.2,
        max_output_tokens=400
    )
    text = resp.output[0].content[0].text if resp.output else "No answer"
    return jsonify({"response": text})
