from flask import Flask, request, jsonify
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# --- 1. Load embedding model and knowledge base ---
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example knowledge base
documents = [
    "Product X warranty lasts 2 years.",
    "Product Y has free shipping within Japan.",
    "Support is available 24/7 via email."
]

# Precompute embeddings for the knowledge base
doc_embeddings = model.encode(documents)

# --- 2. Gemini API setup ---
GEMINI_API_URL = "https://gemini.googleapis.com/v1/your-model-id:generateMessage"
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"

def generate_gemini_response(user_question, retrieved_docs):
    # Construct prompt with retrieved docs
    prompt = f"Use the following documents to answer the question.\n\nDocuments:\n{retrieved_docs}\n\nQuestion: {user_question}\nAnswer:"
    
    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_output_tokens": 300
    }
    
    response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()['output_text']
    else:
        return f"Error: {response.text}"

# --- 3. Route to handle chat ---
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    
    # --- Retrieve relevant document ---
    user_embedding = model.encode([user_input])
    similarities = cosine_similarity(user_embedding, doc_embeddings)
    top_idx = np.argmax(similarities)
    retrieved_doc = documents[top_idx]
    
    # --- Generate response ---
    answer = generate_gemini_response(user_input, retrieved_doc)
    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(debug=True)
