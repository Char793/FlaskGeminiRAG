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
context_columns = ["Restaurant name", "Categories", "Addresses", "Budget", "Transportation", "Operating Hours"]

df = pd.read_csv("knowledge_base.csv")
for _, row in df.iterrows():
    # Only join the relevant columns
    row_text = " | ".join([
        f"{col}: {row[col]}" for col in context_columns if pd.notna(row[col])
    ])
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

# --- NEW FUNCTION: Query Classifier ---
def classify_query(client, query):
    """
    Uses the Gemini API to classify the user's query as relevant or irrelevant.
    Returns: 'RELEVANT' or 'IRRELEVANT'
    """
    CLASSIFIER_MODEL = "gemini-2.5-flash" # Use a fast model for this simple task
    
    # Use a system instruction or a structured prompt to enforce a simple output
    classifier_prompt = (
        "Analyze the following user question. Determine if it is related to "
        "finding, recommending, or asking about a **restaurant, food, dining experience, or location** "
        "that would typically be answered using a restaurant database. "
        "Ignore simple greetings or non-sequitur questions (e.g., weather, history, etc.).\n\n"
        f"Question: \"{query}\"\n\n"
        "**INSTRUCTION: Respond with one word only: RELEVANT or IRRELEVANT.**"
    )
    
    # We use generate_content for a quick classification
    resp = client.models.generate_content(
        model=CLASSIFIER_MODEL,
        contents=classifier_prompt
    )
    
    # Clean and return the classification result
    return resp.text.strip().upper()

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

        # 1. CLASSIFY THE QUERY
        relevance_status = classify_query(client, user_message)
        
        # 2. DECIDE ACTION BASED ON CLASSIFICATION
        if relevance_status == "IRRELEVANT":
            # For irrelevant queries (including simple greetings)
            # Send a general conversational prompt without RAG context
            simple_prompt = (
                "You are a helpful assistant. The user's question is not related to restaurants. "
                "Kindly inform the user that you are specialized in **restaurant recommendations** "
                "and ask what kind of food or dining experience they are looking for."
            )
            resp = client.models.generate_content(model=MODEL, contents=simple_prompt)
            text = getattr(resp, "text", None)
            
            # Add logging for debugging
            print(f"Query: '{user_message}' | Status: IRRELEVANT | Response: {text[:50]}...")
            
            return jsonify({"response": text})
        
        # 3. STANDARD RAG PROCESS FOR RELEVANT QUERIES
        elif relevance_status == "RELEVANT":
            # Perform RAG (Retrieval-Augmented Generation)
            docs = retrieve_docs(user_message, top_k=5)
            context = "\n\n".join(docs)
            
            # Use the improved, strictly formatted prompt from the previous step
            # --- IMPROVED PROMPT ---
            rag_prompt = (
                "You are a helpful assistant specializing in restaurant data. "
                "Use ONLY the following documents to answer the question.\n\n"
                f"Documents:\n{context}\n\n"
                "**INSTRUCTIONS:** "
                "1. Answer the user's question with a brief introduction followed by a list of suggestions. "
                "2. DO NOT use any markdown characters like '*', '**', or '#' in your final response. Use plain text formatting only. "
                "3. Separate items in your list with a simple newline or a dash (-) and a space.\n\n"
                f"Question: {user_message}\n\n"
                "Answer:"
            )

            resp = client.models.generate_content(model=MODEL, contents=rag_prompt)
            text = getattr(resp, "text", None)
            
            # Add logging for debugging
            print(f"Query: '{user_message}' | Status: RELEVANT | Context Used | Response: {text[:50]}...")
            
            return jsonify({"response": text})
            
        else:
            # Fallback for unexpected classification response
            return jsonify({"error": "could not classify query relevance"}), 500

    except Exception as e:
        traceback_str = traceback.format_exc()
        print("Error in /chat handler:\n", traceback_str)
        return jsonify({"error": str(e), "trace": traceback_str}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting app on 0.0.0.0:{port}, GEMINI_MODEL={MODEL}")
    app.run(host="0.0.0.0", port=port)
