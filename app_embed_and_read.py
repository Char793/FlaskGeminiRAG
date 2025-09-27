import os
import traceback
import numpy as np
import pandas as pd
import uuid  # セッションID生成用
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai.types import HttpOptions, GenerateContentConfig 

# --- Flask App ---
app = Flask(__name__)
CORS(app, origins="*")

# --- Env Vars ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set.")

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# --- Gemini Client ---
client = genai.Client(api_key=GEMINI_API_KEY, http_options=HttpOptions(api_version="v1"))

# --- 💡 ステートフル対応: インメモリセッションストア ---
# 実際のアプリケーションでは、Redisやデータベースを使用してください
chat_sessions = {}

# --- Load Knowledge Base (CSV) ---
documents = []
# 💡 修正点: 'Addresses'をRAGコンテキストに必ず含める
context_columns = ["Restaurant name", "Categories", "Addresses", "Transportation", "Budget"]

df = pd.read_csv("knowledge_base.csv")
for _, row in df.iterrows():
    # 選択した列のみを結合
    row_text = " | ".join([
        f"{col}: {row[col]}" for col in context_columns if col in df.columns and pd.notna(row[col])
    ])
    documents.append(row_text)

print(f"✅ Loaded {len(documents)} documents from knowledge_base.csv")

# --- Embeddings (変更なし) ---
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

# --- Retrieval (変更なし) ---
def retrieve_docs(query, top_k=5):
    global doc_embeddings
    if doc_embeddings is None:
        load_or_create_embeddings()

    # Embed user query
    q_emb = embed_texts([query])[0]
    sims = np.dot(doc_embeddings, q_emb) / (np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-8)
    idx = sims.argsort()[::-1][:top_k]
    return [documents[i] for i in idx]

# --- Query Classifier (FINAL FIX: REMOVE system_instruction) ---
def classify_query(client, query):
    """
    Uses the Gemini API to classify the user's query as relevant or irrelevant.
    
    Returns: 'RELEVANT' or 'IRRELEVANT'
    """
    CLASSIFIER_MODEL = "gemini-2.5-flash"
    
    # 💡 FIX: Combine the instruction and query into the contents list
    classification_prompt = (
        "Analyze the following user question. Determine if it is related to "
        "finding, recommending, or asking about a **restaurant, food, dining experience, or location**. "
        f"The user's question is: \"{query}\".\n\n"
        "**INSTRUCTION: Respond with one word only: RELEVANT or IRRELEVANT.**"
    )

    # Note: No 'config' object is passed, as it caused the error
    resp = client.models.generate_content(
        model=CLASSIFIER_MODEL,
        contents=classification_prompt
    )
    
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
        session_id = data.get("session_id")

        if not user_message:
            return jsonify({"error": "no message provided"}), 400

        # 1. セッションの取得または作成
        if session_id and session_id in chat_sessions:
            chat = chat_sessions[session_id]
        else:
            session_id = str(uuid.uuid4())

            # 💡 FIX: Create chat WITHOUT the 'config' object
            chat = client.chats.create(model=MODEL)

            # 💡 FIX: Send the system instruction as the very first message 
            # This sets the persona without using the unsupported 'system_instruction' field.
            initial_instruction = (
                "You are a helpful assistant specializing in restaurant data. "
                "Maintain conversation flow and remember previously suggested restaurants when asked for details like 'where are they?'."
            )
            chat.send_message(initial_instruction)

            chat_sessions[session_id] = chat
            print(f"🆕 New session created: {session_id}")
            

        # 2. クエリの分類
        relevance_status = classify_query(client, user_message)
        response_text = ""
        
        if relevance_status == "IRRELEVANT":
            simple_prompt = "The user's question is not related to restaurants. Kindly inform the user that you are specialized in **restaurant recommendations** and ask what kind of food or dining experience they are looking for."
            resp = client.models.generate_content(model=MODEL, contents=simple_prompt)
            response_text = getattr(resp, "text", None)
            
        elif relevance_status == "RELEVANT":
            # 3. 関連クエリのRAGプロセス
            docs = retrieve_docs(user_message, top_k=5)
            context = "\n\n".join(docs)
            
            rag_prompt_for_chat = (
                f"Use ONLY the following documents to answer the question, but remember the conversation history. The documents are:\n{context}\n\n"
                "**INSTRUCTIONS:** "
                "1. If the question asks for locations (e.g., 'where are they?'), use the conversation history to identify the restaurant names and provide the **full Address** (from the Documents).\n"
                "2. For all suggestions, you **MUST** separate each suggestion onto a new line and use a **numbered list format starting with (1), (2), (3), etc.**\n"
                "3. DO NOT use any markdown characters like '*', '**', or '#' in your final response. Use plain text formatting only.\n"
                f"Question: {user_message}\n\n"
                "Answer:"
            )

            # Use chat.send_message, which preserves history
            resp = chat.send_message(rag_prompt_for_chat)
            response_text = getattr(resp, "text", None)
            
        else:
             return jsonify({"error": "could not classify query relevance", "session_id": session_id}), 500

        return jsonify({"response": response_text, "session_id": session_id})

    except Exception as e:
        traceback_str = traceback.format_exc()
        print("Error in /chat handler:\n", traceback_str)
        return jsonify({"error": str(e), "trace": traceback_str, "session_id": session_id}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting app on 0.0.0.0:{port}, GEMINI_MODEL={MODEL}")
    app.run(host="0.0.0.0", port=port)
