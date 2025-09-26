import os
import datetime
from google.cloud import storage

LOG_BUCKET = os.environ.get("LOG_BUCKET", "my-chat-logs")
storage_client = storage.Client()

def log_chat(user_message, response):
    ts = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
    filename = f"logs/chat-{ts}.txt"

    bucket = storage_client.bucket(LOG_BUCKET)
    blob = bucket.blob(filename)

    log_text = f"Q: {user_message}\nA: {response}\n"
    blob.upload_from_string(log_text, content_type="text/plain")

    print(f"✅ Logged chat to {filename}")
