import csv

def load_documents(filepath):
    documents = []
    if filepath.endswith(".csv"):
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("content", "").strip():
                    documents.append(row["content"].strip())
    elif filepath.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            documents = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError("Unsupported file type: must be .csv or .txt")

    print(f"✅ Loaded {len(documents)} documents from {filepath}")
    return documents
