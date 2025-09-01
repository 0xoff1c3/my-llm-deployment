# retriever.py
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load Excel workbook
excel_path = "IBP_APP_LOGS_TBL.xlsx"
sheets = pd.ExcelFile(excel_path).sheet_names

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Build document chunks
documents = []
for sheet in sheets:
    df = pd.read_excel(excel_path, sheet_name=sheet)
    keep_cols = [c for c in df.columns if df[c].notna().any()]
    if not keep_cols:
        continue
    for _, row in df[keep_cols].iterrows():
        vals = [str(v) for v in row.values if pd.notna(v)]
        if vals:
            chunk = " | ".join(vals)
            if len(chunk) > 1200:
                chunk = chunk[:1200]
            documents.append(chunk)

# Embeddings + FAISS index
doc_embeddings = embedder.encode(documents, convert_to_numpy=True, normalize_embeddings=True)
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(doc_embeddings)  # pyright: ignore [reportCallIssue]

def retrieve_documents(query: str, top_k: int = 10):
    query_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    sims, indices = index.search(query_vec, top_k) # pyright: ignore [reportCallIssue]
    return [documents[i] for i in indices[0]]
