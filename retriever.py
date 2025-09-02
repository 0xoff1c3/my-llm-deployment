import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging
import os
from hdbcli import dbapi
from dotenv import load_dotenv
from difflib import SequenceMatcher

# Setup logging
logging.basicConfig(level=logging.INFO, filename='retriever.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
HANA_HOST = "73a8a9e6-3c93-49a4-8dd3-c5dfb09ea621.hana.trial-us10.hanacloud.ondemand.com"
HANA_PORT = 443
HANA_USER = "DBADMIN"         # or your custom DB user
HANA_PASSWORD = "Hana@123" 
HANA_SCHEMA = ""

# Connect to SAP HANA
def connect_db():
    try:
        conn = dbapi.connect(
            address=HANA_HOST,
            port=HANA_PORT,
            user=HANA_USER,
            password=HANA_PASSWORD
        )
        logging.info("Connected to SAP HANA database")
        return conn
    except Exception as e:
        logging.error(f"Failed to connect to SAP HANA: {repr(e)}")
        raise

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Cache file
index_path = "faiss_index.bin"

# Build document chunks from SAP HANA table
documents = []
chunk_count = 0
conn = connect_db()
cursor = conn.cursor()
table_name = f'"{HANA_SCHEMA}"."IBP_APP_LOGS_TBL"' if HANA_SCHEMA else "IBP_APP_LOGS_TBL"
try:
    cursor.execute(f"SELECT PLANNING_AREA, CUSTOMER_ID, PRODUCT_ID, FORECAST_MODEL, STEP, SEVERITY, MESSAGE, DATE_TIME FROM {table_name}")
    rows = cursor.fetchall()
    if not rows:
        logging.warning("No data in table; using empty documents")
    cols = ['PLANNING_AREA', 'CUSTOMER_ID', 'PRODUCT_ID', 'FORECAST_MODEL', 'STEP', 'SEVERITY', 'MESSAGE', 'DATE_TIME']
    df = pd.DataFrame(rows, columns=cols)  # pyright: ignore [reportArgumentType]
    keep_cols = [c for c in df.columns if df[c].notna().any()] # pyright: ignore [reportGeneralTypeIssues]
    if not keep_cols:
        logging.warning("No non-NULL columns found in table")
    else:
        for _, row in df[keep_cols].iterrows():
            vals = [str(v) for v in row.values if pd.notna(v)]
            if vals:
                full_text = " | ".join(vals)
                if len(full_text) > 600:
                    chunks = []
                    for i in range(0, len(full_text), 400):
                        chunk = full_text[i:i+600]
                        if chunk:
                            chunks.append(chunk)
                    documents.extend(chunks)
                    chunk_count += len(chunks)
                else:
                    documents.append(full_text)
                    chunk_count += 1
    logging.info(f"Processed table '{table_name}' with {chunk_count} chunks")
except Exception as e:
    logging.error(f"Error querying table: {repr(e)}")
    raise
finally:
    cursor.close()
    conn.close()

logging.info(f"Total documents/chunks: {len(documents)}")

# Load or create FAISS index
if os.path.exists(index_path):
    index = faiss.read_index(index_path)
    logging.info("Loaded cached FAISS index")
else:
    if not documents:
        raise ValueError("No documents to embed; load data into table first")
    doc_embeddings = embedder.encode(documents, convert_to_numpy=True, normalize_embeddings=True, batch_size=1)
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(doc_embeddings) # pyright: ignore [reportCallIssue]

    faiss.write_index(index, index_path)
    logging.info("Created and cached FAISS index")

def retrieve_documents(query: str, sim_threshold: float = 0.0,
                       from_date: str = "", to_date: str = ""):
    conn = connect_db()
    cursor = conn.cursor()
    table_name = f'"{HANA_SCHEMA}"."IBP_APP_LOGS_TBL"' if HANA_SCHEMA else "IBP_APP_LOGS_TBL"

    # Build WHERE clause dynamically
    where_clauses = []
    if from_date.strip():
        where_clauses.append(f"DATE_TIME >= '{from_date} 00:00:00'")
    if to_date.strip():
        where_clauses.append(f"DATE_TIME <= '{to_date} 23:59:59'")
    where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    query_sql = f"""
        SELECT PLANNING_AREA, CUSTOMER_ID, PRODUCT_ID, FORECAST_MODEL,
               STEP, SEVERITY, MESSAGE, DATE_TIME
        FROM {table_name}
        {where_sql}
    """

    try:
        cursor.execute(query_sql)
        rows = cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

    if not rows:
        logging.warning("No documents found for given filters")
        return []

    # Build document list
    docs = []
    cols = ['PLANNING_AREA','CUSTOMER_ID','PRODUCT_ID','FORECAST_MODEL',
            'STEP','SEVERITY','MESSAGE','DATE_TIME']
    df = pd.DataFrame(rows, columns=cols) #type: ignore 
    for _, row in df.iterrows():
        vals = [str(v) for v in row.values if pd.notna(v)]
        if vals:
            docs.append(" | ".join(vals))

    # Encode query and docs
    query_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    doc_embeddings = embedder.encode(docs, convert_to_numpy=True, normalize_embeddings=True)

    dimension = doc_embeddings.shape[1]
    temp_index = faiss.IndexFlatIP(dimension)
    temp_index.add(doc_embeddings) #pyright: ignore [reportCallIssue]

    # Retrieve all docs above similarity threshold
    sims, indices = temp_index.search(query_vec, len(docs))  #type: ignore
    candidate_docs = [(docs[i], sims[0][j]) for j, i in enumerate(indices[0]) if sims[0][j] >= sim_threshold]

    # Sort by similarity descending
    candidate_docs.sort(key=lambda x: x[1], reverse=True)

    # Return all results; limit only when displaying
    return [doc for doc, _ in candidate_docs]
