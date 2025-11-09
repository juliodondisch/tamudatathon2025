"""
Get vector embeddings - minimal output version.

Usage:
    # Fine-tuned model (default)
    uvicorn get_embeddings:app --port 8001

    # Baseline model
    MODEL_PATH=baseline uvicorn get_embeddings:app --port 8001

    # Or custom path
    MODEL_PATH=output/baseline-model uvicorn get_embeddings:app --port 8001
"""
import argparse
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
import hashlib

app = FastAPI()

# Load model once at startup
# Can be controlled via MODEL_PATH environment variable
# Options: "finetuned" (default), "baseline", or a custom path
model_env = os.getenv("MODEL_PATH", "finetuned")

if model_env == "finetuned":
    MODEL_PATH = "output/heb-semantic-search"
    MODEL_NAME = "Fine-tuned Model"
elif model_env == "baseline":
    MODEL_PATH = "output/baseline-model"
    MODEL_NAME = "Baseline Model"
else:
    MODEL_PATH = model_env
    MODEL_NAME = f"Custom Model ({model_env})"

print(f"Loading model: {MODEL_NAME}")
print(f"Path: {MODEL_PATH}")
model = SentenceTransformer(MODEL_PATH)
print(f"✅ Model loaded successfully (dimension: {model.get_sentence_embedding_dimension()})")

class EncodingRequest(BaseModel):
    query: str

@app.post("/dense-embed")
def denseEncode(req: EncodingRequest):
    emb = model.encode(req.query)
    return {"dense_embedding": emb.tolist()}

@app.post("/sparse-embed")
def sparseEncode(req: EncodingRequest):
    # simple deterministic n-gram hashing encoder (char n-grams)
    size = 1000
    vec = np.zeros(size, dtype=float)
    text = req.query.lower()

    # character n-grams (3..5)
    min_n = 3
    max_n = 5
    for n in range(min_n, max_n + 1):
        if len(text) < n:
            continue
        for i in range(len(text) - n + 1):
            ngram = text[i:i + n]
            idx = int(hashlib.md5(ngram.encode()).hexdigest(), 16) % size
            vec[idx] += 1.0

    # optional normalization (L2)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = (vec / norm).astype(float)

    return {"sparse_embedding": vec.tolist()}

@app.get("/model-info")
def modelInfo():
    """Get information about the currently loaded model."""
    return {
        "model_name": MODEL_NAME,
        "model_path": MODEL_PATH,
        "embedding_dimension": model.get_sentence_embedding_dimension(),
        "is_finetuned": model_env == "finetuned"
    }

'''
Test Requests:
# Dense:
curl -s -X POST "http://127.0.0.1:8001/dense-embed" \
  -H "Content-Type: application/json" \
  -d '{"query":"hearty organic soups"}' | jq .

# Sparse:
curl -s -X POST "http://127.0.0.1:8001/sparse-embed" \
  -H "Content-Type: application/json" \
  -d '{"query":"hearty organic soups"}' | jq .
'''