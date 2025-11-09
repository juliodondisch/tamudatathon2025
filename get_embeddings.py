"""
Get vector embeddings - minimal output version.

Usage:
    python get_embeddings_simple.py "hearty organic soups"
    python get_embeddings_simple.py "soup" --model output/heb-semantic-search
"""
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
import hashlib
from model_interface_v2 import GrocerySearchModel

app = FastAPI()

# load model once at startup
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME) # Should be GroceryCheckout

class EncodingRequest(BaseModel):
    query: str

@app.post("/dense-embed")
def denseEncode(req: EncodingRequest):
    emb = model.encode(req.query) # should be encode_query
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