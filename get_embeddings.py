"""
Get vector embeddings - minimal output version.

Usage:
    python get_embeddings_simple.py "hearty organic soups"
    python get_embeddings_simple.py "soup" --model output/heb-semantic-search
"""
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import hashlib
from model_interface_v2 import GrocerySearchModel
from PIL import Image
import requests
from io import BytesIO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("image_embeddings")

app = FastAPI()

# load model once at startup
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

IMAGE_MODEL_NAME = "clip-ViT-B-32"
image_model = SentenceTransformer(IMAGE_MODEL_NAME)

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

@app.post("/image-embed")
def imageEncode(req: EncodingRequest):
    try:
        # Fetch image data
        val = req.query.strip()
        logger.info(f"Received input: {val}")

        if(len(val) < 15 and val.isdigit()):
            image_url = "https://images.heb.com/is/image/HEBGrocery/0" + val
            logger.info(f"image url: {image_url}")
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            img = val

    except Exception as e:
        logger.info(f"Error: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to retrieve or decode image: {e}")

    logger.info(f"img value: {img}")
    # Compute embedding
    emb = image_model.encode(img)
    logger.info(f"ebmedding: {emb}")

    return {"image_embedding": emb.tolist()}

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
