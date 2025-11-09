from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests
import os

# --- CONFIG ---
HF_TOKEN = ""
# Replace this with your endpoint URL from Hugging Face (not just the token)
HF_ENDPOINT = ""

# --- APP ---
app = FastAPI()

class Candidate(BaseModel):
    product: str
    text: str

class RerankRequest(BaseModel):
    query: str
    candidates: List[Candidate]

@app.post("/rerank")
def rerank(req: RerankRequest):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    # Prepare query and documents for Hugging Face API
    data = {
        "inputs": {
            "query": req.query,
            "documents": [c.text for c in req.candidates]
        }
    }

    response = requests.post(HF_ENDPOINT, headers=headers, json=data)

    if response.status_code != 200:
        return {"error": response.text}

    scores = response.json()  # The HF model returns a list of scores

    ranked = sorted(
        [
            {"id": c.product, "text": c.text, "score": s}
            for c, s in zip(req.candidates, scores)
        ],
        key=lambda x: x["score"],
        reverse=True,
    )

    return {"results": ranked}
