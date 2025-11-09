from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import CrossEncoder

app = FastAPI()
model = CrossEncoder("BAAI/bge-reranker-base")

class Candidate(BaseModel):
    product: str
    text: str

class RerankRequest(BaseModel):
    query: str
    candidates: List[Candidate]

@app.post("/rerank")
def rerank(req: RerankRequest):
    # Prepare (query, text) pairs for CrossEncoder
    pairs = [(req.query, candidate.text) for candidate in req.candidates]

    # Model returns relevance scores (higher is better)
    scores = model.predict(pairs).tolist()

    # Fixed: use candidate.product instead of Candidate.product_id
    ranked = sorted(
        [{"id": candidate.product, "text": candidate.text, "score": score}
         for candidate, score in zip(req.candidates, scores)],
        key=lambda x: x["score"],
        reverse=True
    )

    return {"results": ranked}

'''
Test request:
curl -X POST "http://127.0.0.1:8002/rerank" \
-H "Content-Type: application/json" \
-d '{
  "query": "organic soup",
  "candidates": [
    {
      "product": "10045036",
      "text": "H-E-B Fish Market Party Tray - Seasoned Shrimp Cocktail"
    },
    {
      "product": "10048008", 
      "text": "Organic Vegetable Soup - Rich and Hearty"
    }
  ]
}'
'''