# Integration Guide - Fine-tuned Model with Java Backend

## Overview

This guide explains how to run the complete system with the fine-tuned sentence transformer integrated into the Java backend via Python FastAPI service.

## Architecture

```
User Query
    ↓
Java Spring Boot Backend (port 8080)
    ↓
DenseEmbeddingService → HTTP → Python FastAPI (port 8001) → Fine-tuned Model
SparseEmbeddingService → HTTP → Python FastAPI (port 8001) → N-gram Encoder
    ↓
PostgreSQL + pgvector (Hybrid Search)
    ↓
Top 10 Product Results
```

## Prerequisites

### Python Environment
```bash
pip install -r model_requirements.txt
# or with uv:
uv pip install -r model_requirements.txt
```

Required packages:
- `sentence-transformers==5.1.2`
- `fastapi`
- `uvicorn`
- `numpy`

### Java Environment
- JDK 17 or higher
- Maven (included via mvnw)

### Database
- PostgreSQL with pgvector extension

## Step-by-Step Startup

### 1. Start PostgreSQL Database

```bash
cd backend
docker-compose up -d
```

Verify database is running:
```bash
docker ps | grep postgres
```

### 2. Start Python Embedding Service

```bash
# From project root
uvicorn get_embeddings:app --port 8001
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8001
```

**Test the service:**
```bash
# Test dense embeddings (fine-tuned model)
curl -X POST "http://127.0.0.1:8001/dense-embed" \
  -H "Content-Type: application/json" \
  -d '{"query":"organic hearty soup"}'

# Test sparse embeddings (n-gram)
curl -X POST "http://127.0.0.1:8001/sparse-embed" \
  -H "Content-Type: application/json" \
  -d '{"query":"organic hearty soup"}'
```

### 3. Start Java Backend

```bash
cd backend
./mvnw spring-boot:run
```

You should see:
```
Started BackendApplication in X seconds
```

The backend will be available at `http://localhost:8080`

## API Endpoints

### Create Database

```bash
POST http://localhost:8080/create-db/{dbName}
Content-Type: application/json

[
  {
    "product_id": "123",
    "title": "Organic Soup",
    "description": "Delicious soup",
    ...
  }
]
```

### Query Products

```bash
POST http://localhost:8080/query
Content-Type: application/json

{
  "query": "organic hearty soup",
  "tableName": "products"
}
```

**Response:**
```json
["product_id_1", "product_id_2", ..., "product_id_10"]
```

## Testing the Integration

### Manual Test

1. **Verify Python service is using fine-tuned model:**
   ```bash
   curl -X POST "http://127.0.0.1:8001/dense-embed" \
     -H "Content-Type: application/json" \
     -d '{"query":"test"}' | python -c "import sys, json; data = json.load(sys.stdin); print(f'Embedding dimension: {len(data[\"dense_embedding\"])}')"
   ```
   Should output: `Embedding dimension: 384`

2. **Test Java backend calls Python service:**
   - Check Java logs when making a query
   - Should see: `Successfully got dense embedding of dimension: 384`
   - Should see: `Successfully got sparse embedding of dimension: 1000`

### Automated Test Script

```bash
python test_integration.py
```

This script will:
1. ✅ Verify Python service is running
2. ✅ Verify Java backend is running
3. ✅ Test embeddings are correct dimensions
4. ✅ Test end-to-end query flow

## Troubleshooting

### Python Service Not Starting

**Error:** `ModuleNotFoundError: No module named 'sentence_transformers'`

**Solution:**
```bash
pip install sentence-transformers fastapi uvicorn
```

### Java Can't Connect to Python Service

**Error:** `Failed to get dense embedding from Python service: Connection refused`

**Solution:**
- Make sure Python service is running on port 8001
- Check firewall settings
- Verify with: `curl http://localhost:8001/docs`

### Model Not Found

**Error:** `OSError: output/heb-semantic-search does not appear to be a valid path`

**Solution:**
- Ensure you've pulled the latest code with models
- Models should be in `output/heb-semantic-search/`
- Check with: `ls -la output/heb-semantic-search/model.safetensors`

### Database Connection Issues

**Error:** `Connection refused: localhost:5432`

**Solution:**
```bash
cd backend
docker-compose down
docker-compose up -d
```

## Performance Notes

### First Request Latency
- **First query:** ~2-3 seconds (model loading)
- **Subsequent queries:** ~100-200ms

### Model Loading
- Models are loaded once at Python service startup
- ~2-3 seconds to load fine-tuned model into memory
- Model stays in memory for fast inference

### Concurrent Requests
- Python service handles concurrent requests
- Consider using multiple workers for production:
  ```bash
  uvicorn get_embeddings:app --port 8001 --workers 4
  ```

## Development Workflow

### Making Changes to Embeddings

1. Modify `get_embeddings.py`
2. Restart Python service: `Ctrl+C` then `uvicorn get_embeddings:app --port 8001`
3. Java backend automatically picks up changes (no restart needed)

### Making Changes to Java Backend

1. Modify Java code
2. Restart with: `./mvnw spring-boot:run`
3. Python service keeps running (no restart needed)

## Production Considerations

### 1. **Use Environment Variables**

```python
# get_embeddings.py
import os
MODEL_PATH = os.getenv("MODEL_PATH", "output/heb-semantic-search")
```

```java
// DenseEmbeddingService.java
@Value("${python.service.url:http://localhost:8001}")
private String pythonServiceUrl;
```

### 2. **Add Health Checks**

```python
@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}
```

### 3. **Add Retry Logic**

Already included in Java services with fallback embeddings.

### 4. **Monitor Performance**

- Log embedding request times
- Monitor Python service memory usage
- Track Java-Python communication latency

## Model Information

### Dense Embeddings (Fine-tuned Model)
- **Model:** `output/heb-semantic-search`
- **Base:** `all-MiniLM-L6-v2`
- **Dimension:** 384
- **Training:** 3 epochs on query-product pairs
- **Performance:** Spearman correlation 0.74 (+189% vs baseline)

### Sparse Embeddings (N-gram)
- **Method:** Character n-grams (3-5)
- **Dimension:** 1000
- **Hashing:** MD5-based deterministic hashing
- **Normalization:** L2 normalized

## Additional Resources

- **Model Documentation:** `SENTENCE_TRANSFORMER_README.md`
- **Quick Start:** `QUICKSTART_MODEL.md`
- **Model Testing:** `python test_model_interface.py`
- **Accuracy Comparison:** `python compare_model_accuracy.py`

## Support

If you encounter issues:
1. Check logs from both Python and Java services
2. Verify all services are running: `ps aux | grep -E "uvicorn|java"`
3. Test each service independently before testing integration
4. Review this guide's troubleshooting section
