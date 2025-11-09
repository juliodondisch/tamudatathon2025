# End-to-End Accuracy Testing Guide

## Overview

This guide shows how to test the accuracy of baseline vs fine-tuned models using the complete system pipeline (Java backend → Python API → Vector Database).

## What Gets Tested

The test script (`test_e2e_accuracy.py`) measures:
- **Spearman Correlation**: How well rankings match true relevance
- **Pearson Correlation**: Linear relationship between predicted and true relevance
- **NDCG@10**: Normalized Discounted Cumulative Gain (search quality metric)

These metrics are calculated on real queries through the full system, not just isolated model testing.

## Prerequisites

1. **Test Data**: `data/` directory with:
   - `products.json`
   - `queries_synth_train.json`
   - `labels_synth_train.json`

2. **Models**: Both models in `output/`:
   - `baseline-model/`
   - `heb-semantic-search/`

3. **Services**:
   - PostgreSQL (docker-compose)
   - You'll start/stop Python and Java services during test

## Quick Start

### Step 1: Ensure Database is Running

```bash
cd backend
docker-compose up -d
```

### Step 2: Run Test Script

```bash
python test_e2e_accuracy.py
```

The script will guide you through:
1. Testing with baseline model
2. Testing with fine-tuned model
3. Comparing results

## Manual Testing Process

### Test Baseline Model

**Terminal 1: Python Service (Baseline)**
```bash
MODEL_PATH=baseline uvicorn get_embeddings:app --port 8001
```

**Terminal 2: Java Backend**
```bash
cd backend
./mvnw spring-boot:run
```

**Terminal 3: Run Test**
```bash
python test_e2e_accuracy.py
# Follow prompts for baseline test
```

**Stop Services:**
- Terminal 1: `Ctrl+C` (stop Python)
- Terminal 2: `Ctrl+C` (stop Java)

### Test Fine-tuned Model

**Terminal 1: Python Service (Fine-tuned)**
```bash
# Default is fine-tuned, or explicitly:
MODEL_PATH=finetuned uvicorn get_embeddings:app --port 8001
```

**Terminal 2: Java Backend**
```bash
cd backend
./mvnw spring-boot:run
```

**Terminal 3: Continue Test**
```bash
# Script will continue from baseline test
# Follow prompts for fine-tuned test
```

## Expected Results

Based on isolated model testing, you should see:

### Baseline Model
- **Spearman**: ~0.25-0.30
- **Pearson**: ~0.25-0.30
- **NDCG@10**: ~0.40-0.50

### Fine-tuned Model
- **Spearman**: ~0.70-0.75 (+189% improvement)
- **Pearson**: ~0.75-0.80
- **NDCG@10**: ~0.65-0.75

### Comparison Output

```
🎯 BASELINE vs FINE-TUNED COMPARISON
================================================================================

Metric                    Baseline     Fine-tuned   Improvement
----------------------------------------------------------------------
Spearman Correlation      0.2572       0.7429       +0.4857 (+188.8%)
Pearson Correlation       0.2597       0.7620       +0.5023 (+193.5%)
NDCG@10                   0.4523       0.7156       +0.2633 (+58.2%)

================================================================================
📝 ASSESSMENT:
================================================================================
🌟 EXCELLENT! Fine-tuning significantly improves search quality!
```

## Model Switching

The Python service (`get_embeddings.py`) now supports model switching via environment variable:

### Fine-tuned (Default)
```bash
uvicorn get_embeddings:app --port 8001
# or explicitly
MODEL_PATH=finetuned uvicorn get_embeddings:app --port 8001
```

### Baseline
```bash
MODEL_PATH=baseline uvicorn get_embeddings:app --port 8001
```

### Custom Path
```bash
MODEL_PATH=/path/to/model uvicorn get_embeddings:app --port 8001
```

### Check Current Model
```bash
curl http://localhost:8001/model-info
```

Response:
```json
{
  "model_name": "Fine-tuned Model",
  "model_path": "output/heb-semantic-search",
  "embedding_dimension": 384,
  "is_finetuned": true
}
```

## Troubleshooting

### Database Creation Fails

**Error**: `Database creation failed: 400`

**Solution:**
- Check PostgreSQL is running: `docker ps | grep postgres`
- Check Java logs for errors
- Verify Python service is responding: `curl http://localhost:8001/model-info`

### Queries Return Empty Results

**Error**: All queries return `[]`

**Solution:**
- Database might not be fully indexed yet - wait 10-30 seconds
- Check Java logs for errors
- Verify table was created: check PostgreSQL directly

### Python Service Not Switching Models

**Error**: Both tests use same model

**Solution:**
- Make sure to stop Python service between tests (`Ctrl+C`)
- Verify model switch with: `curl http://localhost:8001/model-info`
- Check terminal shows correct model loading message

### Test Takes Too Long

The script tests 50 queries by default. To speed up:

**Edit `test_e2e_accuracy.py`:**
```python
# Line ~200
num_queries_to_test = min(20, len(queries_to_test))  # Test only 20
```

**Or use fewer products:**
```python
# Line ~330
if not create_database(products[:50], table_name):  # Use 50 instead of 100
```

## Understanding the Metrics

### Spearman Correlation
- Measures rank correlation
- Range: -1 to 1 (higher is better)
- **0.74** means predictions strongly correlate with true relevance

### Pearson Correlation
- Measures linear relationship
- Range: -1 to 1 (higher is better)
- Similar to Spearman but more sensitive to outliers

### NDCG@10 (Normalized Discounted Cumulative Gain)
- Measures search result quality
- Range: 0 to 1 (higher is better)
- Rewards relevant items appearing earlier in results
- **0.70+** is considered very good for search systems

### Average Rank by Relevance
Shows where products of each relevance level appear in results:
- **Relevance 3** (highly relevant): Should have low rank (1-3)
- **Relevance 0** (not relevant): Should have high rank (8-10)

## Performance Notes

### Test Duration
- **Baseline test**: ~3-5 minutes (50 queries)
- **Fine-tuned test**: ~3-5 minutes (50 queries)
- **Total**: ~10 minutes

### Database Size
- Script uses 100 products for faster testing
- Full database (5000+ products) will take longer but give more accurate results

### Query Processing
- Each query takes ~100-200ms through full pipeline
- Includes: Java → Python → Model → Vector DB → Results

## Next Steps

After confirming improvement:
1. Update production configuration to use fine-tuned model
2. Monitor real user queries for continued performance
3. Consider retraining with more data if available
4. Run full test with all products for production validation

## Files

- **`test_e2e_accuracy.py`**: Main test script
- **`get_embeddings.py`**: Updated with model switching
- **`INTEGRATION_GUIDE.md`**: Full integration documentation
- **`test_integration.py`**: Basic connectivity test
