# Quick Start - Sentence Transformer Model

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r model_requirements.txt
# or with uv:
uv pip install -r model_requirements.txt
```

### 2. Use the Model (Simple)

```python
from model_interface_v2 import GrocerySearchModel

# Load fine-tuned model
model = GrocerySearchModel()

# Generate query embedding
query_embedding = model.encode_query("organic soup")

# Generate product embeddings
products = [
    {"title": "Organic Lentil Soup", "description": "...", "brand": "Amy's"},
    {"title": "Chicken Soup", "description": "...", "brand": "Campbell's"}
]
product_embeddings = model.encode_products(products)

# Shape: query_embedding is (384,), product_embeddings is (2, 384)
```

### 3. Test the Model

```bash
# Test fine-tuned model
python test_model_interface.py

# Test baseline model
python test_model_interface.py --model baseline

# Test specific model path
python test_model_interface.py --model output/heb-semantic-search
```

### 4. Compare Model Performance

```bash
# Compare baseline vs fine-tuned accuracy
python compare_model_accuracy.py
```

**Results:** Fine-tuned model achieves **+189% improvement** in Spearman correlation!

### 5. Download Baseline Model (if needed)

```bash
python download_baseline_model.py
```

## 📁 New Files Added

### Core Files
- **`model_interface_v2.py`** - Main API to use the models
- **`model_requirements.txt`** - Python dependencies
- **`output/heb-semantic-search/`** - 🎯 Fine-tuned model (88MB)
- **`output/baseline-model/`** - Baseline model for comparison (88MB)

### Testing & Utilities
- **`test_model_interface.py`** - Test suite with model switching
- **`compare_model_accuracy.py`** - Compare baseline vs fine-tuned
- **`search_demo.py`** - Product search demonstration
- **`download_baseline_model.py`** - Download baseline model
- **`generate_embeddings.py`** - CLI embedding generator

### Training (Optional)
- **`train_sentence_transformer.py`** - Training script
- **`evaluate_sentence_transformer.py`** - Evaluation script

### Documentation
- **`SENTENCE_TRANSFORMER_README.md`** - Full documentation
- **`QUICKSTART_MODEL.md`** - This file

## 🎯 For Vector Database Integration

The cleanest way to integrate:

```python
from model_interface_v2 import GrocerySearchModel

# Initialize once at startup
model = GrocerySearchModel()

# Encode user queries at search time
query_embedding = model.encode_query("organic soup")
# Returns: numpy array (384,)

# Encode products in batch (do once or when products update)
product_embeddings = model.encode_products(products_list)
# Returns: numpy array (num_products, 384)

# Use cosine similarity for search
from sentence_transformers import util
similarities = util.cos_sim(query_embedding, product_embeddings)
```

## 📊 Model Performance

| Metric | Baseline | Fine-tuned | Improvement |
|--------|----------|------------|-------------|
| Spearman | 0.26 | **0.74** | +189% |
| Pearson | 0.26 | **0.76** | +193% |

The fine-tuned model correctly:
- Gives low scores (0.16) to irrelevant products
- Gives high scores (0.69) to highly relevant products

## 🔧 Model Details

- **Base Model**: `all-MiniLM-L6-v2`
- **Embedding Dimension**: 384
- **Model Size**: 88MB
- **Training**: 3 epochs on query-product pairs
- **Similarity Metric**: Cosine similarity

## 💡 Example Usage

```python
from model_interface_v2 import GrocerySearchModel

# Load model
model = GrocerySearchModel()

# Simple query
embedding = model.encode_query("hearty soup")
print(f"Embedding shape: {embedding.shape}")  # (384,)

# Batch queries
embeddings = model.encode_query(["soup", "pasta", "bread"])
print(f"Batch shape: {embeddings.shape}")  # (3, 384)

# Save embeddings
model.save_embeddings(embeddings, "embeddings.npy")

# Load embeddings
loaded = model.load_embeddings("embeddings.npy")
```

## 📚 More Info

See `SENTENCE_TRANSFORMER_README.md` for complete documentation.

## ⚠️ Note

These files are NEW and won't conflict with existing repository files. All existing files remain untouched.
