# Grocery Search Sentence Transformer

Fine-tuned sentence transformer model for semantic product search in a grocery store application.

## Model Overview

- **Base Model**: `all-MiniLM-L6-v2`
- **Fine-tuned For**: Query-to-product semantic matching
- **Embedding Dimension**: 384
- **Training Data**: Synthetic query-product pairs with relevance scores (0-3)
- **Model Location**: `output/heb-semantic-search/`

## Quick Start

### Installation

```bash
# Using uv
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

### Using the Model (Vector DB Team)

The cleanest way to use the model is through `model_interface.py`:

```python
from model_interface import GrocerySearchModel

# Initialize model
model = GrocerySearchModel()

# Generate query embeddings
query_embedding = model.encode_query("organic soup")
# Output shape: (384,)

# Generate product embeddings (batch processing)
products = [
    {"title": "Organic Lentil Soup", "description": "...", "brand": "Amy's", ...},
    {"title": "Chicken Noodle Soup", "description": "...", "brand": "Campbell's", ...}
]
product_embeddings = model.encode_products(products)
# Output shape: (num_products, 384)

# Calculate similarities (cosine similarity)
from sentence_transformers import util
similarities = util.cos_sim(query_embedding, product_embeddings)
```

### Key API Methods

- `encode_query(query)` - Encode search queries
- `encode_products(products)` - Encode product dictionaries (handles formatting)
- `encode_text(texts)` - Encode arbitrary text
- `get_embedding_dimension()` - Returns 384
- `save_embeddings()` / `load_embeddings()` - Persistence utilities

## Project Structure

```
.
├── model_interface.py          # 🎯 Main API for vector DB integration
├── output/heb-semantic-search/ # 🎯 Trained model files
├── train_model.py              # Training script (if retraining needed)
├── search.py                   # Demo: search products using the model
├── evaluate_model.py           # Compare baseline vs fine-tuned performance
├── generate_embeddings.py      # Utility: generate embeddings from CLI
├── data/                       # Training data
│   ├── products.json
│   ├── queries_synth_train.json
│   └── labels_synth_train.json
└── requirements.txt            # Dependencies
```

## Vector Database Integration

### Recommended Workflow

1. **Load the model once at startup**:
   ```python
   from model_interface import GrocerySearchModel
   model = GrocerySearchModel()
   ```

2. **Precompute product embeddings** (do this once or when products update):
   ```python
   # Load your products
   products = load_products()  # Your function

   # Generate embeddings
   embeddings = model.encode_products(products, show_progress=True)

   # Store in vector database with product IDs
   for product, embedding in zip(products, embeddings):
       vector_db.insert(product['product_id'], embedding)
   ```

3. **At query time**:
   ```python
   # User searches for something
   user_query = "hearty organic soup"

   # Generate query embedding
   query_embedding = model.encode_query(user_query)

   # Search vector database
   results = vector_db.search(query_embedding, top_k=10)
   ```

### Product Formatting

The model expects products with these fields (all optional, will use what's available):
- `title` (most important)
- `description`
- `brand`
- `category_path`
- `ingredients`
- `safety_warning`

The `encode_products()` method automatically formats these fields into the format the model was trained on.

## Performance

The fine-tuned model shows significant improvement over the baseline:

- **Spearman correlation**: Increased from baseline to fine-tuned (~0.15+ improvement)
- **Query understanding**: Better at matching user intent to products
- **Semantic matching**: Can find relevant products even with different wording

Run `python evaluate_model.py` to see full performance metrics.

## Testing the Model

### Interactive Search Demo
```bash
python search.py "organic hearty soup"
```

### Generate Single Embedding
```bash
python generate_embeddings.py "your text here" --model output/heb-semantic-search
```

### Test Model Interface
```bash
python model_interface.py
```

## Retraining (if needed)

```bash
python train_model.py
```

This will:
- Load training data from `data/`
- Fine-tune `all-MiniLM-L6-v2` for 3 epochs
- Save checkpoints to `output/heb-semantic-search/`
- Validate during training

## Notes for Vector DB Team

1. **Embedding Dimension**: 384 (use this for your vector DB schema)
2. **Similarity Metric**: Cosine similarity (model outputs are normalized)
3. **Model is Deterministic**: Same input always produces same embedding
4. **Batch Processing**: Use `batch_size` parameter for large datasets
5. **Model Size**: ~90MB (loads quickly, no GPU required for inference)

## Support

- Model training code: `train_model.py`
- Evaluation code: `evaluate_model.py`
- Search demo: `search.py`
- Main interface: `model_interface.py`

For questions about the model or integration, contact the model training team.
