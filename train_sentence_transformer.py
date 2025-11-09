"""
Fine-tune a sentence transformer for product search.
Features:
- Trains on query-product pairs with relevance scores (0-3)
- Validates during training
- Saves best model
"""
import json
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# ============================================================================
# 1. LOAD DATA
# ============================================================================
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

print("Loading data...")
products = {p['product_id']: p for p in load_json("data/products.json")}
queries_train = {q['query_id']: q['query'] for q in load_json("data/queries_synth_train.json")}
labels_train = load_json("data/labels_synth_train.json")

print(f"Loaded {len(products)} products")
print(f"Loaded {len(queries_train)} queries")
print(f"Loaded {len(labels_train)} labels")

# ============================================================================
# 2. FORMAT PRODUCTS
# ============================================================================
def format_product(product):
    """Combine product fields into searchable text."""
    text_parts = [
        product.get("title", ""),
        product.get("description", ""),
        f"Brand: {product.get('brand', '')}.",
        f"Category: {product.get('category_path', '')}.",
        f"Ingredients: {product.get('ingredients', '')}.",
        f"Warning: {product.get('safety_warning', '')}.",
    ]
    return " ".join([t for t in text_parts if t])

# ============================================================================
# 3. CREATE TRAINING EXAMPLES
# ============================================================================
print("\nCreating training examples...")
train_examples = []

for label in labels_train:
    query_id = label['query_id']
    product_id = label['product_id']
    relevance = label['relevance']
    
    if query_id not in queries_train or product_id not in products:
        continue
    
    query_text = queries_train[query_id]
    product_text = format_product(products[product_id])
    normalized_relevance = relevance / 3.0  # scale 0-3 → 0-1
    
    train_examples.append(
        InputExample(texts=[query_text, product_text], label=normalized_relevance)
    )

print(f"Created {len(train_examples)} training examples")

# ============================================================================
# 4. ANALYZE DATA DISTRIBUTION
# ============================================================================
from collections import Counter
relevance_counts = Counter([ex.label for ex in train_examples])
print("\nRelevance distribution:")
for score in sorted(relevance_counts.keys()):
    count = relevance_counts[score]
    percentage = (count / len(train_examples)) * 100
    print(f"  {score:.2f}: {count:,} ({percentage:.1f}%)")

# ============================================================================
# 5. SPLIT INTO TRAIN/VAL
# ============================================================================
print("\nSplitting into train/validation sets...")
train_examples, val_examples = train_test_split(
    train_examples, 
    test_size=0.1,  # 10% for validation
    random_state=42
)

print(f"Training examples: {len(train_examples)}")
print(f"Validation examples: {len(val_examples)}")

# Show example
print("\n" + "="*80)
print("EXAMPLE TRAINING PAIR:")
print("="*80)
print(f"Query: {train_examples[0].texts[0]}")
print(f"Product: {train_examples[0].texts[1][:200]}...")
print(f"Relevance: {train_examples[0].label:.2f}")
print("="*80)

# ============================================================================
# 6. LOAD PRE-TRAINED MODEL
# ============================================================================
print("\nLoading pre-trained model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"Model embedding dimension: {model.get_sentence_embedding_dimension()}")

# ============================================================================
# 7. SETUP TRAINING
# ============================================================================
print("\nSetting up training...")
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

# Create evaluator
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    val_examples, 
    name='validation'
)

print(f"Batches per epoch: {len(train_dataloader)}")

# ============================================================================
# 8. TRAIN MODEL
# ============================================================================
print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)

num_epochs = 3
warmup_steps = 100
output_path = 'output/heb-semantic-search'

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path=output_path,
    evaluator=evaluator,
    evaluation_steps=500,  # Evaluate every 500 steps
    save_best_model=True,  # Save model with best validation score
    show_progress_bar=True
)

print("\n" + "="*80)
print(f"Training complete! Model saved to: {output_path}")
print("="*80)

# ============================================================================
# 9. FINAL EVALUATION
# ============================================================================
print("\nRunning final evaluation on validation set...")
final_score = evaluator(model)
print(f"Final validation score: {final_score}")

print("\nDone! You can now use the model with search.py")