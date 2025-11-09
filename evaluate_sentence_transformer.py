"""
Compare baseline (pre-trained) model vs fine-tuned model.
This shows how much the model actually learned from fine-tuning.

Usage:
    python compare_baseline.py
    python compare_baseline.py --baseline all-MiniLM-L6-v2 --finetuned output/heb-semantic-search
"""
import json
import numpy as np
import argparse
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, pearsonr

# ============================================================================
# ARGUMENT PARSING
# ============================================================================
parser = argparse.ArgumentParser(description='Compare baseline vs fine-tuned models')
parser.add_argument('--baseline', 
                    default='all-MiniLM-L6-v2',
                    help='Baseline model name or path (default: all-MiniLM-L6-v2)')
parser.add_argument('--finetuned',
                    default='output/heb-semantic-search',
                    help='Fine-tuned model path (default: output/heb-semantic-search)')
args = parser.parse_args()

# ============================================================================
# 1. LOAD DATA
# ============================================================================
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

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

print("Loading data...")
products = {p['product_id']: p for p in load_json("data/products.json")}
queries = {q['query_id']: q['query'] for q in load_json("data/queries_synth_train.json")}
labels = load_json("data/labels_synth_train.json")

# Create test examples (same split as training for fair comparison)
from sentence_transformers import InputExample
all_examples = []

for label in labels:
    query_id = label['query_id']
    product_id = label['product_id']
    relevance = label['relevance']
    
    if query_id not in queries or product_id not in products:
        continue
    
    query_text = queries[query_id]
    product_text = format_product(products[product_id])
    
    all_examples.append({
        'query': query_text,
        'product': product_text,
        'relevance': relevance,
        'query_id': query_id,
        'product_id': product_id
    })

# Use same split as training
_, test_examples = train_test_split(all_examples, test_size=0.1, random_state=42)
print(f"Testing on {len(test_examples):,} examples")

# ============================================================================
# 2. EVALUATE MODEL
# ============================================================================
def evaluate_model(model, examples, model_name):
    """Evaluate model and return metrics."""
    print(f"\nEvaluating: {model_name}")
    print("=" * 80)
    
    queries = [ex['query'] for ex in examples]
    products = [ex['product'] for ex in examples]
    true_relevances = [ex['relevance'] for ex in examples]
    
    # Encode
    print("Encoding queries...")
    query_embeddings = model.encode(queries, show_progress_bar=True, convert_to_tensor=True)
    print("Encoding products...")
    product_embeddings = model.encode(products, show_progress_bar=True, convert_to_tensor=True)
    
    # Calculate cosine similarities
    print("Calculating similarities...")
    similarities = util.cos_sim(query_embeddings, product_embeddings)
    predicted_scores = [similarities[i][i].item() for i in range(len(examples))]
    
    # Calculate correlations
    spearman_corr, _ = spearmanr(true_relevances, predicted_scores)
    pearson_corr, _ = pearsonr(true_relevances, predicted_scores)
    
    # Calculate metrics by relevance level
    print("\nMetrics by relevance level:")
    for rel in [0, 1, 2, 3]:
        indices = [i for i, ex in enumerate(examples) if ex['relevance'] == rel]
        if indices:
            scores = [predicted_scores[i] for i in indices]
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"  Relevance {rel}: avg similarity = {avg_score:.4f} (±{std_score:.4f}) ({len(indices):,} examples)")
    
    print(f"\nOverall Metrics:")
    print(f"  Spearman correlation: {spearman_corr:.4f}")
    print(f"  Pearson correlation:  {pearson_corr:.4f}")
    
    return {
        'spearman': spearman_corr,
        'pearson': pearson_corr,
        'predicted_scores': predicted_scores
    }

# ============================================================================
# 3. COMPARE MODELS
# ============================================================================
print("\n" + "=" * 80)
print(f"BASELINE MODEL: {args.baseline}")
print("=" * 80)
try:
    baseline_model = SentenceTransformer(args.baseline)
    baseline_results = evaluate_model(baseline_model, test_examples, args.baseline)
except Exception as e:
    print(f"❌ Error loading baseline model: {e}")
    exit(1)

print("\n" + "=" * 80)
print(f"FINE-TUNED MODEL: {args.finetuned}")
print("=" * 80)
try:
    finetuned_model = SentenceTransformer(args.finetuned)
    finetuned_results = evaluate_model(finetuned_model, test_examples, args.finetuned)
    
    # ========================================================================
    # 4. SHOW IMPROVEMENT
    # ========================================================================
    print("\n" + "=" * 80)
    print("IMPROVEMENT FROM FINE-TUNING")
    print("=" * 80)
    
    spearman_improvement = finetuned_results['spearman'] - baseline_results['spearman']
    pearson_improvement = finetuned_results['pearson'] - baseline_results['pearson']
    
    improvement_pct = (spearman_improvement / abs(baseline_results['spearman'])) * 100 if baseline_results['spearman'] != 0 else 0
    
    print(f"Spearman: {baseline_results['spearman']:.4f} → {finetuned_results['spearman']:.4f} (+{spearman_improvement:.4f}, +{improvement_pct:.1f}%)")
    print(f"Pearson:  {baseline_results['pearson']:.4f} → {finetuned_results['pearson']:.4f} (+{pearson_improvement:.4f})")
    
    if spearman_improvement > 0.10:
        print("\n✅ EXCELLENT improvement! Fine-tuning is working very well.")
    elif spearman_improvement > 0.05:
        print("\n✅ Significant improvement! Fine-tuning is working well.")
    elif spearman_improvement > 0.02:
        print("\n✅ Moderate improvement. Fine-tuning is helping.")
    elif spearman_improvement > 0:
        print("\n⚠️ Small improvement. Model may need more training or data.")
    else:
        print("\n❌ No improvement or regression. Check your training setup.")
    
    # ========================================================================
    # 5. SHOW EXAMPLE COMPARISONS
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXAMPLE COMPARISONS")
    print("=" * 80)
    
    # Find examples where fine-tuning helped most
    improvements = []
    for i, ex in enumerate(test_examples):
        baseline_score = baseline_results['predicted_scores'][i]
        finetuned_score = finetuned_results['predicted_scores'][i]
        true_rel = ex['relevance']
        
        # Calculate how much closer we got to ideal
        # Ideal: relevance 3 → score 1.0, relevance 0 → score 0.0
        ideal_score = true_rel / 3.0
        baseline_error = abs(baseline_score - ideal_score)
        finetuned_error = abs(finetuned_score - ideal_score)
        improvement = baseline_error - finetuned_error
        
        improvements.append({
            'example': ex,
            'baseline_score': baseline_score,
            'finetuned_score': finetuned_score,
            'improvement': improvement
        })
    
    # Show top 5 most improved
    improvements.sort(key=lambda x: x['improvement'], reverse=True)
    
    print("\nTop 5 Most Improved Examples:")
    for i, item in enumerate(improvements[:5], 1):
        ex = item['example']
        print(f"\n{i}. Relevance: {ex['relevance']}/3")
        print(f"   Query: {ex['query']}")
        print(f"   Product: {ex['product'][:100]}...")
        print(f"   Baseline:   {item['baseline_score']:.4f}")
        print(f"   Fine-tuned: {item['finetuned_score']:.4f}")
        print(f"   Improvement: +{item['improvement']:.4f}")
    
    # Show top 5 most regressed (where it got worse)
    improvements.sort(key=lambda x: x['improvement'])
    
    print("\n\nTop 5 Most Regressed Examples (where it got worse):")
    for i, item in enumerate(improvements[:5], 1):
        ex = item['example']
        if item['improvement'] < 0:
            print(f"\n{i}. Relevance: {ex['relevance']}/3")
            print(f"   Query: {ex['query']}")
            print(f"   Product: {ex['product'][:100]}...")
            print(f"   Baseline:   {item['baseline_score']:.4f}")
            print(f"   Fine-tuned: {item['finetuned_score']:.4f}")
            print(f"   Regression: {item['improvement']:.4f}")

except Exception as e:
    print(f"\n⚠️ Could not load fine-tuned model: {e}")
    print("Make sure training has started and saved at least one checkpoint.")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
