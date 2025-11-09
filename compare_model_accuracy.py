"""
Compare accuracy between baseline (untrained) and fine-tuned models.

This script evaluates both models on test data and shows improvement metrics.

Usage:
    python compare_accuracy.py
    python compare_accuracy.py --baseline output/baseline-model --finetuned output/heb-semantic-search
"""

import json
import numpy as np
import argparse
from model_interface_v2 import GrocerySearchModel
from sentence_transformers import util
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, pearsonr


def load_json(path):
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def load_test_data():
    """Load and prepare test data."""
    print("Loading test data...")

    products = {p['product_id']: p for p in load_json("data/products.json")}
    queries = {q['query_id']: q['query'] for q in load_json("data/queries_synth_train.json")}
    labels = load_json("data/labels_synth_train.json")

    # Create test examples
    all_examples = []
    for label in labels:
        query_id = label['query_id']
        product_id = label['product_id']
        relevance = label['relevance']

        if query_id not in queries or product_id not in products:
            continue

        query_text = queries[query_id]
        product = products[product_id]

        all_examples.append({
            'query': query_text,
            'product': product,
            'relevance': relevance,
            'query_id': query_id,
            'product_id': product_id
        })

    # Use same split as training (10% test)
    _, test_examples = train_test_split(all_examples, test_size=0.1, random_state=42)

    print(f"✅ Loaded {len(test_examples):,} test examples")
    return test_examples


def evaluate_model(model, test_examples, model_name):
    """Evaluate a model on test data."""
    print(f"\n{'=' * 80}")
    print(f"EVALUATING: {model_name}")
    print('=' * 80)

    queries = [ex['query'] for ex in test_examples]
    products = [ex['product'] for ex in test_examples]
    true_relevances = [ex['relevance'] for ex in test_examples]

    # Encode queries and products
    print("Encoding queries...")
    query_embeddings = model.encode_query(queries, batch_size=32)

    print("Encoding products...")
    product_embeddings = model.encode_products(products, batch_size=32, show_progress=True)

    # Calculate cosine similarities
    print("Calculating similarities...")
    similarities = []
    for i in range(len(test_examples)):
        sim = util.cos_sim(query_embeddings[i], product_embeddings[i]).item()
        similarities.append(sim)

    predicted_scores = np.array(similarities)

    # Calculate correlation metrics
    spearman_corr, _ = spearmanr(true_relevances, predicted_scores)
    pearson_corr, _ = pearsonr(true_relevances, predicted_scores)

    # Calculate metrics by relevance level
    print("\n📊 Metrics by Relevance Level:")
    print(f"{'Relevance':<12} {'Avg Score':<12} {'Std Dev':<12} {'Count':<10}")
    print('-' * 50)

    avg_scores_by_relevance = {}
    for rel in [0, 1, 2, 3]:
        indices = [i for i, ex in enumerate(test_examples) if ex['relevance'] == rel]
        if indices:
            scores = predicted_scores[indices]
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            avg_scores_by_relevance[rel] = avg_score
            print(f"{rel:<12} {avg_score:<12.4f} {std_score:<12.4f} {len(indices):<10,}")

    # Calculate separation between relevance levels
    if 0 in avg_scores_by_relevance and 3 in avg_scores_by_relevance:
        separation = avg_scores_by_relevance[3] - avg_scores_by_relevance[0]
    else:
        separation = 0

    print(f"\n📈 Overall Metrics:")
    print(f"  Spearman Correlation: {spearman_corr:.4f}")
    print(f"  Pearson Correlation:  {pearson_corr:.4f}")
    print(f"  Relevance Separation: {separation:.4f} (score[3] - score[0])")

    return {
        'spearman': spearman_corr,
        'pearson': pearson_corr,
        'separation': separation,
        'predicted_scores': predicted_scores,
        'avg_by_relevance': avg_scores_by_relevance
    }


def show_improvements(baseline_results, finetuned_results):
    """Show improvement statistics."""
    print(f"\n{'=' * 80}")
    print("🎯 IMPROVEMENT FROM FINE-TUNING")
    print('=' * 80)

    # Calculate improvements
    spearman_imp = finetuned_results['spearman'] - baseline_results['spearman']
    pearson_imp = finetuned_results['pearson'] - baseline_results['pearson']
    separation_imp = finetuned_results['separation'] - baseline_results['separation']

    spearman_pct = (spearman_imp / abs(baseline_results['spearman'])) * 100 if baseline_results['spearman'] != 0 else 0

    print(f"\n📊 Metric Improvements:")
    print(f"{'Metric':<25} {'Baseline':<12} {'Fine-tuned':<12} {'Improvement':<15}")
    print('-' * 65)
    print(f"{'Spearman Correlation':<25} {baseline_results['spearman']:<12.4f} {finetuned_results['spearman']:<12.4f} +{spearman_imp:.4f} ({spearman_pct:+.1f}%)")
    print(f"{'Pearson Correlation':<25} {baseline_results['pearson']:<12.4f} {finetuned_results['pearson']:<12.4f} +{pearson_imp:.4f}")
    print(f"{'Relevance Separation':<25} {baseline_results['separation']:<12.4f} {finetuned_results['separation']:<12.4f} +{separation_imp:.4f}")

    # Show score distribution changes
    print(f"\n📊 Average Scores by Relevance Level:")
    print(f"{'Relevance':<12} {'Baseline':<12} {'Fine-tuned':<12} {'Change':<12}")
    print('-' * 50)

    for rel in [0, 1, 2, 3]:
        baseline_score = baseline_results['avg_by_relevance'].get(rel, 0)
        finetuned_score = finetuned_results['avg_by_relevance'].get(rel, 0)
        change = finetuned_score - baseline_score
        print(f"{rel:<12} {baseline_score:<12.4f} {finetuned_score:<12.4f} {change:+.4f}")

    # Overall assessment
    print(f"\n{'=' * 80}")
    print("📝 ASSESSMENT:")
    print('=' * 80)

    if spearman_imp > 0.10:
        assessment = "🌟 EXCELLENT! Fine-tuning is working very well."
    elif spearman_imp > 0.05:
        assessment = "✅ SIGNIFICANT improvement. Fine-tuning is effective."
    elif spearman_imp > 0.02:
        assessment = "✅ MODERATE improvement. Fine-tuning is helping."
    elif spearman_imp > 0:
        assessment = "⚠️  SMALL improvement. Consider more training or data."
    else:
        assessment = "❌ NO IMPROVEMENT. Check training setup."

    print(assessment)
    print('=' * 80)


def show_example_comparisons(baseline_results, finetuned_results, test_examples, n=5):
    """Show specific examples where fine-tuning helped most."""
    print(f"\n{'=' * 80}")
    print("📋 EXAMPLE COMPARISONS")
    print('=' * 80)

    improvements = []
    for i, ex in enumerate(test_examples):
        baseline_score = baseline_results['predicted_scores'][i]
        finetuned_score = finetuned_results['predicted_scores'][i]
        true_rel = ex['relevance']

        # Calculate error reduction (how much closer to ideal)
        ideal_score = true_rel / 3.0  # Normalize 0-3 to 0-1
        baseline_error = abs(baseline_score - ideal_score)
        finetuned_error = abs(finetuned_score - ideal_score)
        improvement = baseline_error - finetuned_error

        improvements.append({
            'index': i,
            'example': ex,
            'baseline_score': baseline_score,
            'finetuned_score': finetuned_score,
            'improvement': improvement
        })

    # Sort by improvement
    improvements.sort(key=lambda x: x['improvement'], reverse=True)

    # Show top improvements
    print(f"\n🎯 Top {n} Most Improved Examples:")
    print('=' * 80)

    for i, item in enumerate(improvements[:n], 1):
        ex = item['example']
        product_title = ex['product'].get('title', 'N/A')

        print(f"\n{i}. Relevance: {ex['relevance']}/3")
        print(f"   Query: \"{ex['query']}\"")
        print(f"   Product: {product_title[:80]}{'...' if len(product_title) > 80 else ''}")
        print(f"   Baseline Score:   {item['baseline_score']:.4f}")
        print(f"   Fine-tuned Score: {item['finetuned_score']:.4f}")
        print(f"   Improvement: +{item['improvement']:.4f}")

    # Show regressions if any
    improvements.sort(key=lambda x: x['improvement'])
    worst_items = [item for item in improvements[:n] if item['improvement'] < 0]

    if worst_items:
        print(f"\n⚠️  Top {len(worst_items)} Regressed Examples (got worse):")
        print('=' * 80)

        for i, item in enumerate(worst_items, 1):
            ex = item['example']
            product_title = ex['product'].get('title', 'N/A')

            print(f"\n{i}. Relevance: {ex['relevance']}/3")
            print(f"   Query: \"{ex['query']}\"")
            print(f"   Product: {product_title[:80]}{'...' if len(product_title) > 80 else ''}")
            print(f"   Baseline Score:   {item['baseline_score']:.4f}")
            print(f"   Fine-tuned Score: {item['finetuned_score']:.4f}")
            print(f"   Regression: {item['improvement']:.4f}")


def main():
    """Main comparison function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Compare baseline vs fine-tuned model accuracy')
    parser.add_argument('--baseline',
                        default='output/baseline-model',
                        help='Baseline model path (default: output/baseline-model)')
    parser.add_argument('--finetuned',
                        default='output/heb-semantic-search',
                        help='Fine-tuned model path (default: output/heb-semantic-search)')
    parser.add_argument('--examples', type=int, default=5,
                        help='Number of example comparisons to show (default: 5)')
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("MODEL ACCURACY COMPARISON")
    print("=" * 80)
    print(f"Baseline Model:   {args.baseline}")
    print(f"Fine-tuned Model: {args.finetuned}")
    print("=" * 80)

    # Load test data
    test_examples = load_test_data()

    # Load and evaluate baseline model
    print(f"\n{'=' * 80}")
    print("LOADING BASELINE MODEL")
    print('=' * 80)
    baseline_model = GrocerySearchModel(model_path=args.baseline)
    print(f"✅ Baseline model loaded")

    baseline_results = evaluate_model(baseline_model, test_examples, "Baseline (Untrained)")

    # Load and evaluate fine-tuned model
    print(f"\n{'=' * 80}")
    print("LOADING FINE-TUNED MODEL")
    print('=' * 80)
    finetuned_model = GrocerySearchModel(model_path=args.finetuned)
    print(f"✅ Fine-tuned model loaded")

    finetuned_results = evaluate_model(finetuned_model, test_examples, "Fine-tuned")

    # Show improvements
    show_improvements(baseline_results, finetuned_results)

    # Show example comparisons
    show_example_comparisons(baseline_results, finetuned_results, test_examples, n=args.examples)

    print(f"\n{'=' * 80}")
    print("✅ COMPARISON COMPLETE")
    print('=' * 80)


if __name__ == "__main__":
    main()
