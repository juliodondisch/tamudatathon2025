"""
Generate predictions for HEB test queries using fine-tuned model.

This script:
1. Loads test queries (queries_synth_test.json)
2. Loads all products (products.json)
3. Uses fine-tuned model to find most relevant products for each query
4. Outputs predictions in the same format as training labels

Output format (same as labels_synth_train.json):
[
  {"query_id": "s7", "product_id": "123", "relevance": 3},
  {"query_id": "s7", "product_id": "456", "relevance": 2},
  ...
]

Usage:
    python generate_predictions.py
    python generate_predictions.py --output predictions.json
    python generate_predictions.py --model baseline  # Use baseline model
"""

import json
import numpy as np
import argparse
from pathlib import Path
from model_interface_v2 import GrocerySearchModel
from sentence_transformers import util
from tqdm import tqdm


def load_json(path):
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data, path):
    """Save JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved {len(data):,} predictions to {path}")


def generate_predictions(model, test_queries, products, products_per_query=50):
    """
    Generate predictions for test queries.

    Args:
        model: GrocerySearchModel instance
        test_queries: List of test query dicts
        products: List of all product dicts
        products_per_query: Number of products to predict per query

    Returns:
        List of prediction dicts in label format
    """
    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS")
    print("=" * 80)

    # Encode all products once
    print(f"\nEncoding {len(products):,} products...")
    product_embeddings = model.encode_products(products, batch_size=32, show_progress=True)
    print(f"✅ Product embeddings shape: {product_embeddings.shape}")

    # Generate predictions for each query
    predictions = []

    print(f"\nProcessing {len(test_queries):,} test queries...")
    for query_data in tqdm(test_queries, desc="Generating predictions"):
        query_id = query_data['query_id']
        query_text = query_data['query']

        # Encode query
        query_embedding = model.encode_query(query_text)

        # Calculate similarities
        similarities = util.cos_sim(query_embedding, product_embeddings)[0]

        # Get top products
        top_indices = np.argsort(-similarities.cpu().numpy())[:products_per_query]
        top_scores = similarities[top_indices].cpu().numpy()

        # Assign relevance scores based on similarity
        # Map similarity scores to relevance levels (0-3)
        # This is a heuristic - adjust thresholds based on your data
        for idx, score in zip(top_indices, top_scores):
            product_id = products[idx]['product_id']

            # Convert similarity score to relevance (0-3 scale)
            # Higher similarity = higher relevance
            if score >= 0.7:
                relevance = 3  # Highly relevant
            elif score >= 0.5:
                relevance = 2  # Moderately relevant
            elif score >= 0.3:
                relevance = 1  # Slightly relevant
            else:
                relevance = 0  # Not relevant

            predictions.append({
                'query_id': query_id,
                'product_id': product_id,
                'relevance': relevance
            })

    print(f"\n✅ Generated {len(predictions):,} predictions")
    print(f"   Queries: {len(test_queries):,}")
    print(f"   Products per query: {products_per_query}")

    # Show relevance distribution
    relevance_counts = {}
    for pred in predictions:
        rel = pred['relevance']
        relevance_counts[rel] = relevance_counts.get(rel, 0) + 1

    print(f"\n📊 Relevance Distribution:")
    for rel in sorted(relevance_counts.keys()):
        count = relevance_counts[rel]
        percentage = (count / len(predictions)) * 100
        print(f"   Relevance {rel}: {count:,} ({percentage:.1f}%)")

    return predictions


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate predictions for test queries')
    parser.add_argument('--model', '-m',
                        default='output/heb-semantic-search',
                        help='Model path (default: output/heb-semantic-search, use "baseline" for baseline model)')
    parser.add_argument('--output', '-o',
                        default='predictions.json',
                        help='Output file path (default: predictions.json)')
    parser.add_argument('--products-per-query', '-p',
                        type=int,
                        default=50,
                        help='Number of products to predict per query (default: 50)')
    parser.add_argument('--queries-file',
                        default='data/queries_synth_test.json',
                        help='Test queries file (default: data/queries_synth_test.json)')
    parser.add_argument('--products-file',
                        default='data/products.json',
                        help='Products file (default: data/products.json)')
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("HEB PREDICTION GENERATOR")
    print("=" * 80)

    # Handle model path shortcuts
    if args.model == 'baseline':
        model_path = 'output/baseline-model'
    elif args.model == 'finetuned':
        model_path = 'output/heb-semantic-search'
    else:
        model_path = args.model

    print(f"\nConfiguration:")
    print(f"  Model: {model_path}")
    print(f"  Test queries: {args.queries_file}")
    print(f"  Products: {args.products_file}")
    print(f"  Output: {args.output}")
    print(f"  Products per query: {args.products_per_query}")

    # Load data
    print(f"\n{'=' * 80}")
    print("LOADING DATA")
    print("=" * 80)

    test_queries = load_json(args.queries_file)
    print(f"✅ Loaded {len(test_queries):,} test queries")

    products = load_json(args.products_file)
    print(f"✅ Loaded {len(products):,} products")

    # Load model
    print(f"\n{'=' * 80}")
    print("LOADING MODEL")
    print("=" * 80)

    model = GrocerySearchModel(model_path=model_path)
    print(f"✅ Model loaded: {model_path}")
    print(f"   Embedding dimension: {model.get_embedding_dimension()}")

    # Generate predictions
    predictions = generate_predictions(
        model,
        test_queries,
        products,
        products_per_query=args.products_per_query
    )

    # Save predictions
    print(f"\n{'=' * 80}")
    print("SAVING PREDICTIONS")
    print("=" * 80)

    save_json(predictions, args.output)

    # Verify format
    print(f"\n✅ Predictions saved in correct format:")
    print(f"   Sample: {predictions[0]}")

    print(f"\n{'=' * 80}")
    print("✅ DONE")
    print("=" * 80)
    print(f"\nPredictions saved to: {args.output}")
    print(f"Total predictions: {len(predictions):,}")
    print(f"\nYou can now submit this file!")


if __name__ == "__main__":
    main()
