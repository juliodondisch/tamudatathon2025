"""
End-to-End Accuracy Test - Baseline vs Fine-tuned Model

Tests the complete system pipeline with real vector database:
1. Loads test data (products, queries, relevance labels)
2. Creates database with products via Java backend
3. Runs queries through full pipeline (Java → Python → Vector DB)
4. Measures accuracy (correlation with true relevance)
5. Compares baseline vs fine-tuned model

Prerequisites:
- PostgreSQL running (docker-compose up)
- Java backend NOT running (script will give instructions)
- Python embedding service NOT running (script will give instructions)

Usage:
    python test_e2e_accuracy.py
"""

import json
import requests
import time
import subprocess
import sys
import os
import signal
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
import numpy as np
from sklearn.model_selection import train_test_split


def load_test_data():
    """Load products, queries, and relevance labels."""
    print("=" * 80)
    print("LOADING TEST DATA")
    print("=" * 80)

    data_dir = Path("data")

    # Load products
    with open(data_dir / "products.json", "r") as f:
        products_list = json.load(f)
    products = {p['product_id']: p for p in products_list}
    print(f"✅ Loaded {len(products):,} products")

    # Load queries
    with open(data_dir / "queries_synth_train.json", "r") as f:
        queries_list = json.load(f)
    queries = {q['query_id']: q['query'] for q in queries_list}
    print(f"✅ Loaded {len(queries):,} queries")

    # Load labels
    with open(data_dir / "labels_synth_train.json", "r") as f:
        labels = json.load(f)
    print(f"✅ Loaded {len(labels):,} query-product labels")

    # Create test examples (use same 10% split as model training)
    all_examples = []
    for label in labels:
        query_id = label['query_id']
        product_id = label['product_id']
        relevance = label['relevance']

        if query_id not in queries or product_id not in products:
            continue

        all_examples.append({
            'query_id': query_id,
            'query': queries[query_id],
            'product_id': product_id,
            'relevance': relevance
        })

    # Use same split as training
    _, test_examples = train_test_split(all_examples, test_size=0.1, random_state=42)

    # Group by query for testing
    queries_to_test = {}
    for ex in test_examples:
        query_id = ex['query_id']
        if query_id not in queries_to_test:
            queries_to_test[query_id] = {
                'query': ex['query'],
                'products': []
            }
        queries_to_test[query_id]['products'].append({
            'product_id': ex['product_id'],
            'relevance': ex['relevance']
        })

    print(f"✅ Created {len(queries_to_test):,} test queries")
    print(f"   Total query-product pairs: {len(test_examples):,}")

    return products_list, queries_to_test


def wait_for_service(url, service_name, max_wait=30):
    """Wait for a service to be available."""
    print(f"\nWaiting for {service_name} to start...")
    start_time = time.time()

    while time.time() - start_time < max_wait:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code < 500:
                print(f"✅ {service_name} is ready")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
        print(".", end="", flush=True)

    print(f"\n❌ {service_name} did not start within {max_wait} seconds")
    return False


def create_database(products, table_name="test_products"):
    """Create database with products via Java backend."""
    print(f"\n{'=' * 80}")
    print(f"CREATING DATABASE: {table_name}")
    print("=" * 80)

    url = f"http://localhost:8080/create-db/{table_name}"

    print(f"Sending {len(products):,} products to Java backend...")
    try:
        response = requests.post(url, json=products, timeout=300)

        if response.status_code == 200:
            print(f"✅ Database created successfully: {table_name}")
            return True
        else:
            print(f"❌ Database creation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Failed to create database: {e}")
        return False


def query_backend(query, table_name="test_products"):
    """Query the Java backend and get top 10 product IDs."""
    url = "http://localhost:8080/query"

    try:
        response = requests.post(
            url,
            json={"query": query, "tableName": table_name},
            timeout=30
        )

        if response.status_code == 200:
            return response.json()  # List of product IDs
        else:
            print(f"⚠️  Query failed: {response.status_code}")
            return []
    except Exception as e:
        print(f"⚠️  Query error: {e}")
        return []


def calculate_ndcg_at_k(relevances, k=10):
    """Calculate NDCG@k (Normalized Discounted Cumulative Gain)."""
    relevances = np.array(relevances[:k])

    if len(relevances) == 0:
        return 0.0

    # DCG
    dcg = relevances[0] + np.sum(relevances[1:] / np.log2(np.arange(2, len(relevances) + 1)))

    # Ideal DCG
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = ideal_relevances[0] + np.sum(ideal_relevances[1:] / np.log2(np.arange(2, len(ideal_relevances) + 1)))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_model(queries_to_test, table_name, model_name):
    """Evaluate model by querying backend and measuring accuracy."""
    print(f"\n{'=' * 80}")
    print(f"EVALUATING: {model_name}")
    print("=" * 80)

    all_true_relevances = []
    all_predicted_ranks = []
    ndcg_scores = []
    queries_evaluated = 0

    # Sample queries for faster testing (or use all)
    num_queries_to_test = min(50, len(queries_to_test))  # Test 50 queries
    query_ids = list(queries_to_test.keys())[:num_queries_to_test]

    print(f"\nTesting {num_queries_to_test} queries...")

    for i, query_id in enumerate(query_ids, 1):
        query_data = queries_to_test[query_id]
        query = query_data['query']
        true_products = query_data['products']

        # Query backend
        predicted_product_ids = query_backend(query, table_name)

        if not predicted_product_ids:
            continue

        # Calculate metrics for this query
        # For each product in ground truth, find its rank in predictions
        for true_product in true_products:
            product_id = true_product['product_id']
            relevance = true_product['relevance']

            if product_id in predicted_product_ids:
                # Rank starts at 1
                rank = predicted_product_ids.index(product_id) + 1
                all_true_relevances.append(relevance)
                all_predicted_ranks.append(rank)

        # Calculate NDCG for this query
        # Get relevances in the order they were returned
        returned_relevances = []
        for pred_id in predicted_product_ids:
            # Find relevance of this product for this query
            rel = next((p['relevance'] for p in true_products if p['product_id'] == pred_id), 0)
            returned_relevances.append(rel)

        ndcg = calculate_ndcg_at_k(returned_relevances, k=10)
        ndcg_scores.append(ndcg)

        queries_evaluated += 1

        if i % 10 == 0:
            print(f"   Processed {i}/{num_queries_to_test} queries...", flush=True)

    print(f"\n✅ Evaluated {queries_evaluated} queries")

    # Calculate overall metrics
    if len(all_true_relevances) > 0:
        # Rank correlation (lower rank = better, so negate for correlation)
        spearman_corr, _ = spearmanr(all_true_relevances, [-r for r in all_predicted_ranks])
        pearson_corr, _ = pearsonr(all_true_relevances, [-r for r in all_predicted_ranks])

        # Average NDCG
        avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0

        # Average rank by relevance level
        print(f"\n📊 Metrics:")
        print(f"   Spearman Correlation: {spearman_corr:.4f}")
        print(f"   Pearson Correlation:  {pearson_corr:.4f}")
        print(f"   Average NDCG@10:      {avg_ndcg:.4f}")

        print(f"\n📊 Average Rank by Relevance Level:")
        for rel in [0, 1, 2, 3]:
            ranks = [r for r, tr in zip(all_predicted_ranks, all_true_relevances) if tr == rel]
            if ranks:
                avg_rank = np.mean(ranks)
                print(f"   Relevance {rel}: rank {avg_rank:.2f} ({len(ranks)} products)")

        return {
            'spearman': spearman_corr,
            'pearson': pearson_corr,
            'ndcg': avg_ndcg,
            'queries_evaluated': queries_evaluated
        }
    else:
        print("❌ No valid results to evaluate")
        return None


def compare_results(baseline_results, finetuned_results):
    """Compare baseline vs fine-tuned results."""
    print(f"\n{'=' * 80}")
    print("🎯 BASELINE vs FINE-TUNED COMPARISON")
    print("=" * 80)

    if baseline_results and finetuned_results:
        print(f"\n{'Metric':<25} {'Baseline':<12} {'Fine-tuned':<12} {'Improvement':<15}")
        print("-" * 70)

        for metric in ['spearman', 'pearson', 'ndcg']:
            baseline_val = baseline_results[metric]
            finetuned_val = finetuned_results[metric]
            improvement = finetuned_val - baseline_val
            improvement_pct = (improvement / abs(baseline_val)) * 100 if baseline_val != 0 else 0

            metric_name = {
                'spearman': 'Spearman Correlation',
                'pearson': 'Pearson Correlation',
                'ndcg': 'NDCG@10'
            }[metric]

            print(f"{metric_name:<25} {baseline_val:<12.4f} {finetuned_val:<12.4f} {improvement:+.4f} ({improvement_pct:+.1f}%)")

        # Determine assessment
        spearman_imp = finetuned_results['spearman'] - baseline_results['spearman']

        print(f"\n{'=' * 80}")
        print("📝 ASSESSMENT:")
        print("=" * 80)

        if spearman_imp > 0.10:
            assessment = "🌟 EXCELLENT! Fine-tuning significantly improves search quality!"
        elif spearman_imp > 0.05:
            assessment = "✅ SIGNIFICANT improvement with fine-tuned model."
        elif spearman_imp > 0.02:
            assessment = "✅ MODERATE improvement with fine-tuned model."
        elif spearman_imp > 0:
            assessment = "⚠️  SMALL improvement. Consider more training."
        else:
            assessment = "❌ NO IMPROVEMENT in full system. Check configuration."

        print(assessment)
    else:
        print("❌ Cannot compare - missing results")


def main():
    """Main test function."""
    print("\n" + "=" * 80)
    print("END-TO-END ACCURACY TEST: Baseline vs Fine-tuned Model")
    print("=" * 80)

    # Check prerequisites
    print("\n📋 Prerequisites:")
    print("   1. PostgreSQL running (docker-compose up)")
    print("   2. Java backend NOT running (we'll start it)")
    print("   3. Python embedding service NOT running (we'll start it)")
    print("\n⚠️  This test will:")
    print("   - Test with baseline model first")
    print("   - Then test with fine-tuned model")
    print("   - Compare accuracy metrics")
    print("   - Take ~5-10 minutes to complete")

    input("\nPress Enter to continue...")

    # Load test data
    products, queries_to_test = load_test_data()

    # Test both models
    results = {}

    for model_type, model_name in [("baseline", "Baseline Model"), ("finetuned", "Fine-tuned Model")]:
        print(f"\n{'=' * 80}")
        print(f"TESTING WITH: {model_name}")
        print("=" * 80)

        input(f"\n1. Start Python service with {model_type} model:")
        print(f"   MODEL_PATH={model_type} uvicorn get_embeddings:app --port 8001")
        input("\nPress Enter when Python service is running...")

        # Verify Python service
        if not wait_for_service("http://localhost:8001/docs", "Python embedding service"):
            print("❌ Python service not available. Exiting.")
            return

        # Check which model is loaded
        try:
            response = requests.get("http://localhost:8001/model-info")
            if response.status_code == 200:
                info = response.json()
                print(f"\n✅ Loaded: {info['model_name']}")
                print(f"   Path: {info['model_path']}")
                print(f"   Dimension: {info['embedding_dimension']}")
        except:
            pass

        input("\n2. Start Java backend:")
        print("   cd backend && ./mvnw spring-boot:run")
        input("\nPress Enter when Java backend is running...")

        # Verify Java backend
        if not wait_for_service("http://localhost:8080", "Java backend"):
            print("❌ Java backend not available. Exiting.")
            return

        # Create database with unique name for this model
        table_name = f"test_{model_type}"
        if not create_database(products[:100], table_name):  # Use subset for faster testing
            print(f"❌ Failed to create database. Skipping {model_name}.")
            continue

        # Wait for database to be ready
        print("\nWaiting for database to be indexed...")
        time.sleep(5)

        # Evaluate
        results[model_type] = evaluate_model(queries_to_test, table_name, model_name)

        print(f"\n✅ {model_name} testing complete!")
        input(f"\nStop Python and Java services, then press Enter to continue to next model...")

    # Compare results
    if 'baseline' in results and 'finetuned' in results:
        compare_results(results['baseline'], results['finetuned'])

    print(f"\n{'=' * 80}")
    print("✅ END-TO-END TESTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
