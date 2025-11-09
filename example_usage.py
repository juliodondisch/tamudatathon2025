"""
Simple example of using the fine-tuned sentence transformer model.

This demonstrates the most common use case: encoding queries and products
for vector database integration.

Usage:
    python example_usage.py
"""

from model_interface_v2 import GrocerySearchModel
from sentence_transformers import util
import numpy as np


def main():
    print("=" * 80)
    print("SENTENCE TRANSFORMER - EXAMPLE USAGE")
    print("=" * 80)

    # ========================================================================
    # 1. LOAD MODEL
    # ========================================================================
    print("\n1. Loading fine-tuned model...")
    model = GrocerySearchModel()
    print(f"   ✅ Model loaded")
    print(f"   Embedding dimension: {model.get_embedding_dimension()}")

    # ========================================================================
    # 2. ENCODE A QUERY
    # ========================================================================
    print("\n2. Encoding a query...")
    query = "organic hearty soup for dinner"
    query_embedding = model.encode_query(query)
    print(f"   Query: '{query}'")
    print(f"   Embedding shape: {query_embedding.shape}")
    print(f"   First 5 values: {query_embedding[:5]}")

    # ========================================================================
    # 3. ENCODE PRODUCTS
    # ========================================================================
    print("\n3. Encoding products...")
    sample_products = [
        {
            "title": "Organic Lentil Soup",
            "description": "Hearty and healthy lentil soup made with organic ingredients",
            "brand": "Amy's",
            "category_path": "Food > Canned Goods > Soup"
        },
        {
            "title": "Chocolate Cake Mix",
            "description": "Delicious chocolate cake mix for desserts",
            "brand": "Betty Crocker",
            "category_path": "Food > Baking > Cake Mix"
        },
        {
            "title": "Classic Chicken Noodle Soup",
            "description": "Traditional chicken noodle soup",
            "brand": "Campbell's",
            "category_path": "Food > Canned Goods > Soup"
        }
    ]

    product_embeddings = model.encode_products(sample_products, show_progress=False)
    print(f"   Encoded {len(sample_products)} products")
    print(f"   Embeddings shape: {product_embeddings.shape}")

    # ========================================================================
    # 4. CALCULATE SIMILARITIES
    # ========================================================================
    print("\n4. Calculating similarities...")
    similarities = util.cos_sim(query_embedding, product_embeddings)[0]

    print(f"\n   Results for query: '{query}'")
    print("   " + "-" * 70)

    # Sort by similarity
    sorted_indices = np.argsort(-similarities.cpu().numpy())

    for rank, idx in enumerate(sorted_indices, 1):
        product = sample_products[idx]
        score = similarities[idx].item()
        print(f"   {rank}. [{score:.4f}] {product['title']}")
        print(f"      Brand: {product['brand']} | Category: {product['category_path']}")

    # ========================================================================
    # 5. BATCH ENCODING EXAMPLE
    # ========================================================================
    print("\n5. Batch encoding multiple queries...")
    queries = [
        "organic soup",
        "chocolate dessert",
        "healthy dinner"
    ]

    batch_embeddings = model.encode_query(queries)
    print(f"   Encoded {len(queries)} queries")
    print(f"   Batch embeddings shape: {batch_embeddings.shape}")

    # ========================================================================
    # 6. SWITCHING TO BASELINE MODEL
    # ========================================================================
    print("\n6. Loading baseline model for comparison...")
    baseline_model = GrocerySearchModel(model_path='output/baseline-model')
    print(f"   ✅ Baseline model loaded")

    # Compare embeddings
    baseline_query_emb = baseline_model.encode_query(query)
    baseline_similarities = util.cos_sim(baseline_query_emb,
                                         baseline_model.encode_products(sample_products, show_progress=False))[0]

    print(f"\n   Comparison for query: '{query}'")
    print("   " + "-" * 70)
    print(f"   {'Product':<40} {'Fine-tuned':<12} {'Baseline':<12}")
    print("   " + "-" * 70)

    for idx, product in enumerate(sample_products):
        finetuned_score = similarities[idx].item()
        baseline_score = baseline_similarities[idx].item()
        title = product['title'][:37] + "..." if len(product['title']) > 37 else product['title']
        print(f"   {title:<40} {finetuned_score:<12.4f} {baseline_score:<12.4f}")

    # ========================================================================
    # 7. VECTOR DATABASE INTEGRATION EXAMPLE
    # ========================================================================
    print("\n" + "=" * 80)
    print("VECTOR DATABASE INTEGRATION PATTERN")
    print("=" * 80)

    print("""
    # Typical workflow for vector database integration:

    1. STARTUP - Load model once:
       model = GrocerySearchModel()

    2. INDEX PRODUCTS - Do once or when catalog updates:
       product_embeddings = model.encode_products(all_products, batch_size=32)
       # Store in vector DB with product IDs
       for product, embedding in zip(all_products, product_embeddings):
           vector_db.insert(product['product_id'], embedding)

    3. SEARCH - At query time:
       query_embedding = model.encode_query(user_query)
       results = vector_db.search(query_embedding, top_k=10)
       # Returns most similar products

    4. SIMILARITY - Using cosine similarity:
       similarities = util.cos_sim(query_embedding, product_embeddings)
       top_k = similarities.topk(k=10)
    """)

    print("=" * 80)
    print("✅ EXAMPLE COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  - Test with: python test_model_interface.py")
    print("  - Compare accuracy: python compare_model_accuracy.py")
    print("  - See full docs: SENTENCE_TRANSFORMER_README.md")
    print("=" * 80)


if __name__ == "__main__":
    main()
