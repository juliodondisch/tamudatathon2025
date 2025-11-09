"""
Test script for the GrocerySearchModel interface.

This demonstrates how to use the model and validates it's working correctly.

Usage:
    python test_model.py                                    # Test fine-tuned model (default)
    python test_model.py --model baseline                   # Test baseline model
    python test_model.py --model output/heb-semantic-search # Test specific model path
"""

import json
import numpy as np
import argparse
from model_interface_v2 import GrocerySearchModel
from sentence_transformers import util


def load_json(path):
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def test_model_loading(model_path):
    """Test 1: Model loads successfully."""
    print("=" * 80)
    print("TEST 1: Model Loading")
    print("=" * 80)

    try:
        model = GrocerySearchModel(model_path=model_path)
        print(f"✅ Model loaded successfully")
        print(f"   Model path: {model_path}")
        print(f"   Embedding dimension: {model.get_embedding_dimension()}")
        assert model.get_embedding_dimension() == 384, "Expected 384 dimensions"
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise


def test_query_encoding(model):
    """Test 2: Query encoding works."""
    print("\n" + "=" * 80)
    print("TEST 2: Query Encoding")
    print("=" * 80)

    queries = [
        "organic soup",
        "hearty lentil soup for dinner",
        "gluten free pasta"
    ]

    for query in queries:
        embedding = model.encode_query(query)
        print(f"✅ Query: '{query}'")
        print(f"   Shape: {embedding.shape}")
        print(f"   Type: {type(embedding)}")
        print(f"   Sample values: {embedding[:3]}")

        assert embedding.shape == (384,), f"Expected shape (384,), got {embedding.shape}"
        assert isinstance(embedding, np.ndarray), f"Expected numpy array, got {type(embedding)}"

    print(f"\n✅ All {len(queries)} queries encoded successfully")


def test_batch_query_encoding(model):
    """Test 3: Batch query encoding."""
    print("\n" + "=" * 80)
    print("TEST 3: Batch Query Encoding")
    print("=" * 80)

    queries = [
        "organic soup",
        "chocolate cake",
        "fresh vegetables"
    ]

    embeddings = model.encode_query(queries)
    print(f"✅ Encoded {len(queries)} queries in batch")
    print(f"   Shape: {embeddings.shape}")

    assert embeddings.shape == (len(queries), 384), f"Expected shape ({len(queries)}, 384)"
    print("✅ Batch encoding works correctly")


def test_product_encoding(model):
    """Test 4: Product encoding with real products."""
    print("\n" + "=" * 80)
    print("TEST 4: Product Encoding")
    print("=" * 80)

    # Load real products from data
    try:
        products_data = load_json("data/products.json")
        sample_products = products_data[:5]  # Test with first 5 products

        embeddings = model.encode_products(sample_products, show_progress=False)

        print(f"✅ Encoded {len(sample_products)} products")
        print(f"   Shape: {embeddings.shape}")
        print(f"\n   Sample products:")
        for i, prod in enumerate(sample_products[:3]):
            print(f"   {i+1}. {prod.get('title', 'N/A')}")

        assert embeddings.shape == (len(sample_products), 384)
        print("\n✅ Product encoding works correctly")

        return sample_products, embeddings

    except FileNotFoundError:
        print("⚠️  data/products.json not found, using sample products")

        sample_products = [
            {
                "title": "Organic Lentil Soup",
                "description": "Hearty and healthy lentil soup",
                "brand": "Amy's",
                "category_path": "Food > Canned Goods > Soup"
            },
            {
                "title": "Chocolate Cake Mix",
                "description": "Delicious dessert mix",
                "brand": "Betty Crocker",
                "category_path": "Food > Baking > Cake Mix"
            }
        ]

        embeddings = model.encode_products(sample_products, show_progress=False)
        print(f"✅ Encoded {len(sample_products)} sample products")
        print(f"   Shape: {embeddings.shape}")

        return sample_products, embeddings


def test_semantic_search(model, products, product_embeddings):
    """Test 5: Semantic search functionality."""
    print("\n" + "=" * 80)
    print("TEST 5: Semantic Search")
    print("=" * 80)

    test_queries = [
        "hearty soup for dinner",
        "chocolate dessert",
        "organic healthy food"
    ]

    for query in test_queries:
        print(f"\n📝 Query: '{query}'")

        # Encode query
        query_embedding = model.encode_query(query)

        # Calculate similarities
        similarities = util.cos_sim(query_embedding, product_embeddings)[0]

        # Get top 3 results
        top_indices = np.argsort(-similarities.cpu().numpy())[:3]

        print(f"   Top 3 results:")
        for i, idx in enumerate(top_indices, 1):
            score = similarities[idx].item()
            title = products[idx].get('title', 'N/A')
            print(f"   {i}. [{score:.4f}] {title}")

        # Verify scores are in valid range (cosine similarity is between -1 and 1)
        assert all(-1 <= s <= 1 for s in similarities), "Similarity scores should be in [-1, 1]"

    print("\n✅ Semantic search works correctly")


def test_similarity_sanity(model):
    """Test 6: Sanity check - similar queries should have high similarity."""
    print("\n" + "=" * 80)
    print("TEST 6: Similarity Sanity Check")
    print("=" * 80)

    # Similar queries
    query1 = "organic soup"
    query2 = "organic soup for dinner"

    # Different queries
    query3 = "chocolate cake"

    emb1 = model.encode_query(query1)
    emb2 = model.encode_query(query2)
    emb3 = model.encode_query(query3)

    sim_similar = util.cos_sim(emb1, emb2).item()
    sim_different = util.cos_sim(emb1, emb3).item()

    print(f"Similarity between similar queries:")
    print(f"  '{query1}' <-> '{query2}': {sim_similar:.4f}")
    print(f"\nSimilarity between different queries:")
    print(f"  '{query1}' <-> '{query3}': {sim_different:.4f}")

    assert sim_similar > sim_different, "Similar queries should have higher similarity"
    assert sim_similar > 0.5, "Similar queries should have similarity > 0.5"

    print(f"\n✅ Sanity check passed!")
    print(f"   Similar queries: {sim_similar:.4f} > Different queries: {sim_different:.4f}")


def test_embedding_persistence(model):
    """Test 7: Saving and loading embeddings."""
    print("\n" + "=" * 80)
    print("TEST 7: Embedding Persistence")
    print("=" * 80)

    import tempfile
    import os

    # Create embeddings
    query = "test query"
    embedding = model.encode_query(query)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp:
        tmp_path = tmp.name

    try:
        model.save_embeddings(embedding, tmp_path, format='npy')
        print(f"✅ Saved embedding to {tmp_path}")

        # Load it back
        loaded_embedding = model.load_embeddings(tmp_path, format='npy')
        print(f"✅ Loaded embedding from {tmp_path}")

        # Verify they're the same
        assert np.allclose(embedding, loaded_embedding), "Loaded embedding doesn't match original"
        print("✅ Embeddings match after save/load")

    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_deterministic_encoding(model):
    """Test 8: Same input produces same output."""
    print("\n" + "=" * 80)
    print("TEST 8: Deterministic Encoding")
    print("=" * 80)

    query = "organic hearty soup"

    # Encode same query multiple times
    emb1 = model.encode_query(query)
    emb2 = model.encode_query(query)
    emb3 = model.encode_query(query)

    # Check they're identical
    assert np.allclose(emb1, emb2), "Encodings should be deterministic"
    assert np.allclose(emb2, emb3), "Encodings should be deterministic"

    print(f"✅ Model is deterministic")
    print(f"   Same query encoded 3 times produces identical embeddings")


def run_all_tests(model_path):
    """Run all tests."""
    print("\n" + "=" * 80)
    print("GROCERY SEARCH MODEL - TEST SUITE")
    print("=" * 80)

    try:
        # Test 1: Load model
        model = test_model_loading(model_path)

        # Test 2: Query encoding
        test_query_encoding(model)

        # Test 3: Batch query encoding
        test_batch_query_encoding(model)

        # Test 4: Product encoding
        products, product_embeddings = test_product_encoding(model)

        # Test 5: Semantic search
        test_semantic_search(model, products, product_embeddings)

        # Test 6: Similarity sanity
        test_similarity_sanity(model)

        # Test 7: Persistence
        test_embedding_persistence(model)

        # Test 8: Deterministic
        test_deterministic_encoding(model)

        # Summary
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nThe model is working correctly and ready for production use.")
        print("Vector DB team can safely integrate using model_interface.py")
        print("=" * 80)

        return True

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test the GrocerySearchModel')
    parser.add_argument('--model', '-m',
                        default='output/heb-semantic-search',
                        help='Model path or "baseline" for baseline model (default: output/heb-semantic-search)')
    args = parser.parse_args()

    # Handle "baseline" shortcut
    if args.model == 'baseline':
        model_path = 'output/baseline-model'
    else:
        model_path = args.model

    print(f"\n🧪 Testing model: {model_path}")

    success = run_all_tests(model_path)
    exit(0 if success else 1)
