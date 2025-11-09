"""
Use a model to search for products.
Usage:
    python search.py "hearty organic soups for dinner"
    python search.py "organic soup" --model output/heb-semantic-search
    python search.py "organic soup" --model all-MiniLM-L6-v2  # Use untrained baseline
"""
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
import sys
import argparse

# ============================================================================
# ARGUMENT PARSING
# ============================================================================
parser = argparse.ArgumentParser(description='Search products using a sentence transformer model')
parser.add_argument('query', nargs='*', help='Search query')
parser.add_argument('--model', '-m',
                    default='output/heb-semantic-search', 
                    help='Model name or path (default: output/heb-semantic-search). Use "all-MiniLM-L6-v2" for untrained baseline.')
parser.add_argument('--top-k', '-k', type=int, default=10,
                    help='Number of results to return (default: 10)')
args = parser.parse_args()

# ============================================================================
# 1. LOAD MODEL AND DATA
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

print(f"Loading model: {args.model}")
model = SentenceTransformer(args.model)
print(f"✓ Model loaded (embedding dim: {model.get_sentence_embedding_dimension()})")

print("Loading products...")
products = load_json("data/products.json")
print(f"Loaded {len(products)} products")

# ============================================================================
# 2. PRECOMPUTE PRODUCT EMBEDDINGS (DO THIS ONCE)
# ============================================================================
print("\nComputing product embeddings...")
product_texts = [format_product(p) for p in products]
product_embeddings = model.encode(product_texts, convert_to_tensor=True, show_progress_bar=True)
print(f"Embeddings shape: {product_embeddings.shape}")

# ============================================================================
# 3. SEARCH FUNCTION
# ============================================================================
def search_products(query, top_k=None):
    """
    Search for products matching the query.
    
    Args:
        query: Search query string
        top_k: Number of results to return (uses args.top_k if None)
        
    Returns:
        List of (product, score) tuples
    """
    if top_k is None:
        top_k = args.top_k
        
    # Encode query
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Calculate cosine similarity
    cos_scores = util.cos_sim(query_embedding, product_embeddings)[0]
    
    # Get top k results
    top_results = np.argsort(-cos_scores.cpu().numpy())[:top_k]
    
    results = []
    for idx in top_results:
        product = products[idx]
        score = cos_scores[idx].item()
        results.append((product, score))
    
    return results

# ============================================================================
# 4. DISPLAY RESULTS
# ============================================================================
def display_results(query, results):
    """Pretty print search results."""
    print("\n" + "="*80)
    print(f"SEARCH QUERY: {query}")
    print(f"MODEL: {args.model}")
    print("="*80)
    
    for i, (product, score) in enumerate(results, 1):
        print(f"\n{i}. [{score:.4f}] {product.get('title', 'N/A')}")
        print(f"   Brand: {product.get('brand', 'N/A')}")
        print(f"   Category: {product.get('category_path', 'N/A')}")
        if product.get('description'):
            desc = product['description'][:150]
            print(f"   Description: {desc}...")
    
    print("\n" + "="*80)

# ============================================================================
# 5. MAIN - INTERACTIVE OR COMMAND LINE
# ============================================================================
if __name__ == "__main__":
    # Check if query provided via command line
    if args.query:
        query = " ".join(args.query)
        results = search_products(query)
        display_results(query, results)
    else:
        # Interactive mode
        print("\n" + "="*80)
        print("PRODUCT SEARCH - Interactive Mode")
        print(f"Using model: {args.model}")
        print("="*80)
        print("Enter your search queries (or 'quit' to exit)")
        
        while True:
            query = input("\nSearch: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            results = search_products(query)
            display_results(query, results)
