"""
Sentence Transformer Model Interface for Vector Database Integration

This module provides a clean interface for the fine-tuned sentence transformer
model to be used with vector database implementations.

Usage:
    from model_interface import GrocerySearchModel

    # Initialize model
    model = GrocerySearchModel()

    # Generate embeddings for queries
    query_embedding = model.encode_query("organic soup")

    # Generate embeddings for products (batch processing recommended)
    product_embeddings = model.encode_products([
        {"title": "Organic Lentil Soup", "description": "...", ...},
        {"title": "Chicken Noodle Soup", "description": "...", ...}
    ])
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Union
from pathlib import Path


class GrocerySearchModel:
    """
    Interface for the fine-tuned sentence transformer model.

    This model is optimized for semantic search in grocery products,
    trained on query-product pairs with relevance scores.
    """

    def __init__(self, model_path: str = "output/heb-semantic-search"):
        """
        Initialize the model.

        Args:
            model_path: Path to the fine-tuned model directory
                       (default: "output/heb-semantic-search")
        """
        self.model_path = model_path
        self.model = self._load_model()
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def _load_model(self) -> SentenceTransformer:
        """Load the sentence transformer model."""
        try:
            model = SentenceTransformer(self.model_path)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")

    @staticmethod
    def format_product(product: Dict) -> str:
        """
        Format a product dictionary into searchable text.

        Args:
            product: Dictionary containing product fields
                    (title, description, brand, category_path, ingredients, safety_warning)

        Returns:
            Formatted product text string
        """
        text_parts = [
            product.get("title", ""),
            product.get("description", ""),
            f"Brand: {product.get('brand', '')}.",
            f"Category: {product.get('category_path', '')}.",
            f"Ingredients: {product.get('ingredients', '')}.",
            f"Warning: {product.get('safety_warning', '')}.",
        ]
        return " ".join([t for t in text_parts if t])

    def encode_query(
        self,
        query: Union[str, List[str]],
        convert_to_numpy: bool = True,
        normalize: bool = False,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate embeddings for search queries.

        Args:
            query: Single query string or list of query strings
            convert_to_numpy: Return numpy array instead of torch tensor
            normalize: Whether to L2-normalize embeddings
            batch_size: Batch size for encoding

        Returns:
            Embedding(s) as numpy array of shape (embedding_dim,) for single query
            or (num_queries, embedding_dim) for multiple queries
        """
        embeddings = self.model.encode(
            query,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize,
            batch_size=batch_size,
            show_progress_bar=False
        )
        return embeddings

    def encode_products(
        self,
        products: List[Dict],
        convert_to_numpy: bool = True,
        normalize: bool = False,
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for products.

        Args:
            products: List of product dictionaries
            convert_to_numpy: Return numpy array instead of torch tensor
            normalize: Whether to L2-normalize embeddings
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            Embeddings as numpy array of shape (num_products, embedding_dim)
        """
        product_texts = [self.format_product(p) for p in products]

        embeddings = self.model.encode(
            product_texts,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize,
            batch_size=batch_size,
            show_progress_bar=show_progress
        )
        return embeddings

    def encode_text(
        self,
        texts: Union[str, List[str]],
        convert_to_numpy: bool = True,
        normalize: bool = False,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate embeddings for arbitrary text.

        Args:
            texts: Single text string or list of text strings
            convert_to_numpy: Return numpy array instead of torch tensor
            normalize: Whether to L2-normalize embeddings
            batch_size: Batch size for encoding

        Returns:
            Embedding(s) as numpy array
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize,
            batch_size=batch_size,
            show_progress_bar=False
        )
        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get the dimensionality of the embeddings."""
        return self.embedding_dim

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        output_path: str,
        format: str = "npy"
    ):
        """
        Save embeddings to disk.

        Args:
            embeddings: Numpy array of embeddings
            output_path: Path to save embeddings
            format: Format to save ("npy" or "json")
        """
        if format == "npy":
            np.save(output_path, embeddings)
        elif format == "json":
            with open(output_path, 'w') as f:
                json.dump(embeddings.tolist(), f)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'npy' or 'json'")

    @staticmethod
    def load_embeddings(path: str, format: str = "npy") -> np.ndarray:
        """
        Load embeddings from disk.

        Args:
            path: Path to embeddings file
            format: Format of the file ("npy" or "json")

        Returns:
            Numpy array of embeddings
        """
        if format == "npy":
            return np.load(path)
        elif format == "json":
            with open(path, 'r') as f:
                return np.array(json.load(f))
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'npy' or 'json'")


# Example usage
if __name__ == "__main__":
    # Initialize model
    print("Loading model...")
    model = GrocerySearchModel()
    print(f"Model loaded. Embedding dimension: {model.get_embedding_dimension()}")

    # Example: Encode a query
    print("\n--- Query Encoding ---")
    query = "organic hearty soup for dinner"
    query_embedding = model.encode_query(query)
    print(f"Query: '{query}'")
    print(f"Embedding shape: {query_embedding.shape}")
    print(f"First 5 values: {query_embedding[:5]}")

    # Example: Encode products
    print("\n--- Product Encoding ---")
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

    product_embeddings = model.encode_products(sample_products, show_progress=False)
    print(f"Encoded {len(sample_products)} products")
    print(f"Embeddings shape: {product_embeddings.shape}")

    # Calculate similarity
    from sentence_transformers import util
    similarities = util.cos_sim(query_embedding, product_embeddings)[0]
    print(f"\nSimilarity scores:")
    for i, (prod, score) in enumerate(zip(sample_products, similarities)):
        print(f"  {prod['title']}: {score:.4f}")
