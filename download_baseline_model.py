"""
Download and save the baseline (untrained) model for comparison.

This saves the all-MiniLM-L6-v2 model locally so you can compare
fine-tuned vs baseline performance without re-downloading.
"""

from sentence_transformers import SentenceTransformer
import os

def download_baseline_model():
    """Download and save the baseline model."""
    print("=" * 80)
    print("DOWNLOADING BASELINE MODEL")
    print("=" * 80)

    model_name = 'all-MiniLM-L6-v2'
    output_path = 'output/baseline-model'

    print(f"\nDownloading: {model_name}")
    print(f"Saving to: {output_path}")
    print("\nThis may take a minute...")

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Download and save model
    model = SentenceTransformer(model_name)
    model.save(output_path)

    print("\n" + "=" * 80)
    print("✅ BASELINE MODEL SAVED!")
    print("=" * 80)
    print(f"\nModel location: {output_path}")
    print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")
    print("\nYou can now use this model for comparisons without re-downloading.")
    print("\nUsage in model_interface.py:")
    print(f"    model = GrocerySearchModel(model_path='{output_path}')")
    print("=" * 80)

if __name__ == "__main__":
    download_baseline_model()
