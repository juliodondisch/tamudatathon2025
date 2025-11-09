"""
Get vector embeddings - minimal output version.

Usage:
    python get_embeddings_simple.py "hearty organic soups"
    python get_embeddings_simple.py "soup" --model output/heb-semantic-search
"""
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser()
parser.add_argument('text', help='Text to encode')
parser.add_argument('--model', '-m', default='all-MiniLM-L6-v2',
                    help='Model name or path (default: all-MiniLM-L6-v2)')
parser.add_argument('--format', '-f', choices=['numpy', 'list', 'json'], default='numpy',
                    help='Output format: numpy (default), list, or json')
args = parser.parse_args()

# Load model and encode
model = SentenceTransformer(args.model)
embedding = model.encode(args.text)

# Output based on format
if args.format == 'numpy':
    print(embedding)
elif args.format == 'list':
    print(embedding.tolist())
elif args.format == 'json':
    import json
    print(json.dumps(embedding.tolist()))
