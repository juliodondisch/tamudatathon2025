"""
Convert predictions.json to HEB submission format.

This script converts from:
  {"query_id": "s7", "product_id": "123", "relevance": 3}

To the required submission format:
  {"query_id": "s7", "rank": 1, "product_id": "123"}

Products are ranked by relevance (highest first), then by similarity score.

Usage:
    python convert_to_submission_format.py
    python convert_to_submission_format.py --input predictions.json --output submission.json
"""

import json
import argparse
from collections import defaultdict


def convert_predictions_to_submission(predictions):
    """
    Convert predictions to submission format.

    Args:
        predictions: List of dicts with query_id, product_id, relevance

    Returns:
        List of dicts with query_id, rank, product_id
    """
    # Group predictions by query_id
    by_query = defaultdict(list)
    for pred in predictions:
        by_query[pred['query_id']].append(pred)

    # For each query, sort by relevance (descending) and assign ranks
    submission = []

    for query_id in sorted(by_query.keys()):
        query_preds = by_query[query_id]

        # Sort by relevance (descending)
        query_preds.sort(key=lambda x: x['relevance'], reverse=True)

        # Assign sequential ranks
        for rank, pred in enumerate(query_preds, start=1):
            submission.append({
                'query_id': query_id,
                'rank': rank,
                'product_id': pred['product_id']
            })

    return submission


def validate_submission(submission, min_products_per_query=10):
    """Basic validation of submission format."""
    errors = []

    # Check each entry has required fields
    for i, entry in enumerate(submission):
        if not all(k in entry for k in ['query_id', 'rank', 'product_id']):
            errors.append(f"Entry {i}: Missing required fields")

    # Check ranks are sequential per query
    by_query = defaultdict(list)
    for entry in submission:
        by_query[entry['query_id']].append(entry['rank'])

    for query_id, ranks in by_query.items():
        ranks.sort()
        expected = list(range(1, len(ranks) + 1))
        if ranks != expected:
            errors.append(f"Query {query_id}: Ranks not sequential (expected {expected}, got {ranks})")

        if len(ranks) < min_products_per_query:
            errors.append(f"Query {query_id}: Only {len(ranks)} products (need at least {min_products_per_query})")

    return errors


def main():
    parser = argparse.ArgumentParser(description='Convert predictions to HEB submission format')
    parser.add_argument('--input', '-i',
                        default='predictions.json',
                        help='Input predictions file (default: predictions.json)')
    parser.add_argument('--output', '-o',
                        default='submission.json',
                        help='Output submission file (default: submission.json)')
    args = parser.parse_args()

    print("=" * 80)
    print("CONVERTING TO SUBMISSION FORMAT")
    print("=" * 80)

    # Load predictions
    print(f"\nLoading predictions from: {args.input}")
    with open(args.input, 'r') as f:
        predictions = json.load(f)
    print(f"✅ Loaded {len(predictions):,} predictions")

    # Group by query to show stats
    by_query = defaultdict(list)
    for pred in predictions:
        by_query[pred['query_id']].append(pred)

    print(f"   Queries: {len(by_query):,}")
    print(f"   Avg products per query: {len(predictions) / len(by_query):.1f}")

    # Convert to submission format
    print("\nConverting to submission format...")
    submission = convert_predictions_to_submission(predictions)
    print(f"✅ Converted to {len(submission):,} ranked results")

    # Validate
    print("\nValidating submission format...")
    errors = validate_submission(submission)

    if errors:
        print(f"❌ Validation errors found:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"   - {error}")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more errors")
        return False
    else:
        print("✅ Validation passed!")

    # Save submission
    print(f"\nSaving submission to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(submission, f, indent=2)

    print(f"✅ Saved {len(submission):,} entries")

    # Show sample
    print(f"\n📋 Sample entries:")
    for entry in submission[:5]:
        print(f"   {entry}")

    print("\n" + "=" * 80)
    print("✅ CONVERSION COMPLETE")
    print("=" * 80)
    print(f"\nSubmission file ready: {args.output}")
    print(f"Total entries: {len(submission):,}")
    print(f"Queries covered: {len(by_query):,}")

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
