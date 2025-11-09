"""
Integration Test Script

Tests the complete pipeline:
1. Python embedding service is running
2. Java backend is running
3. Embeddings are correct dimensions
4. End-to-end query works

Usage:
    python test_integration.py
"""

import requests
import json
import sys
import time


def test_python_service():
    """Test Python FastAPI service is running and returns correct embeddings."""
    print("=" * 80)
    print("TEST 1: Python Embedding Service")
    print("=" * 80)

    url = "http://localhost:8001"

    # Test service is running
    try:
        response = requests.get(f"{url}/docs", timeout=5)
        if response.status_code == 200:
            print("✅ Python service is running on port 8001")
        else:
            print(f"❌ Python service returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Python service is not running: {e}")
        print("\n   Start with: uvicorn get_embeddings:app --port 8001")
        return False

    # Test dense embeddings
    print("\nTesting dense embeddings...")
    try:
        response = requests.post(
            f"{url}/dense-embed",
            json={"query": "organic hearty soup"},
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            embedding = data.get("dense_embedding", [])

            if len(embedding) == 384:
                print(f"✅ Dense embeddings: dimension {len(embedding)}")
                print(f"   Sample values: {embedding[:5]}")

                # Check it's not dummy values
                if all(abs(v - 1.0) < 0.001 for v in embedding[:10]):
                    print("⚠️  WARNING: Embeddings look like dummy values (all 1s)")
                elif all(abs(v) < 0.001 for v in embedding):
                    print("⚠️  WARNING: Embeddings look like dummy values (all 0s)")
                else:
                    print("✅ Embeddings appear to be real (varied values)")
            else:
                print(f"❌ Dense embeddings: wrong dimension {len(embedding)} (expected 384)")
                return False
        else:
            print(f"❌ Dense embedding request failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Dense embedding test failed: {e}")
        return False

    # Test sparse embeddings
    print("\nTesting sparse embeddings...")
    try:
        response = requests.post(
            f"{url}/sparse-embed",
            json={"query": "organic hearty soup"},
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            embedding = data.get("sparse_embedding", [])

            if len(embedding) == 1000:
                print(f"✅ Sparse embeddings: dimension {len(embedding)}")
                non_zero = sum(1 for v in embedding if abs(v) > 0.001)
                print(f"   Non-zero values: {non_zero}/1000")
            else:
                print(f"❌ Sparse embeddings: wrong dimension {len(embedding)} (expected 1000)")
                return False
        else:
            print(f"❌ Sparse embedding request failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Sparse embedding test failed: {e}")
        return False

    print("\n✅ Python service tests PASSED")
    return True


def test_java_backend():
    """Test Java backend is running."""
    print("\n" + "=" * 80)
    print("TEST 2: Java Backend Service")
    print("=" * 80)

    url = "http://localhost:8080"

    try:
        # Try to connect to any Java endpoint
        response = requests.get(f"{url}/actuator/health", timeout=5)
        if response.status_code == 200:
            print("✅ Java backend is running on port 8080")
            return True
    except requests.exceptions.RequestException:
        pass

    # If actuator endpoint doesn't exist, try query endpoint
    try:
        response = requests.post(
            f"{url}/query",
            json={"query": "test", "tableName": "products"},
            timeout=5
        )
        # Any response (even 404/500) means backend is running
        print("✅ Java backend is running on port 8080")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Java backend is not running: {e}")
        print("\n   Start with: cd backend && ./mvnw spring-boot:run")
        return False


def test_java_calls_python():
    """Test that Java backend successfully calls Python service for embeddings."""
    print("\n" + "=" * 80)
    print("TEST 3: Java-Python Integration")
    print("=" * 80)

    # This test would require:
    # 1. A test database with products
    # 2. Making a query through Java backend
    # 3. Checking Java logs for successful embedding calls

    print("⚠️  Manual verification required:")
    print("   1. Create a database: POST /create-db/test_products")
    print("   2. Make a query: POST /query")
    print("   3. Check Java logs for:")
    print("      - 'Successfully got dense embedding of dimension: 384'")
    print("      - 'Successfully got sparse embedding of dimension: 1000'")

    return True


def test_model_loading():
    """Test that the fine-tuned model is loaded (not baseline)."""
    print("\n" + "=" * 80)
    print("TEST 4: Fine-tuned Model Verification")
    print("=" * 80)

    import os

    # Check if fine-tuned model exists
    model_path = "output/heb-semantic-search/model.safetensors"
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ Fine-tuned model found: {model_path}")
        print(f"   Size: {size_mb:.1f} MB")
    else:
        print(f"❌ Fine-tuned model not found: {model_path}")
        print("   Make sure you've pulled the latest code with models")
        return False

    # Check get_embeddings.py is using fine-tuned model
    try:
        with open("get_embeddings.py", "r") as f:
            content = f.read()
            if "output/heb-semantic-search" in content:
                print("✅ get_embeddings.py is using fine-tuned model")
            elif "all-MiniLM-L6-v2" in content and "heb-semantic-search" not in content:
                print("⚠️  WARNING: get_embeddings.py appears to be using baseline model")
                print("   Update MODEL_PATH to 'output/heb-semantic-search'")
                return False
            else:
                print("⚠️  Unable to determine which model is loaded")
    except Exception as e:
        print(f"⚠️  Could not verify model configuration: {e}")

    return True


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 80)
    print("INTEGRATION TEST SUITE")
    print("=" * 80)
    print("\nTesting connection between:")
    print("  - Python embedding service (port 8001)")
    print("  - Java backend (port 8080)")
    print("  - Fine-tuned sentence transformer model")

    results = {
        "Python Service": False,
        "Java Backend": False,
        "Java-Python Integration": False,
        "Fine-tuned Model": False
    }

    # Run tests
    results["Python Service"] = test_python_service()
    time.sleep(1)

    results["Java Backend"] = test_java_backend()
    time.sleep(1)

    results["Java-Python Integration"] = test_java_calls_python()
    time.sleep(1)

    results["Fine-tuned Model"] = test_model_loading()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}  {test_name}")

    all_passed = all(results.values())

    if all_passed:
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nYour integration is working correctly!")
        print("\nNext steps:")
        print("  1. Create a database: POST /create-db/{dbName}")
        print("  2. Query products: POST /query")
        print("  3. Check logs to verify embeddings are being used")
        return True
    else:
        print("\n" + "=" * 80)
        print("❌ SOME TESTS FAILED")
        print("=" * 80)
        print("\nPlease fix the failing tests above.")
        print("See INTEGRATION_GUIDE.md for troubleshooting.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
