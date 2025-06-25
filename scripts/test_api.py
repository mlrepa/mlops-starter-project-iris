#!/usr/bin/env python3
"""
Test script for the Iris ML API
Can be used for testing deployed application in CI/CD pipeline
"""

import os
import sys
import time

import requests


def test_api_health(base_url: str) -> bool:
    """Test API health endpoint."""
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        response.raise_for_status()
        data = response.json()
        return bool(data.get("status") == "healthy")
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


def test_prediction_endpoint(base_url: str) -> bool:
    """Test the prediction endpoint with sample data."""
    try:
        # Test data for setosa (should predict class 0)
        test_data = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        }

        response = requests.post(f"{base_url}/predict", json=test_data, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Validate response structure
        required_fields = [
            "predicted_class",
            "predicted_species",
            "confidence",
            "input_features",
        ]
        for field in required_fields:
            if field not in data:
                print(f"Missing field in response: {field}")
                return False

        # Validate prediction values
        if data["predicted_class"] not in [0, 1, 2]:
            print(f"Invalid predicted_class: {data['predicted_class']}")
            return False

        if data["predicted_species"] not in ["setosa", "versicolor", "virginica"]:
            print(f"Invalid predicted_species: {data['predicted_species']}")
            return False

        if not (0 <= data["confidence"] <= 1):
            print(f"Invalid confidence value: {data['confidence']}")
            return False

        print(
            f"✓ Prediction test passed: {data['predicted_species']} "
            f"(confidence: {data['confidence']:.2f})"
        )
        return True

    except Exception as e:
        print(f"Prediction test failed: {e}")
        return False


def test_model_info(base_url: str) -> bool:
    """Test the model info endpoint."""
    try:
        response = requests.get(f"{base_url}/model-info", timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data.get("model_loaded", False):
            print("Model is not loaded")
            return False

        print(
            f"✓ Model info test passed: {data.get('model_type', 'Unknown')} "
            f"model loaded"
        )
        return True

    except Exception as e:
        print(f"Model info test failed: {e}")
        return False


def main() -> None:
    """Main test function."""
    # Get API URL from environment or use default
    api_url = os.getenv("API_URL", "http://localhost:8000")

    print(f"🧪 Testing API at: {api_url}")

    # Wait for API to be ready (useful for CI)
    print("⏳ Waiting for API to be ready...")
    for attempt in range(30):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(2)
        if attempt == 29:
            print("❌ API not ready after 60 seconds")
            sys.exit(1)

    # Run tests
    tests = [
        ("Health Check", lambda: test_api_health(api_url)),
        ("Model Info", lambda: test_model_info(api_url)),
        ("Prediction Endpoint", lambda: test_prediction_endpoint(api_url)),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔬 Running: {test_name}")
        if test_func():
            print(f"✅ {test_name} PASSED")
            passed += 1
        else:
            print(f"❌ {test_name} FAILED")

    print(f"\n📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
