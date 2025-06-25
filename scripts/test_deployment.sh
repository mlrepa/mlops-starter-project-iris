#!/bin/bash

# Test script for deployed Iris ML application
set -e

echo "🧪 Testing deployed Iris ML application..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
API_URL=${API_URL:-"http://localhost:8000"}
UI_URL=${UI_URL:-"http://localhost:8501"}

# Function to print colored output
print_status() {
    echo -e "${GREEN}[TEST]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0

run_test() {
    local test_name="$1"
    local test_command="$2"

    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    print_status "Running: $test_name"

    if eval "$test_command"; then
        print_success "✓ $test_name PASSED"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        print_error "✗ $test_name FAILED"
        return 1
    fi
}

echo "Testing API endpoints..."

# Test 1: API Health Check
run_test "API Health Check" \
    'curl -s -f "$API_URL/health" | grep -q "healthy"'

# Test 2: Valid Prediction Request
run_test "Valid Prediction Request" \
    'curl -s -f -X POST "$API_URL/predict" \
     -H "Content-Type: application/json" \
     -d "{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}" \
     | grep -q "predicted_species"'

echo ""
echo "Testing UI endpoints..."

# Test 3: UI Accessibility
run_test "UI Accessibility" \
    'curl -s -f "$UI_URL" > /dev/null'

echo ""
echo "📊 Test Results Summary:"
echo "   Total Tests: $TOTAL_TESTS"
echo "   Passed: $PASSED_TESTS"
echo "   Failed: $((TOTAL_TESTS - PASSED_TESTS))"

if [ $PASSED_TESTS -eq $TOTAL_TESTS ]; then
    print_success "🎉 All tests passed!"
    exit 0
else
    print_error "❌ Some tests failed!"
    exit 1
fi
