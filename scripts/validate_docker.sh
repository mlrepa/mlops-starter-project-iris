#!/bin/bash

# Validate Docker builds script
set -e

echo "🔧 Validating Docker builds..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

print_status "Docker is running ✓"

# Ensure model file exists (create dummy if needed for testing)
if [ ! -f "models/model.joblib" ]; then
    print_status "Creating dummy model file for testing..."
    mkdir -p models
    echo "dummy model content" > models/model.joblib
fi

print_status "Model file present ✓"

# Test API build (build from root directory with api context)
print_status "Testing API Docker build..."
if docker build -f api/Dockerfile -t iris-api-test:latest .; then
    print_success "API build successful ✓"
else
    print_error "API build failed ✗"
    exit 1
fi

# Test UI build (build from root directory with ui context)
print_status "Testing UI Docker build..."
if docker build -f ui/Dockerfile -t iris-ui-test:latest .; then
    print_success "UI build successful ✓"
else
    print_error "UI build failed ✗"
    exit 1
fi

# Clean up test images
print_status "Cleaning up test images..."
docker rmi iris-api-test:latest iris-ui-test:latest 2>/dev/null || true

print_success "🎉 All Docker builds validated successfully!"
echo ""
echo "✅ Ready for deployment with:"
echo "   make deploy-local"
