#!/bin/bash

# Local deployment script for Iris ML application
set -e

echo "🚀 Starting local deployment of Iris ML application..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if model file exists
if [ ! -f "models/model.joblib" ]; then
    print_warning "Model file not found. Running training pipeline..."
    make run-pipeline
fi

print_status "Model file verified ✓"

# Stop any existing containers
print_status "Stopping existing containers..."
docker-compose -f docker-compose.local.yml down 2>/dev/null || true

# Build and start containers
print_status "Building and starting containers..."
docker-compose -f docker-compose.local.yml up --build -d

# Wait for services to be ready
print_status "Waiting for services to start..."
sleep 10

# Health check for API
print_status "Checking API health..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null; then
        print_status "API is healthy ✓"
        break
    fi
    if [ $i -eq 30 ]; then
        print_error "API health check failed"
        docker-compose -f docker-compose.local.yml logs iris-api
        exit 1
    fi
    sleep 2
done

# Health check for UI
print_status "Checking UI health..."
for i in {1..30}; do
    if curl -s http://localhost:8501 > /dev/null; then
        print_status "UI is healthy ✓"
        break
    fi
    if [ $i -eq 30 ]; then
        print_error "UI health check failed"
        docker-compose -f docker-compose.local.yml logs iris-ui
        exit 1
    fi
    sleep 2
done

print_status "🎉 Local deployment successful!"
echo ""
echo "📱 Application URLs:"
echo "   API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo "   UI: http://localhost:8501"
echo ""
echo "🧪 To run tests:"
echo "   ./scripts/test_deployment.sh"
echo ""
echo "🛑 To stop:"
echo "   docker-compose -f docker-compose.local.yml down"
