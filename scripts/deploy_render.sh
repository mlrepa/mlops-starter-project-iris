#!/bin/bash

# Render deployment script for Iris ML application
set -e

echo "🚀 Deploying Iris ML application to Render..."

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

# Check if render.yaml exists
if [ ! -f "render.yaml" ]; then
    print_error "render.yaml not found! Please ensure the file exists in the root directory."
    exit 1
fi

print_status "render.yaml found ✓"

# Check if model file exists
if [ ! -f "models/model.joblib" ]; then
    print_warning "Model file not found. Running training pipeline..."
    make run-pipeline
fi

print_status "Model file verified ✓"

# Validate Docker setup
print_status "Validating Docker setup..."
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Test build locally first
print_status "Testing builds locally..."

# Build API image
print_status "Building API image..."
if ! docker build -f api/Dockerfile -t iris-api-test:latest .; then
    print_error "Failed to build API image"
    exit 1
fi

# Build UI image
print_status "Building UI image..."
if ! docker build -f ui/Dockerfile -t iris-ui-test:latest .; then
    print_error "Failed to build UI image"
    exit 1
fi

print_status "Local builds successful ✓"

# Check if Render CLI is installed
if command -v render &> /dev/null; then
    print_status "Render CLI found. Deploying..."

    # Deploy using Render CLI
    if render deploy; then
        print_status "🎉 Deployment to Render successful!"
        echo ""
        echo "🌐 Your application should be available at:"
        echo "   Check your Render dashboard for service URLs"
        echo ""
        echo "🔍 Monitor deployment:"
        echo "   render logs --service iris-api"
        echo "   render logs --service iris-ui"
    else
        print_error "Render deployment failed"
        exit 1
    fi
else
    print_warning "Render CLI not found."
    echo ""
    echo "🔧 To deploy to Render:"
    echo "   1. Install Render CLI: https://render.com/docs/cli"
    echo "   2. Login to Render: render auth login"
    echo "   3. Run this script again"
    echo ""
    echo "📋 Or deploy manually:"
    echo "   1. Push your code to GitHub"
    echo "   2. Connect your repository to Render"
    echo "   3. Render will automatically deploy using render.yaml"
    echo ""
    print_status "Local validation completed successfully!"
fi

# Cleanup test images
print_status "Cleaning up test images..."
docker rmi iris-api-test:latest iris-ui-test:latest 2>/dev/null || true

print_status "✅ Deployment process completed!"
