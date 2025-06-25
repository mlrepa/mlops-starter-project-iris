# Deployment Guide

This guide covers how to deploy the Iris ML application both locally and to Render.

## Architecture Overview

The application consists of two main components:
- **API Service** (FastAPI): Serves ML model predictions via REST API
- **UI Service** (Streamlit): Provides interactive web interface for users


## Local Deployment

### Prerequisites

- Docker and Docker Compose installed
- Python 3.11+
- Make (for using Makefile commands)
- uv (for dependency management)

### Quick Start

1. **Train the model** (if not already done):
   ```bash
   make run-pipeline
   ```

2. **Deploy locally**:
   ```bash
   make deploy-local
   ```

3. **Test the deployment**:
   ```bash
   make test-deployment
   ```

### Manual Local Deployment

If you prefer to run commands manually:

```bash
# Build and start services
docker-compose -f docker-compose.local.yml up --build -d

# Check logs
docker-compose -f docker-compose.local.yml logs -f

# Test the services
python scripts/test_api.py

# Stop services
docker-compose -f docker-compose.local.yml down
```

### Local URLs

Once deployed locally, access the application at:
- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **UI**: http://localhost:8501

## Render Deployment

### Prerequisites

1. **GitHub Repository**: Push your code to GitHub
2. **Render Account**: Create account at [render.com](https://render.com)
3. **Render CLI** (optional): Install for command-line deployment

### Deployment Methods

#### Method 1: Automatic Deployment (Recommended)

1. **Push to deployment branch**:
   ```bash
   git checkout deployment
   git push origin deployment
   ```

2. **GitHub Actions will automatically**:
   - Run tests
   - Train and evaluate the model
   - Build Docker images
   - Test the deployment
   - Ready for production

#### Method 2: Manual Render Dashboard

1. **Connect Repository**:
   - Go to Render Dashboard
   - Click "New" → "Blueprint"
   - Connect your GitHub repository
   - Select the `render.yaml` file

2. **Deploy Services**:
   - Render will automatically create both services
   - Monitor deployment progress in dashboard

#### Method 3: Render CLI

1. **Install Render CLI**:
   ```bash
   # macOS
   brew install render

   # Other platforms: see https://render.com/docs/cli
   ```

2. **Login and Deploy**:
   ```bash
   render auth login
   make deploy-render
   ```

### Render Configuration

The deployment is configured via `render.yaml`:

```yaml
services:
  # FastAPI Backend
  - type: web
    name: iris-api
    runtime: docker
    dockerfilePath: ./api/Dockerfile
    plan: free

  # Streamlit Frontend
  - type: web
    name: iris-ui
    runtime: docker
    dockerfilePath: ./ui/Dockerfile
    plan: free
    envVars:
      - key: API_URL
        fromService:
          name: iris-api
          property: url
```

### Production URLs

After deployment, your services will be available at:
- **API**: `https://iris-api.onrender.com`
- **UI**: `https://iris-ui.onrender.com`

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) includes:

1. **Test Job**: Code quality, linting, unit tests
2. **Train Model Job**: Data processing and model training
3. **Evaluate Model Job**: Model evaluation and metrics
4. **Deploy Job** (deployment branch only): Build and test deployment

### Triggering Deployment

The deploy job runs automatically when:
- Code is pushed to the `deployment` branch
- All previous jobs (test, train-model, evaluate-model) pass

```bash
# Trigger deployment
git checkout deployment
git merge main  # or your feature branch
git push origin deployment
```

## Testing

### API Testing

```bash
# Test API directly
curl -X POST "https://your-api-url/predict" \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

# Use test script
API_URL="https://your-api-url" python scripts/test_api.py
```

### Load Testing

For production readiness, consider load testing:

```bash
# Using curl (basic)
for i in {1..100}; do
  curl -s -X POST "https://your-api-url/predict" \
    -H "Content-Type: application/json" \
    -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}' &
done
wait
```

## Monitoring

### Render Dashboard
- View service logs
- Monitor resource usage
- Check deployment status

### Health Checks
- **API Health**: `GET /health`
- **Model Status**: `GET /model-info`

### Logs
```bash
# Local
docker-compose -f docker-compose.local.yml logs -f

# Render CLI
render logs --service iris-api
render logs --service iris-ui
```

## Troubleshooting

### Common Issues

1. **Model not loading**:
   - Ensure `models/model.joblib` exists
   - Check model file permissions in Docker

2. **Docker build fails**:
   - Verify Dockerfile syntax
   - Check if all required files are present

3. **Services can't communicate**:
   - Verify API_URL environment variable
   - Check Docker network configuration

4. **Render deployment fails**:
   - Check render.yaml syntax
   - Verify repository is connected
   - Check service logs in Render dashboard

### Debug Commands

```bash
# Check Docker images
docker images | grep iris

# Inspect container
docker inspect <container_id>

# Execute into container
docker exec -it <container_id> /bin/bash

# Check API locally
curl http://localhost:8000/health
```

## Dependency Management

The project uses `pyproject.toml` for dependency management with the following groups:

- **Core dependencies**: Required for the ML pipeline (pandas, scikit-learn, joblib, etc.)
- **dev**: Development dependencies (testing, linting, formatting)
- **api**: FastAPI backend dependencies (fastapi, uvicorn, pydantic)
- **ui**: Streamlit frontend dependencies (streamlit, plotly, requests)

Install dependencies for different components:
```bash
# Install core dependencies + dev tools
uv sync --group dev

# Install core dependencies + API dependencies
uv sync --group api

# Install core dependencies + UI dependencies
uv sync --group ui

# Install all dependencies
uv sync --group dev --group api --group ui
```

## Environment Variables

### Local Development
- `API_URL`: URL of the API service (default: `http://localhost:8000`)

### Production (Render)
- `API_URL`: Automatically set by Render to connect UI to API
- `PORT`: Automatically set by Render for service port

## Security Considerations

1. **Model Security**: Model file should be read-only in production
2. **API Rate Limiting**: Consider implementing rate limiting for production
3. **Input Validation**: API validates input schema and ranges
4. **HTTPS**: Render provides HTTPS by default
5. **Environment Variables**: Sensitive data should use Render's encrypted environment variables

## Scaling

### Render Scaling Options
- **Horizontal Scaling**: Increase instance count
- **Vertical Scaling**: Upgrade to paid plans for more resources
- **Auto-scaling**: Available on paid plans

### Performance Optimization
- **Model Loading**: Model is loaded once at startup
- **Caching**: Consider adding response caching for frequent requests
- **Database**: For production, consider storing predictions in a database

## Cost Optimization

### Render Free Tier
- 750 hours per month per service
- Services sleep after inactivity
- Suitable for development and demos

### Paid Plans
- Always-on services
- More resources and scaling options
- Better for production workloads
