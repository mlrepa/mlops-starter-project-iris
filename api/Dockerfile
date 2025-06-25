FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (curl needed for health checks)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and uv.lock from parent directory
COPY ../pyproject.toml ../uv.lock ./

# Install dependencies using uv (creates virtual environment)
RUN uv sync --group api

# Set up environment to use uv's virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Copy the rest of the application code into the container at /app
COPY api/main.py .

# Create models directory (model file will be mounted as volume)
RUN mkdir -p models
# Create a non-root user
RUN adduser --disabled-password --gecos '' appuser && chown -R appuser /app
USER appuser

# Expose port 8000 to the outside world
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Command to run the app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
