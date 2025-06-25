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
RUN uv sync --group ui

# Set up environment to use uv's virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Copy the rest of the application code into the container at /app
COPY ui/app.py .

# Expose the default Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
# The --server.address=0.0.0.0 is important for containerization
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
