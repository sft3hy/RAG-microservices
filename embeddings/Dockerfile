# Use slim Python image for smaller size
FROM python:3.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8001

# Create app directory
WORKDIR /app

# Install system dependencies (for numpy, sentence-transformers, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE $PORT

# Healthcheck (optional, good for Kubernetes/Helm)
HEALTHCHECK CMD curl --fail http://127.0.0.1:${PORT}/docs || exit 1

# Run as non-root user
RUN useradd -m appuser
USER appuser

# Start FastAPI with uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
