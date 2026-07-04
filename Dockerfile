FROM python:3.11-slim

# Set non-interactive timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    libsndfile1 \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY reasoning_engine/ reasoning_engine/
COPY main.py .

# Set environment variables for determinism
ENV PYTHONHASHSEED=42
ENV PYTHONPATH=/app

# Define the entrypoint
CMD ["python", "main.py"]
