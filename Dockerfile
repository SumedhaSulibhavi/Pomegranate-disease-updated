# Use Python 3.11 (PyTorch-compatible)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Upgrade pip and install CPU-only PyTorch
# We do this BEFORE 'pip install -r requirements.txt' to use Docker caching and avoid GPU bloat.
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    sed -i '/torch/d' requirements.txt && \
    pip install -r requirements.txt

# Copy all project files
COPY . .

# Expose port (Render sets PORT env var, defaulting to 10000 here for documentation)
EXPOSE 10000

# Start the app using the PORT environment variable
ENTRYPOINT ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-10000} app:app"]
