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

# Upgrade pip
RUN pip install --upgrade pip

# ---------------------------------------------------------------------------
# OPTIMIZATION: Install CPU-only PyTorch first.
# This prevents downloading the huge 2GB+ GPU version.
# We do this BEFORE 'pip install -r requirements.txt' to use Docker caching.
# ---------------------------------------------------------------------------
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies
# (pip will see torch is already installed and skip it)
RUN pip install -r requirements.txt

# Copy all project files
COPY . .

# Expose port (Render sets PORT env var, defaulting to 10000 here for documentation)
EXPOSE 10000

# Start the app using the PORT environment variable
CMD sh -c "gunicorn -b 0.0.0.0:${PORT:-10000} app:app"
