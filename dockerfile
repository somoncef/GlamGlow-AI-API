# Use official Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Create directory for saved models
RUN mkdir -p saved_models

# Expose the port Hugging Face Spaces uses
EXPOSE 8080

# Start Gunicorn server
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "app:app"]