# Dockerfile for Railway
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY brain.py .
COPY memory.py .
COPY brain_metadata.json .
COPY widget.js .
COPY templates/ ./templates/

# Create required directories
RUN mkdir -p conversations pdfs

# Set environment variables
ENV PORT=5000
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5000

# Start the application
CMD ["python", "app.py"]