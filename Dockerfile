FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for sentence-transformers
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY app.py .
COPY brain.py .
COPY memory.py .
COPY brain_metadata.json .
COPY templates/ ./templates/
COPY pdfs/ ./pdfs/

# Create directories for conversations
RUN mkdir -p conversations

ENV PORT=5000
EXPOSE 5000

CMD ["python", "app.py"]