FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py brain.py memory.py brain_metadata.json ./
COPY templates/ ./templates/
COPY static/ ./static/

RUN mkdir -p conversations pdfs

ENV PORT=5000
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["python", "app.py"]