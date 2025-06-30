# syntax = docker/dockerfile:1.2
FROM python:3.11-slim

# Install build tools
RUN apt-get update && apt-get install -y \
    git build-essential wget curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .


RUN --mount=type=cache,target=/export/bamboo/cache/pip \
        pip install -vvv -r requirements.txt

COPY . .

CMD ["uvicorn", "iri_api:app", "--host", "0.0.0.0", "--port", "8000"]
