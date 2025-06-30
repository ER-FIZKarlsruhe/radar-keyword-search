FROM python:3.11-slim

# Install build tools
RUN apt-get update && apt-get install -y \
    git build-essential wget curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install Python dependencies
ARG PIP_CACHE_DIR=/root/.cache/pip
ENV PIP_CACHE_DIR=${PIP_CACHE_DIR}
RUN pip install --upgrade pip \
    && pip install --cache-dir=${PIP_CACHE_DIR} -r requirements.txt

COPY . .

CMD ["uvicorn", "iri_api:app", "--host", "0.0.0.0", "--port", "8000"]
