FROM python:3.11-slim

# Install build tools
RUN apt-get update && apt-get install -y \
    git build-essential wget curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install Python dependencies
ENV PIP_CACHE_DIR=/export/bamboo/cache/pip
ENV PIP_BUILD=/export/bamboo/cache/pip/build
ENV TMPDIR=/export/bamboo/cache/pip/tmp

RUN pip install -vvv \
       --cache-dir=$PIP_CACHE_DIR \
       --build=$PIP_BUILD         \
       --log $PIP_CACHE_DIR/pip.log \
       -r requirements.txt

COPY . .

CMD ["uvicorn", "iri_api:app", "--host", "0.0.0.0", "--port", "8000"]
