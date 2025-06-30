FROM python:3.11-slim

# Install build tools
RUN apt-get update && apt-get install -y \
    git build-essential wget curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# 1. Tell pip where its cache should live
ENV PIP_CACHE_DIR=/export/bamboo/cache/pip

# 2. Force ALL temp/build files onto your NFS
ENV TMPDIR=/export/bamboo/cache/pip/tmp
ENV TEMP=$TMPDIR
ENV TMP=$TMPDIR

RUN pip install -vvv \
       --cache-dir=$PIP_CACHE_DIR \
       --log $PIP_CACHE_DIR/pip.log \
       -r requirements.txt

COPY . .

CMD ["uvicorn", "iri_api:app", "--host", "0.0.0.0", "--port", "8000"]
