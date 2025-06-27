# radar-keyword-search

## Build

docker build -t radar/radar-keywords-search .

## Run server

docker run -p 8000:8000 radar/radar-keywords-search

## Search keyword in document
curl -X POST http://localhost:8000/extract-iris -H "Content-Type: application/json" -d '{"document": "hydrogen"}'