# radar-keyword-search

## Build

docker login -u radar-docker docker.dev.fiz-karlsruhe.de

docker build -t radar/radar-keywords-search --tag=docker.dev.fiz-karlsruhe.de/radar-keyword-search:0.1 .

docker push docker.dev.fiz-karlsruhe.de/radar-keyword-search:0.1

## Run server

docker run -p 8000:8000 docker.dev.fiz-karlsruhe.de/radar-keyword-search:0.1

## Search keyword in document
curl -X POST http://localhost:8000/extract-iris -H "Content-Type: application/json" -d '{"document": "Introducing Photochemical Action Plots as a Tool for Unlocking On-Off Switchable Behavior in a Polymeric Eosin Y Photocatalyst"}'