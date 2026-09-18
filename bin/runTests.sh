#!/usr/bin/env bash
set -euo pipefail

# Run from the repo root regardless of where this script is invoked from.
cd "$(dirname "$0")/.."

python3 -m venv radar-keywords-test-env
source radar-keywords-test-env/bin/activate

echo "HTTP_PROXY=${HTTP_PROXY:-<not set>}"
echo "HTTPS_PROXY=${HTTPS_PROXY:-<not set>}"
echo "NO_PROXY=${NO_PROXY:-<not set>}"

python -m pip --version
python -m pip install -vvv -r requirements-test.txt

python -m pytest -q --cov=iri_api --cov-report=term-missing