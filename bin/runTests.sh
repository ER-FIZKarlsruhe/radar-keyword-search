#!/usr/bin/env bash

# Copyright 2025-2026 FIZ Karlsruhe - Leibniz-Institut für Informationsinfrastruktur GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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