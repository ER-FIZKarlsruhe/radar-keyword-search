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

"""Paths of the keyword benchmark."""
import os
from pathlib import Path

BENCHMARK_EVAL_DIR = Path(__file__).resolve().parent

# Downloaded data and the built sets live in evaluation/data/, which is git-ignored: the
# repository is public and some sources may not be redistributed (KPBiomed is CC BY-NC).
BENCHMARK_DIR = Path(os.getenv(
    "KEYWORD_BENCHMARK_DIR",
    BENCHMARK_EVAL_DIR.parent / "data" / "benchmarks",
))
RESULTS_DIR = Path(os.getenv("KEYWORD_BENCHMARK_RESULTS_DIR", BENCHMARK_EVAL_DIR / "results"))
