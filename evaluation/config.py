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

"""Shared settings for the keyword-extraction evaluation scripts.

Every path and URL can be overridden with an environment variable, so the
scripts run unchanged on another machine.
"""

import os
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent

OAI_BASE_URL = os.getenv("RADAR_OAI_URL", "https://www.radar-service.eu/oai/OAIHandler")
OAI_METADATA_PREFIX = "radar"

SERVICE_URL = os.getenv("KEYWORD_SERVICE_URL", "https://ai.radar-service.eu/keywords").rstrip("/")

# Harvested OAI-PMH pages live outside the project, next to the repo checkout.
HARVEST_DIR = Path(os.getenv("KEYWORD_EVAL_HARVEST_DIR", EVAL_DIR.parents[1] / "radar-keyword-eval-data" / "harvest"))
RESULTS_DIR = Path(os.getenv("KEYWORD_EVAL_RESULTS_DIR", EVAL_DIR / "results"))
REPORTS_DIR = Path(os.getenv("KEYWORD_EVAL_REPORTS_DIR", EVAL_DIR / "reports"))

MAX_KEYWORDS = 10

# "exact" accepts only distance 0 (case-insensitive identical labels); "fuzzy"
# leaves max_distance unset, so the service applies its default threshold.
MODES = {
    "exact": {"max_distance": 0},
    "fuzzy": {},
}

# One entry per research domain. "models" are keys of bert_models.BERT_MODEL_REGISTRY;
# every model is run in every mode, so each dataset gets len(models) * len(MODES) runs.
DOMAINS = {
    "culture": {
        "oai_set": "radar4culture",
        "terminology": "both",
        "ontology_collection": "NFDI4CULTURE",
        # Culture metadata is largely German: a multilingual sentence-embedding
        # baseline next to the domain-specific historic multilingual encoder.
        "models": ["multilingual-e5", "historic-multilingual-bert"],
    },
    "chem": {
        "oai_set": "radar4chem",
        "terminology": "tib",
        "ontology_collection": "NFDI4CHEM",
        # General scientific encoder next to the chemistry-specific one.
        "models": ["scibert", "chemicalbert"],
    },
    "memory": {
        "oai_set": "radar4memory",
        "terminology": "both",
        # TIB has no NFDI4Memory collection; NFDI4CULTURE is the closest (history, heritage).
        "ontology_collection": "NFDI4CULTURE",
        # Mostly German (often with a parallel English abstract) historical texts:
        # the multilingual sentence-embedding baseline next to the historic encoder.
        "models": ["multilingual-e5", "historic-multilingual-bert"],
    },
}
