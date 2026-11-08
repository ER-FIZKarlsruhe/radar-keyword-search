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

"""Run the keyword service over harvested datasets and store the results as JSON.

For each domain, every configured model is run in every mode (exact / fuzzy)
on the first --limit datasets. Output: RESULTS_DIR/<domain>_<timestamp>.json.

    python evaluation/run_extraction.py                    # all domains, first 10 datasets
    python evaluation/run_extraction.py chem --limit 50
    python evaluation/run_extraction.py culture --require-abstract
    python evaluation/run_extraction.py culture --models gbert multilingual-e5
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from config import DOMAINS, EVAL_DIR, MAX_KEYWORDS, MODES, RESULTS_DIR, SERVICE_URL
from records import document_text, load_datasets

SCHEMA_VERSION = 1
# The first request for a model makes the service download/load it.
REQUEST_TIMEOUT = 600


def _request(method: str, path: str, payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        f"{SERVICE_URL}{path}", data=data, method=method,
        headers={"Content-Type": "application/json", "User-Agent": "radar-keyword-eval/1.0"},
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def _git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=EVAL_DIR, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def extract(payload: dict) -> dict:
    """Call /extract-iris once; never raises, failures are recorded in the result."""
    started = time.monotonic()
    result = {"status": "ok", "error": None}
    for attempt in range(2):
        try:
            response = _request("POST", "/extract-iris", payload)
            # The service returns {keyword: match-or-null} in KeyBERT rank order.
            keywords = [
                {"rank": i + 1, "keyword": kw, "match": match} for i, (kw, match) in enumerate(response.items())
            ]
            result.update(status="ok", error=None, keywords=keywords)
            break
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            result.update(status="error", error=f"HTTP {e.code}: {detail}", keywords=[])
            if e.code < 500:
                break
        except OSError as e:
            result.update(status="error", error=str(e), keywords=[])
    result["elapsed_s"] = round(time.monotonic() - started, 2)
    return result


def run_domain(domain: str, limit: int, require_abstract: bool, models) -> Path:
    config = DOMAINS[domain]
    models = models or config["models"]

    datasets = [d for d in load_datasets(config["oai_set"]) if d["abstract"] or not require_abstract][:limit]
    runs = []
    for model in models:
        for mode, mode_params in MODES.items():
            request = {
                "model": model,
                "max_keywords": MAX_KEYWORDS,
                "terminology": config["terminology"],
                "ontology_collection": config["ontology_collection"],
                **mode_params,
            }
            runs.append({"id": f"{model}/{mode}", "model": model, "mode": mode, "request": request})

    output = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "domain": domain,
            "oai_set": config["oai_set"],
            "started": datetime.now(timezone.utc).isoformat(),
            "service_url": SERVICE_URL,
            "service_health": _request("GET", "/"),
            "git_commit": _git_commit(),
            "limit": limit,
            "require_abstract": require_abstract,
            "document": "title + abstract",
        },
        "runs": runs,
        "datasets": [],
    }

    total = len(datasets) * len(runs)
    done = 0
    for dataset in datasets:
        document = document_text(dataset)
        entry = {**dataset, "document": document, "results": {}}
        for run in runs:
            done += 1
            result = extract({"document": document, **run["request"]})
            entry["results"][run["id"]] = result
            linked = sum(1 for k in result["keywords"] if k["match"] and k["match"].get("matched"))
            status = result["error"] or f"{len(result['keywords'])} keywords, {linked} linked"
            print(f"  [{done}/{total}] {dataset['identifier']} {run['id']}: {status} ({result['elapsed_s']}s)")
        output["datasets"].append(entry)

    output["meta"]["finished"] = datetime.now(timezone.utc).isoformat()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RESULTS_DIR / f"{domain}_{stamp}.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  {domain}: results written to {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("domains", nargs="*", metavar="domain", help=f"one of {', '.join(DOMAINS)} (default: all)")
    parser.add_argument("--limit", type=int, default=10, help="number of datasets per domain (default: 10)")
    parser.add_argument("--require-abstract", action="store_true", help="skip datasets without an abstract")
    parser.add_argument("--models", nargs="+", help="override the domain's configured models")
    args = parser.parse_args()
    unknown = set(args.domains) - set(DOMAINS)
    if unknown:
        parser.error(f"unknown domain(s): {', '.join(sorted(unknown))}")
    for domain in args.domains or DOMAINS:
        run_domain(domain, args.limit, args.require_abstract, args.models)


if __name__ == "__main__":
    sys.exit(main())
