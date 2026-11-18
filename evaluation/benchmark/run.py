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

"""Compare keyword extraction methods on the benchmark sets (see prepare.py).

Sends each document to a running keyword service (POST /extract-iris with
terminology "none", so no TIB/GND lookup) once per `candidates` option and reports
precision, recall and F1 of the top 10 keywords against each document's gold
keywords, per set and method. Standard library only.

Matching is lenient about inflection: a keyword matches a gold keyword if both have the
same number of words and each word pair is within the service's allowed edit distance
(a quarter of the word's length), after lowercasing and dropping GND qualifiers such as
"Wien <Stadt>" or "Bank (Möbel)".

    python run.py                                   # service at http://localhost:8000
    python run.py --service-url http://host:8000 --model gbert --sets l4s-med-de
"""
import argparse
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from settings import BENCHMARK_DIR, RESULTS_DIR
from scoring import gold_in_text, score

TOP_N = 10

# The service's two candidate options.
METHODS = ["words", "pos-ngrams"]

DEFAULT_SERVICE_URL = os.getenv("KEYWORD_BENCHMARK_SERVICE_URL", "http://localhost:8000")

# Seconds for connecting and for each read (urllib has one timeout for both). Generous
# because the first request for a model makes the service download and load it.
REQUEST_TIMEOUT = 300


class ServiceError(Exception):
    pass


def extract(service_url: str, model: str, method: str, text: str) -> list:
    """Keywords for one document from the service, best first."""
    payload = {"document": text, "model": model, "candidates": method,
               "terminology": "none", "max_keywords": TOP_N}
    request = urllib.request.Request(
        f"{service_url}/extract-iris", data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", "User-Agent": "radar-keyword-benchmark/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            # {keyword: match-or-null} in rank order.
            return list(json.loads(response.read().decode("utf-8")))
    except urllib.error.HTTPError as e:
        raise ServiceError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise ServiceError(str(e)) from e


def load_sets(names):
    sets = {}
    for path in sorted((BENCHMARK_DIR / "sets").glob("*.jsonl")):
        if names and path.stem not in names:
            continue
        sets[path.stem] = [json.loads(line) for line in open(path, encoding="utf-8")]
    return sets


def _mean(values) -> float:
    return float(statistics.mean(values)) if values else 0.0


def print_table(summary: dict, methods: list):
    print(f"\n{'set':<16}{'n':>4}{'gold∈text':>10}  " + "".join(f"{m:>17}" for m in methods))
    print(f"{'':<30}  " + "".join(f"{'P / R / F1':>17}" for _ in methods))
    for name, s in summary.items():
        cells = "".join(f"{s['methods'][m]['p']:>7.2f}/{s['methods'][m]['r']:.2f}/{s['methods'][m]['f1']:.2f}" for m in methods)
        print(f"{name:<16}{s['n']:>4}{s['gold_in_text']:>10.2f}  {cells}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--service-url", default=DEFAULT_SERVICE_URL,
                        help="base URL of the keyword service (default: $KEYWORD_BENCHMARK_SERVICE_URL or %(default)s)")
    parser.add_argument("--model", default="multilingual-e5", help="model name as listed by GET /models")
    parser.add_argument("--sets", nargs="*", help="set names (default: all in BENCHMARK_DIR/sets)")
    parser.add_argument("--methods", nargs="*", default=METHODS, choices=METHODS)
    args = parser.parse_args()
    service_url = args.service_url.rstrip("/")

    sets = load_sets(args.sets)
    results = {"meta": {"model": args.model, "top_n": TOP_N, "methods": args.methods, "service_url": service_url,
                        "created": datetime.now(timezone.utc).isoformat(timespec="seconds")},
               "documents": [], "summary": {}}
    failures = 0

    for name, docs in sets.items():
        started = time.monotonic()
        per_method = {m: [] for m in args.methods}
        in_text = []
        for doc in docs:
            # `lang` is the set's metadata language and only reported: the service detects
            # the language per paragraph itself.
            entry = {"id": doc["id"], "set": name, "lang": doc.get("lang"), "gold": doc["gold"],
                     "keywords": {}, "scores": {}, "errors": {}}
            in_text.append(gold_in_text(doc["text"], doc["gold"]))
            for m in args.methods:
                try:
                    kws = extract(service_url, args.model, m, doc["text"])
                except ServiceError as e:
                    # A failed document counts as no keywords, so it lowers the scores visibly.
                    failures += 1
                    entry["errors"][m] = str(e)
                    print(f"  {doc['id']} [{m}]: {e}", flush=True)
                    kws = []
                entry["keywords"][m] = kws
                entry["scores"][m] = score(kws, doc["gold"])
                per_method[m].append(entry["scores"][m])
            results["documents"].append(entry)
        results["summary"][name] = {
            "n": len(docs),
            "gold_in_text": _mean(in_text),
            "methods": {m: {k: _mean([s[k] for s in v]) for k in ("p", "r", "f1")} for m, v in per_method.items()},
        }
        print(f"{name}: {len(docs)} documents in {time.monotonic() - started:.0f}s", flush=True)

    print_table(results["summary"], args.methods)
    if failures:
        print(f"\n{failures} failed requests (scored as empty keyword lists, see 'errors')")
    results["meta"]["failed_requests"] = failures
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"benchmark_{args.model}_{datetime.now():%Y%m%d_%H%M%S}.json"
    json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
