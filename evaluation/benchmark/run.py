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

Runs the extraction in-process (no service, no TIB/GND lookup) with one BERT model and
reports precision, recall and F1 of the top 10 keywords against each document's gold
keywords, per set and method. Needs the service's full requirements.

Matching is lenient about inflection: a keyword matches a gold keyword if both have the
same number of words and each word pair is within the service's allowed edit distance
(a quarter of the word's length), after lowercasing and dropping GND qualifiers such as
"Wien <Stadt>" or "Bank (Möbel)".
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR))
from bert_models import BERT_MODEL_REGISTRY  # noqa: E402
from keyword_extraction import KeywordExtractor  # noqa: E402
from pos_ngrams import PosNgramExtractor, detect_language  # noqa: E402
from stop_words import STOP_WORDS  # noqa: E402

from settings import BENCHMARK_DIR, RESULTS_DIR  # noqa: E402
from scoring import gold_in_text, score  # noqa: E402

TOP_N = 10
BERT_MAX_TOKENS = 512
EMBED_BATCH_SIZE = 32

# The service's two candidate options.
METHODS = ["words", "pos-ngrams"]


class CachedEmbedding:
    """Mean-pooled embeddings like the service's _BertEmbedding, cached per text."""

    def __init__(self, model_key: str):
        entry = BERT_MODEL_REGISTRY[model_key]
        self._prefix = entry.get("prefix", "")
        self._tokenizer = AutoTokenizer.from_pretrained(entry["hf_model"])
        self._model = AutoModel.from_pretrained(entry["hf_model"]).eval()
        self._cache = {}

    def clear(self):
        self._cache.clear()

    def __call__(self, docs):
        docs = [str(d) for d in docs]
        missing = [d for d in dict.fromkeys(docs) if d not in self._cache]
        for i in range(0, len(missing), EMBED_BATCH_SIZE):
            batch = missing[i:i + EMBED_BATCH_SIZE]
            enc = self._tokenizer([self._prefix + d for d in batch], padding=True, truncation=True,
                                  max_length=BERT_MAX_TOKENS, return_tensors="pt")
            with torch.no_grad():
                out = self._model(**enc)[0]
            mask = enc["attention_mask"].unsqueeze(-1).float()
            vecs = ((out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)).numpy()
            self._cache.update(zip(batch, vecs))
        return np.stack([self._cache[d] for d in docs])


# ---- methods ----

def make_methods(embed: CachedEmbedding):
    keybert = KeywordExtractor(embed)
    pos_ngrams = PosNgramExtractor()

    return {
        "words": lambda doc: [k for k, _ in keybert.extract_keywords(doc, top_n=TOP_N, stop_words=STOP_WORDS)],
        "pos-ngrams": lambda doc: pos_ngrams(doc, embed, TOP_N),
    }


def load_sets(names):
    sets = {}
    for path in sorted((BENCHMARK_DIR / "sets").glob("*.jsonl")):
        if names and path.stem not in names:
            continue
        sets[path.stem] = [json.loads(line) for line in open(path, encoding="utf-8")]
    return sets


def print_table(summary: dict, methods: list):
    print(f"\n{'set':<16}{'n':>4}{'gold∈text':>10}  " + "".join(f"{m:>17}" for m in methods))
    print(f"{'':<30}  " + "".join(f"{'P / R / F1':>17}" for _ in methods))
    for name, s in summary.items():
        cells = "".join(f"{s['methods'][m]['p']:>7.2f}/{s['methods'][m]['r']:.2f}/{s['methods'][m]['f1']:.2f}" for m in methods)
        print(f"{name:<16}{s['n']:>4}{s['gold_in_text']:>10.2f}  {cells}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="multilingual-e5", help="key of bert_models.BERT_MODEL_REGISTRY")
    parser.add_argument("--sets", nargs="*", help="set names (default: all in BENCHMARK_DIR/sets)")
    parser.add_argument("--methods", nargs="*", default=METHODS, choices=METHODS)
    args = parser.parse_args()

    embed = CachedEmbedding(args.model)
    methods = make_methods(embed)
    sets = load_sets(args.sets)
    results = {"meta": {"model": args.model, "top_n": TOP_N, "methods": args.methods,
                        "created": datetime.now(timezone.utc).isoformat(timespec="seconds")},
               "documents": [], "summary": {}}

    for name, docs in sets.items():
        started = time.monotonic()
        per_method = {m: [] for m in args.methods}
        in_text = []
        for doc in docs:
            # Metadata languages are unreliable (German abstracts labelled "en", records with
            # abstracts in both languages): pos-ngrams detects the language per paragraph, as
            # in the service. `lang` is only reported.
            lang = detect_language(doc["text"]) or "en"
            entry = {"id": doc["id"], "set": name, "lang": lang, "gold": doc["gold"], "keywords": {}, "scores": {}}
            in_text.append(gold_in_text(doc["text"], doc["gold"]))
            for m in args.methods:
                kws = methods[m](doc["text"])
                entry["keywords"][m] = kws
                entry["scores"][m] = score(kws, doc["gold"])
                per_method[m].append(entry["scores"][m])
            embed.clear()
            results["documents"].append(entry)
        results["summary"][name] = {
            "n": len(docs),
            "gold_in_text": float(np.mean(in_text)) if in_text else 0.0,
            "methods": {m: {k: float(np.mean([s[k] for s in v])) if v else 0.0 for k in ("p", "r", "f1")}
                        for m, v in per_method.items()},
        }
        print(f"{name}: {len(docs)} documents in {time.monotonic() - started:.0f}s", flush=True)

    print_table(results["summary"], args.methods)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"benchmark_{args.model}_{datetime.now():%Y%m%d_%H%M%S}.json"
    json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
