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

"""Build the benchmark sets: one JSONL file per set, one document per line.

Each line: {"id", "set", "lang", "text", "gold": [reference keywords]}.

Sources (downloaded into BENCHMARK_DIR/raw, see README.md):

- LLMs4Subjects (SemEval-2025 Task 5, CC BY 4.0): TIBKAT records with GND subjects assigned
  by subject librarians. Split by language and DDC class into chemistry, medicine and
  humanities sets. The gold subjects are *assigned*, so many don't occur in the text.
- KPBiomed (CC BY-NC 4.0, internal evaluation only): PubMed abstracts with author keywords.
- RADAR4Chem / RADAR4Culture: the evaluation harvest (../harvest.py), author keywords.
- custom/*.json: hand-made references, e.g. a document with an LLM-made keyword list.

Standard library only. Samples are drawn with a fixed seed, so a rerun gives the same sets.
"""
import argparse
import glob
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import HARVEST_DIR  # noqa: E402
from records import document_text, load_datasets  # noqa: E402

from settings import BENCHMARK_DIR  # noqa: E402

RAW_DIR = BENCHMARK_DIR / "raw"
L4S_GLOB = "llms4subjects/shared-task-datasets/TIBKAT/all-subjects/data/test/gold-standard-testset/**/*.jsonld"

# DDC hundreds/tens (first two digits of the ddc-dbn notation) per LLMs4Subjects set.
HUMANITIES_DDC = (
    {f"1{d}" for d in range(10)}                        # philosophy, psychology
    | {f"2{d}" for d in range(10)}                      # religion
    | {f"4{d}" for d in range(10)}                      # language
    | {f"7{d}" for d in range(10)} - {"71"}             # arts (without landscape planning)
    | {f"8{d}" for d in range(10)}                      # literature
    | {f"9{d}" for d in range(10)}                      # history, geography
)
L4S_SETS = {
    "chem": {"54"},
    "med": {"61"},
    "hum": HUMANITIES_DDC,
}

# Abstracts shorter than this are mostly publisher blurbs or tables of contents.
MIN_ABSTRACT_CHARS = 300
MIN_GOLD = 2


def _l4s_records():
    for path in sorted(glob.glob(str(RAW_DIR / L4S_GLOB), recursive=True)):
        graph = json.load(open(path, encoding="utf-8"))["@graph"]
        names = {n["@id"]: n.get("sameAs") for n in graph if "sameAs" in n}
        record = next((n for n in graph if "title" in n), None)
        if record is None:
            continue
        subject_ids = re.findall(r"'@id': '([^']+)'", str(record.get("dcterms:subject", "")))
        gold = [names[i] for i in subject_ids if names.get(i)]
        abstract = record.get("abstract") or ""
        if isinstance(abstract, list):
            abstract = "\n\n".join(abstract)
        title = record["title"] if isinstance(record["title"], str) else " ".join(record["title"])
        yield {
            "id": "l4s:" + Path(path).stem,
            "lang": str(record.get("language", ""))[-2:],
            "ddc": set(re.findall(r"ddc-dbn\)(\d\d)", str(record.get("subject", "")))),
            "title": title,
            "abstract": abstract,
            "gold": gold,
        }


def build_l4s(per_set: int, rng: random.Random) -> dict:
    pools = {f"l4s-{name}-{lang}": [] for name in L4S_SETS for lang in ("de", "en")}
    for rec in _l4s_records():
        if rec["lang"] not in ("de", "en") or len(rec["abstract"]) < MIN_ABSTRACT_CHARS or len(rec["gold"]) < MIN_GOLD:
            continue
        for name, ddc in L4S_SETS.items():
            if rec["ddc"] & ddc:
                pools[f"l4s-{name}-{rec['lang']}"].append(rec)
                break  # a record counts for one set only, the first that fits
    sets = {}
    for set_name, pool in pools.items():
        sample = rng.sample(pool, min(per_set, len(pool)))
        sets[set_name] = [
            {"id": r["id"], "set": set_name, "lang": r["lang"],
             "text": f"{r['title']}\n\n{r['abstract']}", "gold": r["gold"]}
            for r in sample
        ]
    return sets


def build_kpbiomed(per_set: int, rng: random.Random) -> dict:
    path = RAW_DIR / "kpbiomed_test.jsonl"
    pool = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if len(d.get("abstract") or "") >= MIN_ABSTRACT_CHARS and len(d.get("keyphrases") or []) >= MIN_GOLD:
                pool.append(d)
    sample = rng.sample(pool, min(per_set, len(pool)))
    return {"kpbiomed-en": [
        {"id": f"kpbiomed:{d['id']}", "set": "kpbiomed-en", "lang": "en",
         "text": f"{d['title']}\n\n{d['abstract']}", "gold": d["keyphrases"]}
        for d in sample
    ]}


def build_radar(per_set: int, rng: random.Random) -> dict:
    sets = {}
    for name, oai_set in (("radar-chem", "radar4chem"), ("radar-culture", "radar4culture")):
        if not (HARVEST_DIR / oai_set).is_dir():
            print(f"skipping {name}: no harvest in {HARVEST_DIR / oai_set}")
            continue
        pool = [d for d in load_datasets(oai_set)
                if len(d["abstract"]) >= MIN_ABSTRACT_CHARS and len(d["author_keywords"]) >= MIN_GOLD]
        sample = rng.sample(pool, min(per_set, len(pool)))
        # RADAR records don't state their language; run.py detects it per document.
        sets[name] = [{"id": d["identifier"], "set": name, "lang": None,
                       "text": document_text(d), "gold": d["author_keywords"]} for d in sample]
    return sets


def build_custom() -> dict:
    """custom/<set>.json: a list of {"id", "lang", "text" or "text_file", "gold"}."""
    sets = {}
    for path in sorted((BENCHMARK_DIR / "custom").glob("*.json")):
        docs = []
        for d in json.load(open(path, encoding="utf-8")):
            text = d.get("text") or Path(d["text_file"]).read_text(encoding="utf-8")
            docs.append({"id": d["id"], "set": path.stem, "lang": d.get("lang"), "text": text, "gold": d["gold"]})
        sets[path.stem] = docs
    return sets


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--per-set", type=int, default=30, help="documents per set (default 30)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sets = {}
    for builder in (build_l4s, build_kpbiomed, build_radar):
        sets.update(builder(args.per_set, random.Random(args.seed)))
    sets.update(build_custom())

    out_dir = BENCHMARK_DIR / "sets"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, docs in sets.items():
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"{name}: {len(docs)} documents")


if __name__ == "__main__":
    main()
