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

"""Compare the benchmark methods with keywords chosen by ChatGPT, collected by hand.

ChatGPT's web interface may not be automated, so the keywords are collected manually:

1. `chatgpt.py prompts`: writes one prompt file per batch of documents to
   CHATGPT_DIR/prompts/batch_NN.txt and the document keys to CHATGPT_DIR/manifest.json.
2. Paste each prompt into ChatGPT and save its answer (the JSON object, surrounding text and
   ``` fences are tolerated) as CHATGPT_DIR/answers/batch_NN.txt.
3. `chatgpt.py import`: reads the answers into CHATGPT_DIR/chatgpt_keywords.json and lists
   documents that are missing or have no keywords.

report.py then scores ChatGPT like the other methods and renders the HTML report.
"""
import argparse
import json
import re
import sys

from settings import BENCHMARK_DIR

CHATGPT_DIR = BENCHMARK_DIR / "chatgpt"
BATCH_SIZE = 10
TOP_N = 10

PROMPT_HEAD = """Du bist ein Fachreferent, der wissenschaftliche Texte für einen Bibliothekskatalog \
inhaltlich erschließt.

Bestimme für jedes der folgenden {n} Dokumente die {top_n} relevantesten Schlagwörter, sortiert \
nach ihrer inhaltlichen Bedeutung für das Dokument (wichtigstes zuerst).

Regeln:
- Schlagwörter in der Sprache des Dokuments (bei gemischtsprachigen Dokumenten: Sprache des Titels).
- Fachbegriffe in ihrer üblichen Grundform (Nominativ Singular, außer der Begriff ist im Plural üblich).
- Schlagwörter müssen nicht wörtlich im Text vorkommen, wenn sie das Thema treffend benennen (z. B. Oberbegriffe, Synonyme).
- Synonyme zusammenführen statt mehrere Plätze zu belegen.
- Keine Erklärungen, keine Relevanzangaben.

Antworte ausschließlich mit einem JSON-Objekt, ohne weiteren Text, in genau diesem Format:
{{"{first}": ["Schlagwort 1", "Schlagwort 2", ...], "{second}": [...], ...}}
Verwende als Schlüssel die Dokument-IDs in eckigen Klammern. Jedes Dokument muss enthalten sein.
"""


def _load_documents():
    docs = []
    for path in sorted((BENCHMARK_DIR / "sets").glob("*.jsonl")):
        docs += [json.loads(line) for line in open(path, encoding="utf-8")]
    return docs


def write_prompts():
    docs = _load_documents()
    (CHATGPT_DIR / "prompts").mkdir(parents=True, exist_ok=True)
    (CHATGPT_DIR / "answers").mkdir(parents=True, exist_ok=True)
    manifest = {}
    batches = [docs[i:i + BATCH_SIZE] for i in range(0, len(docs), BATCH_SIZE)]
    for b, batch in enumerate(batches, 1):
        keys = [f"D{b:02d}-{i:02d}" for i in range(1, len(batch) + 1)]
        parts = [PROMPT_HEAD.format(n=len(batch), top_n=TOP_N, first=keys[0], second=keys[min(1, len(keys) - 1)])]
        for key, doc in zip(keys, batch):
            manifest[key] = {"id": doc["id"], "set": doc["set"], "batch": b}
            parts.append(f"[{key}]\n{doc['text'].strip()}\n")
        (CHATGPT_DIR / "prompts" / f"batch_{b:02d}.txt").write_text("\n".join(parts), encoding="utf-8")
    json.dump(manifest, open(CHATGPT_DIR / "manifest.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(docs)} documents in {len(batches)} prompts: {CHATGPT_DIR / 'prompts'}")
    print(f"save the answers as {CHATGPT_DIR / 'answers'}/batch_NN.txt")


def _parse_answer(text: str) -> dict:
    """The JSON object in a ChatGPT answer, tolerating ``` fences and text around it."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    return json.loads(text[start:end + 1])


def import_answers() -> int:
    manifest = json.load(open(CHATGPT_DIR / "manifest.json", encoding="utf-8"))
    keywords, problems = {}, []
    for path in sorted((CHATGPT_DIR / "answers").glob("batch_*.*")):
        try:
            answer = _parse_answer(path.read_text(encoding="utf-8"))
        except ValueError as e:
            problems.append(f"{path.name}: {e}")
            continue
        for key, kws in answer.items():
            key = re.sub(r"[\[\]\s]", "", key)
            if key not in manifest:
                problems.append(f"{path.name}: unknown key {key}")
            elif not isinstance(kws, list):
                problems.append(f"{path.name}: {key} is not a list")
            else:
                keywords[manifest[key]["id"]] = [str(k).strip() for k in kws if str(k).strip()][:TOP_N]
    missing = [k for k, v in manifest.items() if v["id"] not in keywords]
    batches = sorted({manifest[k]["batch"] for k in missing})
    json.dump(keywords, open(CHATGPT_DIR / "chatgpt_keywords.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(keywords)} of {len(manifest)} documents have ChatGPT keywords")
    if missing:
        print(f"missing: {len(missing)} documents, batches {batches}")
    for p in problems:
        print("problem:", p)
    return 1 if problems else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["prompts", "import"])
    args = parser.parse_args()
    if args.command == "prompts":
        write_prompts()
    else:
        sys.exit(import_answers())


if __name__ == "__main__":
    main()
