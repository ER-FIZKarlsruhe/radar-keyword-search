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

"""HTML report of a benchmark run (run.py), with ChatGPT's keywords (chatgpt.py) as one more method.

Two comparisons, both with the same matching (scoring.py):

- every method, ChatGPT included, against the reference keywords of the sets;
- every method against ChatGPT's keywords, taken as the reference: how close is each
  method to what ChatGPT picks?

Looks like ../report.py and reuses its CSS and ../charts.py. Standard library only.

    python benchmark_report.py [results/benchmark_….json]   # default: newest result
"""
import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import charts  # noqa: E402
import report as eval_report  # noqa: E402

from chatgpt import CHATGPT_DIR  # noqa: E402
from scoring import matches, normalize, score  # noqa: E402
from settings import BENCHMARK_DIR, RESULTS_DIR  # noqa: E402

CHATGPT = "chatgpt"
# charts.SERIES has four colours; the benchmark compares up to nine methods.
EXTRA_CSS = """
:root { --s5:#8a5cd1; --s6:#c2417a; --s7:#2a9d8f; --s8:#b5832a; --s9:#6b7280; }
@media (prefers-color-scheme: dark) { :root { --s5:#9d74e0; --s6:#d75a90; --s7:#45b8aa; --s8:#d4a24a; --s9:#9ca3af; } }
ol.kws li.gpt { box-shadow:inset 3px 0 0 var(--s6); }
.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
td.best { font-weight:700; }
"""
COLORS = charts.SERIES + ["var(--s5)", "var(--s6)", "var(--s7)", "var(--s8)", "var(--s9)"]

_e = eval_report._e


def _mean(values: List[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


def _load(result_path: Path):
    data = json.load(open(result_path, encoding="utf-8"))
    chatgpt_path = CHATGPT_DIR / "chatgpt_keywords.json"
    gpt = json.load(open(chatgpt_path, encoding="utf-8")) if chatgpt_path.exists() else {}
    texts = {}
    for path in sorted((BENCHMARK_DIR / "sets").glob("*.jsonl")):
        for line in open(path, encoding="utf-8"):
            d = json.loads(line)
            texts[d["id"]] = d["text"]
    methods = list(data["meta"]["methods"])
    for doc in data["documents"]:
        if doc["id"] in gpt:
            doc["keywords"][CHATGPT] = gpt[doc["id"]]
            doc["scores"][CHATGPT] = score(gpt[doc["id"]], doc["gold"])
    if gpt:
        methods.append(CHATGPT)
    return data, methods, texts, gpt


def compute(data: Dict, methods: List[str]) -> Dict:
    """Per set and overall: mean P/R/F1 against the reference and, where ChatGPT answered,
    mean P/R/F1 against ChatGPT's keywords."""
    sets: Dict[str, List[Dict]] = {}
    for doc in data["documents"]:
        sets.setdefault(doc["set"], []).append(doc)
    groups = dict(sets)
    groups["all"] = data["documents"]
    out = {}
    for name, docs in groups.items():
        with_gpt = [d for d in docs if CHATGPT in d["keywords"]]
        out[name] = {
            "n": len(docs),
            "n_gpt": len(with_gpt),
            "ref": {m: {k: _mean([d["scores"][m][k] for d in docs if m in d["scores"]]) for k in ("p", "r", "f1")}
                    for m in methods},
            "vs_gpt": {m: {k: _mean([score(d["keywords"][m], d["keywords"][CHATGPT])[k] for d in with_gpt]) for k in ("p", "r", "f1")}
                       for m in methods if m != CHATGPT},
        }
    return out


def _bars_chart(stats: Dict, methods: List[str], key: str, title: str, subtitle: str) -> str:
    color = {m: COLORS[i % len(COLORS)] for i, m in enumerate(methods)}
    groups = []
    for name, s in stats.items():
        bars = []
        for m in methods:
            v = s[key].get(m)
            if v is None or v["f1"] is None:
                continue
            tip = f"{name} · {m}\nF1 {v['f1']:.3f} · P {v['p']:.3f} · R {v['r']:.3f}\n{s['n_gpt'] if key == 'vs_gpt' else s['n']} documents"
            bars.append({"label": m, "color": color[m], "value": v["f1"], "tip": tip})
        if bars:
            groups.append((name, bars))
    return charts.grouped_hbars(groups, title, subtitle)


def _table(stats: Dict, methods: List[str], key: str) -> str:
    cols = [m for m in methods if m in stats["all"][key]]
    head = "".join(f"<th class='num'>{_e(m)}<br><span class='m'>P / R / F1</span></th>" for m in cols)
    rows = []
    for name, s in stats.items():
        f1s = {m: s[key][m]["f1"] for m in cols if s[key][m]["f1"] is not None}
        best = max(f1s, key=f1s.get) if f1s else None
        cells = []
        for m in cols:
            v = s[key][m]
            cell = "–" if v["f1"] is None else f"{v['p']:.2f} / {v['r']:.2f} / {v['f1']:.2f}"
            cells.append(f"<td class='num{' best' if m == best else ''}'>{cell}</td>")
        n = s["n_gpt"] if key == "vs_gpt" else s["n"]
        rows.append(f"<tr><td>{_e(name)}</td><td class='num'>{n}</td>{''.join(cells)}</tr>")
    return f"<div class='scroll'><table class='stats'><tr><th>Set</th><th class='num'>n</th>{head}</tr>{''.join(rows)}</table></div>"


def _keyword_list(keywords: List[str], gold: List[str], gpt: Optional[List[str]]) -> str:
    gold_n = [normalize(g) for g in gold]
    gpt_n = [normalize(g) for g in gpt] if gpt else []
    items = []
    for i, kw in enumerate(keywords, 1):
        k = normalize(kw)
        in_gold = any(matches(k, g) for g in gold_n if g)
        in_gpt = any(matches(k, g) for g in gpt_n if g)
        classes = ["linked" if in_gold else "near"] + (["gpt"] if in_gpt else [])
        items.append(f"<li class='{' '.join(classes)}' data-k='{_e(' '.join(k))}'><span class='rank'>{i}</span>"
                     f"<span class='kw'>{_e(kw)}</span></li>")
    return f"<ol class='kws'>{''.join(items)}</ol>" if items else "<span class='err'>–</span>"


def _documents(data: Dict, methods: List[str], texts: Dict, set_name: str) -> str:
    rows = []
    for doc in (d for d in data["documents"] if d["set"] == set_name):
        text = texts.get(doc["id"], "")
        title, _, rest = text.partition("\n\n")
        chips = "".join(f"<span class='chip'>{_e(g)}</span>" for g in doc["gold"])
        gpt = doc["keywords"].get(CHATGPT)
        cells = "".join(f"<td>{_keyword_list(doc['keywords'].get(m, []), doc['gold'], gpt if m != CHATGPT else None)}</td>"
                        for m in methods)
        rows.append(f"<tr class='row'><td class='ds'><div class='title'>{_e(title[:300])}</div>"
                    f"<div class='id'>{_e(doc['id'])} · {_e(doc['lang'])}</div>"
                    f"<details><summary>Text</summary><p>{_e(rest)}</p></details>"
                    f"<div class='chips'>{chips}</div></td>{cells}</tr>")
    head = "".join(f"<th>{_e(m)}</th>" for m in methods)
    return f"<div class='scroll'><table><tr><th>Document · reference keywords</th>{head}</tr>{''.join(rows)}</table></div>"


def render(data: Dict, methods: List[str], texts: Dict, gpt: Dict, result_path: Path) -> str:
    stats = compute(data, methods)
    meta = data["meta"]
    n_docs = len(data["documents"])
    gpt_note = (f"ChatGPT keywords for {stats['all']['n_gpt']} of {n_docs} documents"
                if gpt else "no ChatGPT keywords yet (run chatgpt.py import)")
    set_names = [s for s in stats if s != "all"]
    toc = "".join(f"<a href='#set-{_e(s)}'>{_e(s)}</a>" for s in set_names)

    body = [f"<p class='meta'>Model {_e(meta['model'])} · top {meta['top_n']} keywords · {n_docs} documents · "
            f"{gpt_note} · run {_e(meta['created'])} ({_e(result_path.name)}) · report {datetime.now():%Y-%m-%d %H:%M}</p>",
            "<p class='legend'><span style='background:var(--ok)'>matches a reference keyword</span>"
            "<span style='background:var(--near)'>no reference match</span>"
            "<span style='box-shadow:inset 3px 0 0 var(--s6);background:var(--near)'>also chosen by ChatGPT</span></p>",
            "<h2>Against the reference keywords</h2>",
            "<div class='charts'>",
            _bars_chart(stats, methods, "ref", "F1@10 against the reference",
                        "Reference: GND subjects (l4s-*), author keywords (kpbiomed, radar-*), ChatGPT list (chatgpt-med-de)."),
            "</div>",
            _table(stats, methods, "ref")]
    if gpt:
        body += ["<h2>Agreement with ChatGPT</h2>",
                 "<p class='meta'>ChatGPT's keywords taken as the reference: recall = share of ChatGPT's keywords a method finds too.</p>",
                 "<div class='charts'>",
                 _bars_chart(stats, [m for m in methods if m != CHATGPT], "vs_gpt", "F1@10 against ChatGPT's keywords",
                             "Only documents ChatGPT answered for."),
                 "</div>",
                 _table(stats, methods, "vs_gpt")]
    body.append(f"<h2>Documents</h2><nav class='toc'>{toc}</nav>"
                "<p class='meta'>Numbers are ranks (1 = most relevant) · hover a keyword to highlight it across methods</p>")
    for s in set_names:
        body.append(f"<section class='domain' id='set-{_e(s)}'><h3>{_e(s)} ({stats[s]['n']})</h3>"
                    f"{_documents(data, methods, texts, s)}</section>")
    page = eval_report._page("Keyword benchmark — methods vs. reference and ChatGPT", "\n".join(body))
    return page.replace("</style>", EXTRA_CSS + "</style>", 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("result", nargs="?", type=Path, help="run.py result JSON (default: newest)")
    args = parser.parse_args()
    result_path = args.result or max(RESULTS_DIR.glob("benchmark_*.json"), key=lambda p: p.stat().st_mtime)
    data, methods, texts, gpt = _load(result_path)
    out = RESULTS_DIR / (result_path.stem + ".html")
    out.write_text(render(data, methods, texts, gpt, result_path), encoding="utf-8")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
