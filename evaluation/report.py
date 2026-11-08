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

"""Build HTML reports and statistics from run_extraction.py result files.

For each result JSON this writes REPORTS_DIR/<name>.html (one row per dataset,
one column per model/mode run, keywords rank-aligned) and <name>.stats.json.

    python evaluation/report.py                          # newest result file per domain
    python evaluation/report.py evaluation/results/chem_20261108-101500.json
"""

import argparse
import html
import json
import statistics
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import charts
from config import DOMAINS, REPORTS_DIR, RESULTS_DIR


# -------------------------------
# Statistics
# -------------------------------
def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _is_linked(kw: Dict) -> bool:
    return bool(kw["match"] and kw["match"].get("matched"))


def author_hit(keyword: str, author_keywords: List[str]) -> Optional[str]:
    """The author keyword an extracted keyword agrees with (equal, or one contains the other)."""
    k = _norm(keyword)
    for author in author_keywords:
        a = _norm(author)
        if k == a or (len(k) > 3 and k in a) or (len(a) > 3 and a in k):
            return author
    return None


def _summary(values: List[float]) -> Dict:
    if not values:
        return {"n": 0, "median": None, "mean": None, "min": None, "max": None}
    return {
        "n": len(values),
        "median": statistics.median(values),
        "mean": round(statistics.mean(values), 2),
        "min": min(values),
        "max": max(values),
    }


def _jaccard(a: set, b: set) -> Optional[float]:
    return round(len(a & b) / len(a | b), 3) if a | b else None


def compute_stats(data: Dict) -> Dict:
    runs = [r["id"] for r in data["runs"]]
    per_run = {}
    for run in runs:
        extracted, linked, exact_linked, distances, linked_ranks, elapsed = [], [], [], [], [], []
        sources, ontologies, linked_distances = Counter(), Counter(), Counter()
        by_rank: Dict[int, Dict[str, int]] = {}
        errors = 0
        author_total = author_found = with_author = extracted_agreeing = 0
        for ds in data["datasets"]:
            result = ds["results"].get(run)
            if not result or result["status"] != "ok":
                errors += 1
                continue
            kws = result["keywords"]
            extracted.append(len(kws))
            linked_kws = [k for k in kws if _is_linked(k)]
            linked.append(len(linked_kws))
            exact_linked.append(sum(1 for k in linked_kws if k["match"]["distance"] == 0))
            distances.extend(k["match"]["distance"] for k in kws if k["match"])
            linked_ranks.extend(k["rank"] for k in linked_kws)
            elapsed.append(result["elapsed_s"])
            for k in linked_kws:
                sources[k["match"]["source"]] += 1
                ontologies[k["match"].get("ontology_name") or k["match"]["source"]] += 1
                linked_distances[k["match"]["distance"]] += 1
            for k in kws:
                bucket = by_rank.setdefault(k["rank"], {"linked": 0, "total": 0})
                bucket["total"] += 1
                bucket["linked"] += _is_linked(k)
            if ds["author_keywords"]:
                with_author += 1
                author_total += len(ds["author_keywords"])
                hits = {author_hit(k["keyword"], ds["author_keywords"]) for k in kws} - {None}
                author_found += len(hits)
                extracted_agreeing += sum(1 for k in kws if author_hit(k["keyword"], ds["author_keywords"]))
        n_extracted, n_linked = sum(extracted), sum(linked)
        n_with_author_extracted = sum(
            len(ds["results"][run]["keywords"]) for ds in data["datasets"]
            if ds["author_keywords"] and ds["results"].get(run, {}).get("status") == "ok"
        )
        per_run[run] = {
            "datasets_ok": len(extracted),
            "datasets_failed": errors,
            "keywords_per_dataset": _summary(extracted),
            "linked_per_dataset": _summary(linked),
            "linked_per_dataset_values": linked,
            "linked_distance_counts": {str(d): n for d, n in sorted(linked_distances.items())},
            "linked_by_rank": {str(r): by_rank[r] for r in sorted(by_rank)},
            "exact_linked_per_dataset": _summary(exact_linked),
            "keywords_total": n_extracted,
            "linked_total": n_linked,
            "link_rate": round(n_linked / n_extracted, 3) if n_extracted else None,
            "datasets_without_link": sum(1 for n in linked if n == 0),
            "candidate_distance": _summary(distances),
            "linked_rank": _summary(linked_ranks),
            "linked_by_source": dict(sources),
            "linked_by_ontology": dict(ontologies.most_common()),
            "author_keywords": {
                "datasets_with_author_keywords": with_author,
                "author_keywords_total": author_total,
                "author_keywords_found": author_found,
                "recall": round(author_found / author_total, 3) if author_total else None,
                "precision": round(extracted_agreeing / n_with_author_extracted, 3) if n_with_author_extracted else None,
            },
            "elapsed_s": _summary(elapsed),
        }

    # How similar two runs' keyword lists are, averaged (median) over datasets.
    agreement = {}
    for a, b in combinations(runs, 2):
        scores = []
        for ds in data["datasets"]:
            ra, rb = ds["results"].get(a), ds["results"].get(b)
            if ra and rb and ra["status"] == rb["status"] == "ok":
                j = _jaccard({_norm(k["keyword"]) for k in ra["keywords"]}, {_norm(k["keyword"]) for k in rb["keywords"]})
                if j is not None:
                    scores.append(j)
        agreement[f"{a} vs {b}"] = {"keyword_jaccard_median": statistics.median(scores) if scores else None, "n": len(scores)}

    docs = [ds["document"] for ds in data["datasets"]]
    return {
        "source": data["meta"],
        "datasets": {
            "count": len(docs),
            "with_abstract": sum(1 for ds in data["datasets"] if ds["abstract"]),
            "with_author_keywords": sum(1 for ds in data["datasets"] if ds["author_keywords"]),
            "document_words": _summary([len(d.split()) for d in docs]),
        },
        "runs": per_run,
        "run_agreement": agreement,
    }


# -------------------------------
# HTML
# -------------------------------
CSS = """
:root { --bg:#fff; --fg:#1d2330; --muted:#667085; --line:#e4e7ec; --head:#f5f7fa;
  --ok:#e7f6ec; --ok-fg:#1a6b3c; --near:#f4f4f5; --none:#fdecec; --none-fg:#a1332c; --hl:#fff2b3; --author:#b7791f; }
@media (prefers-color-scheme: dark) { :root { --bg:#14171d; --fg:#e6e8ec; --muted:#98a2b3; --line:#2b313c;
  --head:#1c2129; --ok:#163626; --ok-fg:#7fd4a3; --near:#232831; --none:#3a1d1d; --none-fg:#f0a39d; --hl:#5a4a12; --author:#e0b25a; } }
* { box-sizing:border-box; }
body { margin:0; padding:24px 16px; background:var(--bg); color:var(--fg); font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:17px; margin:32px 0 8px; }
.meta { color:var(--muted); font-size:12.5px; margin-bottom:16px; }
.scroll { overflow-x:auto; border:1px solid var(--line); border-radius:8px; }
table { border-collapse:collapse; width:100%; }
th, td { border-bottom:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; }
th { background:var(--head); font-weight:600; position:sticky; top:0; white-space:nowrap; }
.stats td.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
.stats td:first-child { color:var(--muted); white-space:nowrap; }
.ds { min-width:260px; max-width:340px; }
.ds .title { font-weight:600; }
.ds .id, .ds .sub { color:var(--muted); font-size:12px; }
.ds details { font-size:12.5px; margin-top:6px; } .ds details p { white-space:pre-wrap; margin:4px 0; }
.chips { margin-top:6px; } .chip { display:inline-block; border:1px solid var(--author); color:var(--author); border-radius:10px; padding:0 7px; margin:2px 2px 0 0; font-size:12px; }
ol.kws { list-style:none; margin:0; padding:0; min-width:220px; }
ol.kws li { display:grid; grid-template-columns:22px 1fr; gap:4px; padding:3px 6px; margin-bottom:3px; border-radius:5px; cursor:default; }
ol.kws li.linked { background:var(--ok); } ol.kws li.near { background:var(--near); } ol.kws li.none { background:var(--none); }
ol.kws li.hl { outline:2px solid var(--hl); background:var(--hl); }
.rank { color:var(--muted); font-size:12px; font-variant-numeric:tabular-nums; }
.kw { font-weight:500; } .kw .star { color:var(--author); }
.m { font-size:12px; color:var(--muted); } .linked .m a { color:var(--ok-fg); } .none .m { color:var(--none-fg); }
.err { color:var(--none-fg); font-size:12.5px; }
.legend span { display:inline-block; padding:1px 8px; border-radius:5px; margin-right:8px; font-size:12.5px; }
"""

JS = """
document.querySelectorAll('tr.row').forEach(row => {
  row.addEventListener('mouseover', e => {
    const li = e.target.closest('li[data-k]'); if (!li) return;
    row.querySelectorAll('li[data-k="' + CSS.escape(li.dataset.k) + '"]').forEach(x => x.classList.add('hl'));
  });
  row.addEventListener('mouseout', () => row.querySelectorAll('li.hl').forEach(x => x.classList.remove('hl')));
});
"""


def _e(value) -> str:
    return html.escape(str(value)) if value is not None else ""


def _fmt(value) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def _keyword_item(kw: Dict, author_keywords: List[str]) -> str:
    match = kw["match"]
    star = ' <span class="star" title="agrees with an author keyword">★</span>' if author_hit(kw["keyword"], author_keywords) else ""
    if match and match.get("matched"):
        cls = "linked"
        where = match.get("ontology_name") or match["source"].upper()
        detail = f'<a href="{_e(match["iri"])}" target="_blank" rel="noopener">{_e(match["label"])}</a> · {_e(where)} · d={match["distance"]}'
    elif match:
        cls = "near"
        where = match.get("ontology_name") or match["source"].upper()
        detail = f'closest: {_e(match["best_term"])} · {_e(where)} · d={match["distance"]} (not linked)'
    else:
        cls = "none"
        detail = "no candidate found"
    return (
        f'<li class="{cls}" data-k="{_e(_norm(kw["keyword"]))}"><span class="rank">{kw["rank"]}</span>'
        f'<span><span class="kw">{_e(kw["keyword"])}</span>{star}<br><span class="m">{detail}</span></span></li>'
    )


def _stats_table(stats: Dict) -> str:
    runs = list(stats["runs"])
    rows = [
        ("Datasets ok / failed", lambda s: f'{s["datasets_ok"]} / {s["datasets_failed"]}'),
        ("Keywords per dataset (median)", lambda s: _fmt(s["keywords_per_dataset"]["median"])),
        ("Linked per dataset (median)", lambda s: _fmt(s["linked_per_dataset"]["median"])),
        ("Linked per dataset (mean)", lambda s: _fmt(s["linked_per_dataset"]["mean"])),
        ("Linked per dataset (min–max)", lambda s: f'{_fmt(s["linked_per_dataset"]["min"])}–{_fmt(s["linked_per_dataset"]["max"])}'),
        ("Linked at distance 0 (median)", lambda s: _fmt(s["exact_linked_per_dataset"]["median"])),
        ("Link rate (linked / extracted)", lambda s: f'{s["linked_total"]} / {s["keywords_total"]} = {_fmt(s["link_rate"])}'),
        ("Datasets with no link at all", lambda s: _fmt(s["datasets_without_link"])),
        ("Closest-candidate distance (median)", lambda s: _fmt(s["candidate_distance"]["median"])),
        ("Rank of linked keywords (median)", lambda s: _fmt(s["linked_rank"]["median"])),
        ("Linked by source", lambda s: ", ".join(f"{k}: {v}" for k, v in s["linked_by_source"].items()) or "–"),
        ("Top ontologies", lambda s: ", ".join(f"{k}: {v}" for k, v in list(s["linked_by_ontology"].items())[:5]) or "–"),
        ("Author keywords found (recall)", lambda s: f'{s["author_keywords"]["author_keywords_found"]} / {s["author_keywords"]["author_keywords_total"]} = {_fmt(s["author_keywords"]["recall"])}'),
        ("Extracted agreeing with authors (precision)", lambda s: _fmt(s["author_keywords"]["precision"])),
        ("Response time s (median)", lambda s: _fmt(s["elapsed_s"]["median"])),
    ]
    head = "".join(f"<th>{_e(r)}</th>" for r in runs)
    body = "".join(
        f"<tr><td>{_e(label)}</td>" + "".join(f'<td class="num">{_e(fn(stats["runs"][r]))}</td>' for r in runs) + "</tr>"
        for label, fn in rows
    )
    agreement = "".join(
        f'<tr><td>{_e(pair)}</td><td class="num">{_fmt(v["keyword_jaccard_median"])}</td><td class="num">{v["n"]}</td></tr>'
        for pair, v in stats["run_agreement"].items()
    )
    return (
        f'<div class="scroll"><table class="stats"><tr><th>Metric</th>{head}</tr>{body}</table></div>'
        f'<h2>Keyword agreement between runs</h2><p class="meta">Median Jaccard overlap of the extracted keyword lists per dataset '
        f'(1 = identical lists). Exact and fuzzy runs of one model should extract the same keywords; only linking differs.</p>'
        f'<div class="scroll"><table class="stats"><tr><th>Runs</th><th>Jaccard (median)</th><th>Datasets</th></tr>{agreement}</table></div>'
    )


def _charts(data: Dict, stats: Dict) -> str:
    runs = data["runs"]
    models = list(dict.fromkeys(r["model"] for r in runs))
    color = {m: charts.SERIES[i % len(charts.SERIES)] for i, m in enumerate(models)}
    by_id = stats["runs"]
    run_of = {(r["model"], r["mode"]): r["id"] for r in runs}
    max_kw = max([r["request"].get("max_keywords") or 10 for r in runs] + [1])
    out = []

    rows = []
    for m in models:
        ex, fz = by_id.get(run_of.get((m, "exact"))), by_id.get(run_of.get((m, "fuzzy")))
        tip = lambda s, mode: f"{m} · {mode}\n{s['linked_total']} of {s['keywords_total']} keywords linked ({charts.pct(s['link_rate'])})" if s else ""  # noqa: E731
        rows.append({"label": m, "color": color[m], "exact": ex and ex["link_rate"], "fuzzy": fz and fz["link_rate"],
                     "exact_tip": tip(ex, "exact"), "fuzzy_tip": tip(fz, "fuzzy")})
    out.append(charts.dumbbell(rows, "Link rate: exact vs fuzzy",
                               "Share of all extracted keywords linked to a vocabulary term. The gap is what fuzzy matching adds."))

    panels = []
    for r in runs:
        values = by_id[r["id"]]["linked_per_dataset_values"]
        panels.append({"label": f"{r['model']} · {r['mode']}", "color": color[r["model"]],
                       "counts": Counter(values), "marker": by_id[r["id"]]["linked_per_dataset"]["median"], "tip_unit": "datasets"})
    out.append(charts.histograms(panels, list(range(0, max_kw + 1)), "Linked keywords per dataset",
                                 "How many of each dataset's keywords got linked; line = median.", "linked keywords", "datasets"))

    fuzzy_runs = [r for r in runs if r["mode"] == "fuzzy"]
    if fuzzy_runs:
        max_d = max([int(d) for r in fuzzy_runs for d in by_id[r["id"]]["linked_distance_counts"]] + [5])
        panels = [{"label": f"{r['model']} · fuzzy", "color": color[r["model"]],
                   "counts": {int(d): n for d, n in by_id[r["id"]]["linked_distance_counts"].items()},
                   "tip_unit": "linked keywords"} for r in fuzzy_runs]
        out.append(charts.histograms(panels, list(range(0, max_d + 1)), "Match distance of linked keywords (fuzzy)",
                                     "Hamming distance between keyword and linked label. Everything right of 0 is a non-exact link - check these in the table below.",
                                     "distance", "linked keywords"))

    series = []
    for r in runs:
        points = {int(rank): (b["linked"] / b["total"], b["linked"], b["total"])
                  for rank, b in by_id[r["id"]]["linked_by_rank"].items() if b["total"]}
        series.append({"label": f"{r['model']} · {r['mode']}", "color": color[r["model"]], "dashed": r["mode"] == "exact", "points": points})
    out.append(charts.lines(series, list(range(1, max_kw + 1)), "Link rate by keyword rank",
                            "Are the most relevant keywords (rank 1) more often linkable than the tail?", "KeyBERT rank (1 = most relevant)"))

    # Exact and fuzzy runs of a model extract the same keywords, so author agreement is per model.
    groups = [("Recall", []), ("Precision", [])]
    for m in models:
        s = by_id.get(run_of.get((m, "fuzzy")) or run_of.get((m, "exact")))
        a = s["author_keywords"]
        groups[0][1].append({"label": m, "color": color[m], "value": a["recall"],
                             "tip": f"{m}\n{a['author_keywords_found']} of {a['author_keywords_total']} author keywords found"})
        groups[1][1].append({"label": m, "color": color[m], "value": a["precision"],
                             "tip": f"{m}\nshare of extracted keywords agreeing with an author keyword: {charts.pct(a['precision'])}"})
    n_author = next(iter(by_id.values()))["author_keywords"]["datasets_with_author_keywords"]
    out.append(charts.grouped_hbars(groups, "Agreement with author keywords",
                                    f"Over the {n_author} datasets that have author keywords (equal or contained, case-insensitive)."))

    all_sources = sorted({src for s in by_id.values() for src in s["linked_by_source"]})
    if len(all_sources) > 1:
        palette = {"gnd": ("GND", charts.SERIES[2], False), "tib": ("TIB", charts.SERIES[3], True)}
        categories = [palette.get(src, (src.upper(), charts.SERIES[2], False)) for src in all_sources]
        rows = [(f"{r['model']} · {r['mode']}", {palette.get(k, (k.upper(),))[0]: v for k, v in by_id[r["id"]]["linked_by_source"].items()})
                for r in runs]
        out.append(charts.stacked_hbars(rows, categories, "Linked keywords by vocabulary",
                                        "Which terminology the winning match came from."))
    return f'<div class="charts">{"".join(out)}</div>'


def render_html(data: Dict, stats: Dict) -> str:
    meta = data["meta"]
    runs = data["runs"]
    domain_cfg = runs[0]["request"] if runs else {}
    ds_stats = stats["datasets"]
    header = "".join(f"<th>{_e(r['model'])}<br><span class='m'>{_e(r['mode'])}</span></th>" for r in runs)
    rows = []
    for ds in data["datasets"]:
        chips = "".join(f'<span class="chip">{_e(k)}</span>' for k in ds["author_keywords"]) or '<span class="sub">no author keywords</span>'
        abstract = f"<p>{_e(ds['abstract'])}</p>" if ds["abstract"] else "<p><em>No abstract — only the title was sent.</em></p>"
        cells = [
            f'<td class="ds"><div class="title">{_e(ds["title"])}</div>'
            f'<div class="id"><a href="https://doi.org/{_e(ds["identifier"])}" target="_blank" rel="noopener">{_e(ds["identifier"])}</a>'
            f' · {len(ds["document"].split())} words · {_e(", ".join(ds["subject_areas"]))}</div>'
            f'<div class="chips">{chips}</div><details><summary>Abstract</summary>{abstract}</details></td>'
        ]
        for run in runs:
            result = ds["results"].get(run["id"])
            if not result or result["status"] != "ok":
                cells.append(f'<td><div class="err">{_e(result["error"] if result else "not run")}</div></td>')
            else:
                items = "".join(_keyword_item(k, ds["author_keywords"]) for k in result["keywords"])
                cells.append(f'<td><ol class="kws">{items}</ol></td>')
        rows.append(f'<tr class="row">{"".join(cells)}</tr>')

    title = f"Keyword extraction — {meta['domain']} ({meta['oai_set']})"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(title)}</title><style>{CSS}{charts.CHART_CSS}</style></head><body>
<h1>{_e(title)}</h1>
<div class="meta">Service {_e(meta['service_url'])} · run {_e(meta['started'][:19])} UTC · commit {_e((meta.get('git_commit') or '')[:7])} ·
terminology <b>{_e(domain_cfg.get('terminology'))}</b> · collection <b>{_e(domain_cfg.get('ontology_collection'))}</b> ·
max {_e(domain_cfg.get('max_keywords'))} keywords · document = {_e(meta['document'])}<br>
{ds_stats['count']} datasets, {ds_stats['with_abstract']} with abstract, {ds_stats['with_author_keywords']} with author keywords,
median {_fmt(ds_stats['document_words']['median'])} words per document.
Exact = max_distance 0, fuzzy = service default threshold.</div>
<h2>Charts</h2>
{_charts(data, stats)}
<h2>Statistics</h2>
{_stats_table(stats)}
<h2>Keywords per dataset</h2>
<p class="meta legend"><span style="background:var(--ok)">linked</span><span style="background:var(--near)">candidate too far / IRI dead</span>
<span style="background:var(--none)">no candidate</span>★ agrees with an author keyword · d = Hamming distance ·
numbers are KeyBERT rank (1 = most relevant) · hover a keyword to highlight it across runs</p>
<div class="scroll"><table><tr><th>Dataset</th>{header}</tr>{''.join(rows)}</table></div>
<script>{JS}{charts.CHART_JS}</script>
</body></html>"""


def build(result_path: Path) -> Path:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    stats = compute_stats(data)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    html_path = REPORTS_DIR / f"{result_path.stem}.html"
    html_path.write_text(render_html(data, stats), encoding="utf-8")
    (REPORTS_DIR / f"{result_path.stem}.stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  {html_path}")
    return html_path


def newest_results() -> List[Path]:
    paths = []
    for domain in DOMAINS:
        found = sorted(RESULTS_DIR.glob(f"{domain}_*.json"))
        if found:
            paths.append(found[-1])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="*", type=Path, help="result JSON files (default: newest per domain)")
    args = parser.parse_args()
    paths = args.results or newest_results()
    if not paths:
        parser.error(f"no result files in {RESULTS_DIR} - run run_extraction.py first")
    for path in paths:
        build(path)


if __name__ == "__main__":
    main()
