# Keyword extraction evaluation

Scripts to measure how well the deployed keyword service works on existing RADAR metadata.
Standard library only — no extra dependencies.

| Step | Script | Output |
|------|--------|--------|
| 1. Harvest OAI-PMH metadata once | `harvest.py [domain…]` | `../../radar-keyword-eval-data/harvest/<oai_set>/page_NNNN.xml` (outside the repo) |
| 2. Run the keyword service | `run_extraction.py [domain…] [--limit N] [--require-abstract] [--models …]` | `results/<domain>_<timestamp>.json` |
| 3. Build reports + statistics | `report.py [result.json…]` | `reports/<name>.html`, `reports/<name>.stats.json` |

Run them from this folder (`cd evaluation`). Domains, models, terminology and TIB collection
are configured in `config.py`:

| Domain | OAI set | Terminology | TIB collection | Models |
|--------|---------|-------------|----------------|--------|
| culture | `radar4culture` | GND + TIB (`both`) | `NFDI4CULTURE` | `multilingual-e5`, `historic-multilingual-bert` |
| chem | `radar4chem` | TIB | `NFDI4CHEM` | `scibert`, `chemicalbert` |
| memory | `radar4memory` | GND + TIB (`both`) | `NFDI4CULTURE` (TIB has no NFDI4Memory collection) | `multilingual-e5`, `historic-multilingual-bert` |

Each model runs in two modes: **exact** (`max_distance=0`) and **fuzzy** (the service's default
threshold), with 10 keywords. The document sent is title + abstract.

Environment overrides: `KEYWORD_SERVICE_URL` (default `https://ai.radar-service.eu/keywords`),
`RADAR_OAI_URL`, `KEYWORD_EVAL_HARVEST_DIR`, `KEYWORD_EVAL_RESULTS_DIR`, `KEYWORD_EVAL_REPORTS_DIR`.

## Result JSON

```text
{ schema_version, meta: {domain, oai_set, service_url, service_health, git_commit, started, finished, ...},
  runs: [ {id: "<model>/<mode>", model, mode, request: {...exact payload sent...}} ],
  datasets: [ {identifier, title, abstract, author_keywords, subject_areas, document,
               results: { "<run id>": {status, error, elapsed_s,
                                       keywords: [ {rank, keyword, match: null | {iri, label, best_term,
                                                    distance, matched, source, ontology_name, ...}} ]}}} ] }
```

`rank` is the KeyBERT order (1 = most relevant). The service does not return KeyBERT's
similarity scores, so rank is the only relevance signal available. `match` is the service's
closest candidate; `matched: true` means it is within the threshold (and, for TIB, the IRI
is live).

## Statistics

`report.py` computes per run: keywords and linked keywords per dataset (median/mean/min/max),
link rate, links at distance 0, closest-candidate distance, rank of linked keywords, links by
source and ontology, agreement with the authors' own keywords (recall/precision, matched
case-insensitively by equality or containment), and response time. It also computes the
keyword-list overlap (Jaccard) between every pair of runs.

The HTML report opens with inline-SVG charts (`charts.py`, no external libraries, hover for
values): link rate exact vs fuzzy, linked keywords per dataset, match distance of fuzzy links,
link rate by KeyBERT rank, agreement with author keywords, and — when both are searched —
GND vs TIB share. Colour always identifies the model; exact vs fuzzy is shown by hollow vs
filled markers and dashed vs solid lines.
