# Keyword benchmark

Compares keyword extraction methods against reference keywords, without the service and
without any TIB/GND lookup: which method picks the keywords a human (or an LLM) would pick?

| Step | Script | Output |
|------|--------|--------|
| 1. Download the sources once | see below | `$KEYWORD_BENCHMARK_DIR/raw/` |
| 2. Build the sets | `prepare.py [--per-set 30] [--seed 42]` | `$KEYWORD_BENCHMARK_DIR/sets/<set>.jsonl` |
| 3. Run the comparison | `run.py [--model multilingual-e5] [--sets …] [--methods …]` | table on stdout, `results/benchmark_<model>_<timestamp>.json` |

`KEYWORD_BENCHMARK_DIR` defaults to `../data/benchmarks` (`evaluation/data/`, git-ignored). The data
must never be committed: the repository is public and KPBiomed may not be redistributed
commercially. `evaluation/data/` also holds `documents/` (each benchmark text as a plain file) and
`documents-index.tsv` (file -> set, id, reference and ChatGPT keywords) for reading.

## Sets

| Set | Language | Source | Reference keywords |
|-----|----------|--------|--------------------|
| `l4s-chem-de/en` | de / en | LLMs4Subjects (SemEval-2025 Task 5), DDC 540 | GND subjects assigned by subject librarians |
| `l4s-med-de/en` | de / en | LLMs4Subjects, DDC 610 | GND subjects |
| `l4s-hum-de/en` | de / en | LLMs4Subjects, DDC 1xx, 2xx, 4xx, 7xx (without 71x), 8xx, 9xx | GND subjects |
| `kpbiomed-en` | en | KPBiomed test split (PubMed) | author keywords |
| `radar-chem`, `radar-culture` | mixed | RADAR harvest (`../harvest.py`) | author keywords |
| `custom/<name>.json` | any | hand-made | anything, e.g. an LLM's keyword list |

GND subjects are *assigned*: many of them don't occur in the text, so no extractive method can
find them. The column `gold∈text` shows the share of reference keywords that occur in the text,
i.e. the best recall an extractive method could reach.

Licenses: LLMs4Subjects CC BY 4.0 (TIBKAT metadata CC0), KPBiomed CC BY-NC 4.0, RADAR metadata
as published by RADAR.

## Download

```bash
D=../data/benchmarks/raw && mkdir -p $D && cd $D
# LLMs4Subjects: only the gold-standard test set (sparse checkout, ~360 MB)
git clone --depth 1 --filter=blob:none --sparse https://github.com/jd-coderepos/llms4subjects.git
git -C llms4subjects sparse-checkout set shared-task-datasets/TIBKAT/all-subjects/data/test/gold-standard-testset
# KPBiomed test split (~45 MB)
curl -L -o kpbiomed_test.jsonl https://huggingface.co/datasets/taln-ls2n/kpbiomed/resolve/main/test.jsonl
```

RADAR sets come from the evaluation harvest (`../harvest.py`).

The raw downloads are only needed to (re)build the sets with `prepare.py`; `run.py`,
`chatgpt.py` and `benchmark_report.py` only read `sets/`, `custom/` and `chatgpt/`. Keep `sets/`:
the ChatGPT answers belong to exactly these documents. A rebuild only yields the same sets with
the default `--per-set 30 --seed 42` and unchanged sources; otherwise new ChatGPT answers are needed.

## Methods

The service's two `candidates` options, run in-process with the service's own code:

| Method | Candidates | Ranking |
|--------|------------|---------|
| `words` | single words without stop words (service default) | cosine to the document embedding (KeyBERT, `keyword_extraction.py`) |
| `pos-ngrams` | contiguous 1-3 word sequences with a noun, never across stop words, punctuation, verbs or generic adjectives (spaCy as tagger only) | standardised cosine plus position, frequency and length terms, MMR (`pos_ngrams.py`) |

The weights in `pos_ngrams.py` were tuned against ChatGPT's keywords on every other document
of each set and checked on the rest (F1@10 vs. ChatGPT with `multilingual-e5`: 0.227 tuned,
0.232 held out). Variants that scored lower and were dropped: KeyBERT on 1-3 word n-grams
(0.06) or spaCy noun phrases (0.21), rare-word scoring with wordfreq (0.17-0.19), the same
ranking on noun phrases (0.22), a sentence-mean document embedding, and dropping phrases that
extend a picked keyword or replacing the picked part by them (0.21). Skipping parts of a picked
keyword (`keratosis` after `keratosis pilaris`) kept F1 vs. ChatGPT (0.227) and cut overlapping
keywords from 4.4 to 1.6 pairs per document, but lowered F1 vs. the reference keywords from
0.115 to 0.091 (these are mostly single words that a picked phrase now hides).

## Scoring

Precision, recall and F1 of the top 10 keywords per document, averaged per set. A keyword
matches a reference keyword if both have the same number of words and each word pair differs
by at most a quarter of the word's length (edit distance), ignoring case and GND qualifiers
(`Wien <Stadt>`).

## Requirements

The service's `requirements.txt`. On the development host, download models with
`HF_HUB_DISABLE_XET=1` (see `CLAUDE.md`).

## Comparison with ChatGPT

ChatGPT's web interface may not be automated, so its keywords are collected by hand:

```bash
python chatgpt.py prompts     # one prompt per 10 documents: $KEYWORD_BENCHMARK_DIR/chatgpt/prompts/batch_NN.txt
# paste each prompt into ChatGPT, save the answer as $KEYWORD_BENCHMARK_DIR/chatgpt/answers/batch_NN.txt
python chatgpt.py import      # parses the answers, lists missing documents
python benchmark_report.py    # HTML report of the newest run.py result, with ChatGPT as one more method
```

The report (`results/<run>.html`, styled like `../report.py`) shows every method against the
reference keywords, and every method against ChatGPT's keywords taken as the reference.
