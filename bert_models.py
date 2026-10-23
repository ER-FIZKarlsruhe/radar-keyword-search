# Alternative embedding models the "keybert" backend in iri_api.py can
# serve, selectable per-request via DocumentRequest.model. They're all
# transformer encoders loaded through the same AutoTokenizer/AutoModel +
# mean-pooling path, so adding a new one only means adding an entry here.
# Each entry carries:
#   - "label": the user-facing name /models hands to the frontend's dropdown.
#   - "kind": "sentence-embedding" for models trained for semantic similarity
#     (what KeyBERT's cosine ranking actually relies on), "masked-lm" for raw
#     domain-pretrained encoders. A masked-LM checkpoint is not automatically
#     a good embedding model, so the domain models are candidates to benchmark
#     against the sentence-embedding baselines, not assumed to beat them.
#   - "languages": what the model was trained on; English-only models can do
#     poorly on German metadata even when their domain fits.
#   - "domains": the research domains (NFDI consortium in parentheses) the
#     model is a candidate for. A model listed once can serve several domains,
#     so shared checkpoints like SciBERT aren't duplicated (or loaded twice).
#   - "prefix" (optional): text prepended to every input, for models trained
#     with one (E5 expects "query: " for symmetric similarity tasks).
# Every domain has at least two candidates. Domain picks are based on training
# corpus, not on measured keyword-extraction quality.
BERT_MODEL_REGISTRY = {
    # Interdisciplinary / multilingual baselines
    "multilingual-e5": {
        "hf_model": "intfloat/multilingual-e5-base",
        "label": "Interdisciplinary (multilingual baseline)",
        "kind": "sentence-embedding",
        "languages": ["multilingual"],
        "domains": [
            "Interdisciplinary", "Culture (NFDI4Culture)", "Language and text (Text+)",
            "Social sciences (KonsortSWD)", "Economics (BERD@NFDI)",
            "Earth sciences (NFDI4Earth)", "Mathematics (MaRDI)",
        ],
        "prefix": "query: ",
    },
    "multilingual-mpnet": {
        "hf_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "label": "Interdisciplinary (multilingual sentence embeddings)",
        "kind": "sentence-embedding",
        "languages": ["multilingual"],
        "domains": ["Interdisciplinary", "Language and text (Text+)", "Social sciences (KonsortSWD)"],
    },
    "scibert": {
        "hf_model": "allenai/scibert_scivocab_uncased",
        "label": "Scientific literature (general)",
        "kind": "masked-lm",
        "languages": ["en"],
        "domains": [
            "Interdisciplinary", "Chemistry (NFDI4Chem)", "Engineering (NFDI4Ing)",
            "Earth sciences (NFDI4Earth)", "Physics (PUNCH4NFDI)", "Mathematics (MaRDI)",
        ],
    },
    "gbert": {
        "hf_model": "deepset/gbert-base",
        "label": "German texts (general)",
        "kind": "masked-lm",
        "languages": ["de"],
        "domains": ["Language and text (Text+)"],
    },
    # Chemistry / NFDI4Chem
    "chemicalbert": {
        "hf_model": "recobo/chemical-bert-uncased",
        "label": "Chemistry (chemical industry and literature)",
        "kind": "masked-lm",
        "languages": ["en"],
        "domains": ["Chemistry (NFDI4Chem)"],
    },
    "materialsbert": {
        "hf_model": "pranav-s/MaterialsBERT",
        "label": "Chemistry (polymers and materials chemistry)",
        "kind": "masked-lm",
        "languages": ["en"],
        "domains": ["Chemistry (NFDI4Chem)"],
    },
    # Culture / NFDI4Culture
    "historic-multilingual-bert": {
        "hf_model": "dbmdz/bert-base-historic-multilingual-cased",
        "label": "Culture (historic multilingual texts / cultural heritage)",
        "kind": "masked-lm",
        "languages": ["multilingual"],
        "domains": ["Culture (NFDI4Culture)"],
    },
    "macberth": {
        "hf_model": "emanjavacas/MacBERTh",
        "label": "Culture (historical English / humanities)",
        "kind": "masked-lm",
        "languages": ["en"],
        "domains": ["Culture (NFDI4Culture)"],
    },
    # Engineering / NFDI4Ing
    "patentbert": {
        "hf_model": "anferico/bert-for-patents",
        "label": "Engineering (patents and technology)",
        "kind": "masked-lm",
        "languages": ["en"],
        "domains": ["Engineering (NFDI4Ing)"],
    },
    "matscibert": {
        "hf_model": "m3rg-iitd/matscibert",
        "label": "Engineering (materials science)",
        "kind": "masked-lm",
        "languages": ["en"],
        "domains": ["Engineering (NFDI4Ing)"],
    },
    # Biology / NFDI4Biodiversity, DataPLANT
    "biobert": {
        "hf_model": "dmis-lab/biobert-base-cased-v1.1",
        "label": "Biology (life sciences)",
        "kind": "masked-lm",
        "languages": ["en"],
        "domains": ["Biology (NFDI4Biodiversity, DataPLANT)"],
    },
    "biolinkbert": {
        "hf_model": "michiyasunaga/BioLinkBERT-base",
        "label": "Biology (life sciences, citation-aware)",
        "kind": "masked-lm",
        "languages": ["en"],
        "domains": ["Biology (NFDI4Biodiversity, DataPLANT)"],
    },
    # Medicine / NFDI4Health
    "pubmedbert": {
        "hf_model": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "label": "Medicine (biomedical literature)",
        "kind": "masked-lm",
        "languages": ["en"],
        "domains": ["Medicine (NFDI4Health)"],
    },
    "clinicalbert": {
        "hf_model": "emilyalsentzer/Bio_ClinicalBERT",
        "label": "Medicine (English clinical texts)",
        "kind": "masked-lm",
        "languages": ["en"],
        "domains": ["Medicine (NFDI4Health)"],
    },
    # Social sciences / KonsortSWD
    "mpnet": {
        "hf_model": "sentence-transformers/all-mpnet-base-v2",
        "label": "Social sciences (English sentence embeddings)",
        "kind": "sentence-embedding",
        "languages": ["en"],
        "domains": ["Social sciences (KonsortSWD)"],
    },
    # Economics / BERD@NFDI. The pretrained FinBERT, not ProsusAI/finbert:
    # that one is a sentiment classifier under a non-commercial licence.
    "finbert": {
        "hf_model": "yiyanghkust/finbert-pretrain",
        "label": "Economics (financial and business texts)",
        "kind": "masked-lm",
        "languages": ["en"],
        "domains": ["Economics (BERD@NFDI)"],
    },
    # Earth sciences / NFDI4Earth
    "indus": {
        "hf_model": "nasa-impact/nasa-smd-ibm-v0.1",
        "label": "Earth sciences (NASA science literature)",
        "kind": "masked-lm",
        "languages": ["en"],
        "domains": ["Earth sciences (NFDI4Earth)"],
    },
    # Physics / PUNCH4NFDI
    "physbert": {
        "hf_model": "thellert/physbert_cased",
        "label": "Physics (arXiv physics papers)",
        "kind": "masked-lm",
        "languages": ["en"],
        "domains": ["Physics (PUNCH4NFDI)"],
    },
    # Mathematics / MaRDI
    "mathbert": {
        "hf_model": "tbs17/MathBERT",
        "label": "Mathematics (textbooks and arXiv abstracts)",
        "kind": "masked-lm",
        "languages": ["en"],
        "domains": ["Mathematics (MaRDI)"],
    },
}
