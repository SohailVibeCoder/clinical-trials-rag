# clinical-trials-rag
Clinical trials RAG system using hybrid BM25 + dense retrieval, cross-encoder reranking, and HyDE over 13,967 ClinicalTrials.gov trials for four hard-to-treat cancers

# Clinical Trials RAG System

**MSc Generative AI — University of Warwick**

A Retrieval-Augmented Generation (RAG) system for clinical trial intelligence across four hard-to-treat cancers: glioblastoma, pancreatic cancer, ovarian cancer, and small cell lung cancer (SCLC).

## Architecture

```
Query → Filter Parsing → [NCT ID Lookup | Hybrid Search (BM25 + Dense)] → RRF Fusion → Cross-Encoder Reranking → HyDE (semantic queries) → GPT-4o-mini Generation
```

### Components

| Component | Implementation |
|-----------|---------------|
| **Data source** | ClinicalTrials.gov v2 API |
| **Chunking** | Parent-child (5 types: metadata, summary, eligibility, intervention, description) |
| **Vector store** | ChromaDB (PersistentClient) with boolean metadata filters |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 (local, free) |
| **Sparse retrieval** | BM25Okapi (rank-bm25) |
| **Fusion** | Reciprocal Rank Fusion (k=60, BM25 weight=0.4, dense weight=0.6) |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| **HyDE** | GPT-4o-mini hypothetical document generation |
| **Generation** | GPT-4o-mini with structured prompt (temperature=0) |

## Setup

### Prerequisites

- Python 3.9+
- OpenAI API key (for GPT-4o-mini generation and HyDE)
- ~1 GB free disk space (vectorstore is ~692 MB)

### 1. Create virtual environment

Navigate to the project directory, then:

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

This installs: chromadb, sentence-transformers, rank-bm25, openai, requests, python-dotenv. The first import of `sentence-transformers` will download the embedding model (~80 MB) and cross-encoder model (~80 MB) automatically.

### 3. Configure API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

You can get an API key from [platform.openai.com](https://platform.openai.com/api-keys). The system uses GPT-4o-mini which costs ~$0.15 per 1M input tokens.

### 4. Verify installation

```bash
python -c "import chromadb, sentence_transformers, rank_bm25, openai; print('All dependencies installed')"
```

## Usage

### Option A: Jupyter Notebook (recommended)

Open `notebooks/rag_pipeline.ipynb` in VS Code or Jupyter and run all cells sequentially. Make sure the notebook kernel is set to the `.venv` virtual environment.

### Option B: Terminal

You can run individual pipeline stages from the terminal:

```bash
# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 1. Ingest trials from ClinicalTrials.gov API
python -m src.ingest

# 2. Chunk trials into parent-child chunks
python -m src.chunking

# 3. Run a single query through the full pipeline (interactive)
python -c "
from src.retrieval import build_indexes, hybrid_search
from src.reranker import rerank
from src.generation import generate_rag

collection, bm25_index = build_indexes()
query = 'What phase 3 glioblastoma trials are currently recruiting?'
candidates = hybrid_search(query, collection, bm25_index)
reranked = rerank(query=query, candidates=candidates, top_k=5)
answer = generate_rag(query, reranked, prompt_version='C')
print(answer)
"
```

### First Run Times

The first run will build all indexes from scratch:

| Step | Time | Runs Once? |
|------|------|------------|
| Ingest trials from API | ~5 min | Yes (saved to `data/raw/`) |
| Chunk into 63,906 chunks | ~10 sec | Yes (saved to `data/processed/`) |
| Build ChromaDB vectorstore | ~16 min | Yes (persisted to `vectorstore/`) |
| Build BM25 index | ~5 sec | Every run (in-memory) |

Subsequent runs detect existing data and skip ingestion/embedding (instant startup, except BM25 which rebuilds in ~5 seconds).

### Running the Evaluation

1. Run `notebooks/rag_pipeline.ipynb` Section 7 (Ablation Study) — 100 API calls across 5 configs (~25 min)
2. Run `notebooks/evaluation.ipynb` — scores generation quality and produces summary tables

## Project Structure

```
clinical-trials-rag/
├── src/
│   ├── ingest.py          # ClinicalTrials.gov API ingestion
│   ├── chunking.py        # Parent-child chunking (5 chunk types)
│   ├── retrieval.py       # Hybrid BM25 + Dense retrieval with RRF
│   ├── reranker.py        # Cross-encoder reranking
│   ├── hyde.py            # Hypothetical Document Embeddings
│   ├── generation.py      # GPT-4o-mini baseline + RAG generation
│   └── evaluate.py        # Ablation study framework
├── notebooks/
│   ├── rag_pipeline.ipynb # Main pipeline notebook (Sections 1-7)
│   └── evaluation.ipynb   # Generation quality evaluation
├── evaluation/
│   ├── test_queries.json  # 20 test queries (6 types)
│   ├── eval_scores.json   # LLM-assisted evaluation scores
│   └── ablation_results_*.json
├── data/
│   ├── raw/               # Ingested trial JSON
│   └── processed/         # Chunked data
├── vectorstore/           # ChromaDB persistent storage (~692 MB, not in git)
├── requirements.txt
└── README.md
```

## Evaluation Results

### Ablation Study (Retrieval Metrics)

| Config | P@5 | R@5 |
|--------|-----|-----|
| 1. Baseline LLM | 0.000 | 0.000 |
| 2. Naive RAG (dense only) | 0.025 | 0.009 |
| 3. Hybrid (BM25 + Dense) | 0.125 | 0.277 |
| 4. Hybrid + Reranker | 0.175 | 0.295 |
| 5. Full System (+ HyDE) | 0.175 | 0.295 |

### Generation Quality (Baseline vs Full System)

| Metric | Baseline | Full System | Delta |
|--------|----------|-------------|-------|
| Factual Accuracy | 0.500 | 0.800 | +0.300 |
| Grounding | 0.000 | 0.900 | +0.900 |
| Hallucination | 0.250 | 0.100 | −0.150 |
| Completeness | 0.325 | 0.600 | +0.275 |

Generation quality scores were assigned using LLM-assisted evaluation and cross-validated across three independent LLMs (GPT-4o, Gemini, Claude Sonnet).

## Key Design Decisions

- **Local embeddings**: all-MiniLM-L6-v2 runs locally (no API cost for embeddings)
- **Boolean metadata filters**: ChromaDB `$contains` is unreliable in v1.5.5; boolean fields (`has_glioblastoma`, `has_phase3`, etc.) ensure accurate filtered retrieval
- **NCT ID direct lookup**: queries containing NCT IDs bypass text search for exact metadata retrieval
- **HyDE skip for factual queries**: NCT ID lookups skip HyDE to avoid unnecessary API calls
- **Reranking with original query**: HyDE-expanded retrieval is always reranked against the original user query, not the hypothetical document

## Data

- **13,967 unique trials** across 4 cancer types
- **63,906 chunks** (5 types per trial)
- Dataset distribution: SCLC 45%, ovarian 26%, pancreatic 18%, glioblastoma 11%

> **Note**: The `vectorstore/` directory (~692 MB) is not included in the repository. It is automatically built on first run.
