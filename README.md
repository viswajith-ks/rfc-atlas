# RFC Atlas

A structured, deterministic Retrieval-Augmented Generation (RAG) pipeline and stateless API for the complete IETF RFC ecosystem.

RFC Atlas is designed to handle the 50-year historical volatility of the IETF standards corpus. It compiles legacy plaintext and modern XML documents into a strictly typed, versioned vector database, resolving protocol lineage (obsoletions/updates) and official technical errata at runtime.

## Core Architecture

The backend operates strictly on a **data-first compiler pattern**, isolating the offline heavy data processing from the stateless, lightweight RAG runtime.

* **Hybrid Ingestion Engine:** Parses modern XML (`xml2rfc` v3) via `lxml` and legacy TXT formats via heuristic regex state machines. Normalizes everything into a unified Pydantic canonical tree.
* **Hierarchy-Aware Chunking:** Bypasses naive token-window splitting. Chunking respects document boundaries, preserving structural integrity for `abnf`, `sourcecode`, `artwork`, and `prose`.
* **Zero-IPC Vector Forge:** Designed to run ephemerally on Kaggle Dual-T4 GPUs. Uses `nomic-embed-text-v1.5` in FP16 precision, orchestrating chunk assignment without IPC overhead.
* **LanceDB Hybrid Retrieval:** Embeddings are written to PyArrow Parquet files and pushed down into segregated LanceDB tables (`prose`, `security`, `abnf`, `references`). Queries run against both Dense (HNSW) and Sparse (Tantivy BM25) indices.
* **In-Memory Temporal Lineage:** Replaces heavy graph databases with an $O(1)$ memory lookup to dynamically trace `OBSOLETES` and `UPDATES` trajectories, preventing the LLM from synthesizing deprecated standards.
* **Runtime Errata Interception:** Technical errata are fetched directly from the IETF and injected into semantic chunks at runtime before reaching the LLM context window.
* **Synthesis & API Boundary:** Uses a local Cross-Encoder (`bge-reranker-v2-m3`) for absolute precision re-ranking, and queries Gemini 2.5 Flash for the final response. Exposed via a FastAPI Server-Sent Events (SSE) endpoint compatible with the Vercel AI SDK.

## Current Development Status

The Python backend is **100% feature-complete** (Phases 0 through 7).
* [x] Corpus Sync (`rsync` / `curl`)
* [x] Ingestion & Parsing
* [x] Chunking & Embeddings
* [x] LanceDB Vector Store
* [x] Retrieval, Reranking, & Context Assembly
* [x] LLM Synthesis
* [x] FastAPI Boundary & Streaming
* [ ] Next.js Frontend (Pending Phase 8)

## Requirements

* **Python 3.11+**
* **Linux OS** (Strictly required for OS-level `fork()` Copy-on-Write memory sharing during the Pebble multiprocessing ingestion phase).
* **`rsync` & `curl`** (Required for syncing the authoritative IETF corpus natively from `rsync.rfc-editor.org`).

## Installation

Clone the repository and install the project along with its optional dependency groups.

```bash
git clone https://github.com/yourusername/rfc-atlas.git
cd rfc-atlas

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install all components (vector-store, synthesis, api, and dev tools)
pip install -e ".[vector-store,synthesis,api,dev]"
```

## API Credentials & Kaggle Integration

To run the pipeline and queries successfully, you must configure external API credentials.
Create a `.env` file in the root directory:

```text
GEMINI_API_KEY=your_google_gemini_key_here
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_token
```

### ⚠️ Important: Kaggle Account Requirement
Because generating 3.5+ million dense vectors for the historical RFC archive takes over 15 hours on a local CPU, **RFC Atlas leverages Kaggle's free Dual-T4 GPU notebooks to forge the vector database.**

If you run the pipeline `--from-scratch`, the system will automatically zip the repository, push it to Kaggle, run the job on the GPUs, and download the finished Parquet database back to your local machine. **This requires a valid Kaggle account and API token.** If you do not have Kaggle credentials configured, the `--from-scratch` pipeline will crash.

*(Note: Standard incremental updates run purely on your local CPU and do not require Kaggle).*

## Running the End-to-End Pipeline

RFC Atlas comes with a master orchestrator script that automatically syncs the latest IETF RFC corpus, normalizes the data, creates the vector database, and runs QA audits.

**To run an incremental update (default):**
*(Downloads only new/changed RFCs, embeds them locally on CPU, and updates LanceDB.)*
```bash
python scripts/run_pipeline.py
```

**To rebuild the entire database from zero:**
*(Wipes the `data/` folder, syncs the 50-year corpus, and ships the embedding job to a remote Kaggle Dual-T4 GPU cluster.)*
```bash
python scripts/run_pipeline.py --from-scratch
```

## Modular Tooling & Execution Scripts

If you want to run specific parts of the pipeline manually, the `scripts/` directory contains all executable entry points.

### 1. Ingestion & Pipeline
* `python scripts/run_ingestion.py`: Runs the multi-core parser over the raw XML and TXT corpus to generate canonical JSON artifacts.
* `python scripts/ingest_incremental_docs.py`: Processes only newly downloaded or XML-upgraded RFCs sequentially.
* `python scripts/sync_kaggle_vectors.py`: Zips the codebase, pushes it to an ephemeral Kaggle GPU container, runs the embedding model, and downloads the resulting Parquet shards.

### 2. Quality Assurance & Telemetry
* `python scripts/qa_auditor.py`: Asserts conservation of mass between raw text and chunk boundaries to catch data loss.
* `python scripts/analyze_chunk_length.py`: Outputs statistical distributions of chunk sizes.
* `python scripts/audit_lancedb.py`: Validates LanceDB table structures and verifies Tantivy BM25 FTS index health.
* `python scripts/audit_retrieval.py` / `audit_synthesis.py`: Fires a battery of semantic queries through the intent router to ensure the retrieval engine and LLM grounding constraints are functioning.

### 3. Server & Client
* `python scripts/serve.py`: Boots the Uvicorn/FastAPI server locally on port 8000.
* `python scripts/chat.py`: An interactive terminal UI communicating directly with the internal Python synthesis classes.
* `python scripts/chat_api.py`: An interactive terminal client that communicates with the system via HTTP POST requests, simulating the Vercel AI SDK streaming consumption.

## Repository Structure

```text
rfc_atlas/
├── data/                  # Local storage for all corpora, models, and databases
├── scripts/               # Entry points for CLI operations and deployment
├── src/rfc_atlas/
│   ├── api/               # FastAPI routing and lifecycle factories
│   ├── chunking/          # Recursive hierarchy-aware chunking algorithms
│   ├── graph/             # O(1) in-memory lineage mapping
│   ├── ingestion/         # Pebble multi-process orchestrator
│   ├── metadata/          # RFC index and temporal data schemas
│   ├── normalization/     # BCP-14 extraction and Pydantic canonical trees
│   ├── parsers/           # LXML modern parsers and regex legacy parsers
│   ├── retrieval/         # LanceDB dense/sparse search & Cross-Encoder reranking
│   ├── synthesis/         # LLM prompting and Gemini integration
│   └── vector_store/      # LanceDB PyArrow table construction and GPU Forge
└── tests/                 # Comprehensive pytest suite
```

## License
MIT License
