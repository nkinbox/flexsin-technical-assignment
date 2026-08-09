# Document-Aware AI Chatbot (RAG)

A chatbot that answers questions **strictly from uploaded documents**, with a page-level citation for every claim — and an explicit refusal when the documents don't contain the answer.

Design rationale: **[execution.md](execution.md)** · Deployment: **[DEPLOY.md](DEPLOY.md)**

---

## Grounding

Preventing hallucination is the central design goal, so it is enforced in **code** rather than requested in a prompt. Three independent layers:

**1. Relevance gate.** After retrieval, chunks beyond a cosine-distance threshold are discarded. If nothing survives, the API returns a refusal **without invoking the model at all** — a model that is never called cannot hallucinate, and unanswerable questions cost nothing.

The threshold is calibrated from measured distances rather than chosen arbitrarily:

| Question type | Top-result distance |
|---|---|
| Relevant to the documents | 0.23 – 0.67 |
| Unrelated to the documents | 0.95 – 1.08 |

The default of **0.80** sits near the centre of that gap. Both sides of the margin are asserted in the test suite, so a regression fails the build.

**2. Structured output.** The model responds into a JSON schema — `{answer, citations[], found}` — so citations arrive machine-readable and `found: false` is an unambiguous "not in the documents" signal, rather than something parsed out of prose.

**3. Citation validation.** Every citation the model returns is checked against what was actually retrieved. References to non-existent sources are removed and the answer flagged.

Citation text is always taken from the retrieved chunk, never from the model, so a quoted source cannot be reworded or invented.

---

## Stack

| Layer | Choice |
|---|---|
| Generation | Vertex AI `gemini-2.5-flash` — service-account auth, no API keys |
| Embeddings | `all-MiniLM-L6-v2`, local ONNX — no PyTorch, no per-chunk network call |
| Vector store | ChromaDB — persistent, cosine similarity |
| API | FastAPI — interactive docs at `/docs` |
| UI | Streamlit |
| Serving | Docker Compose + Caddy |

Embeddings run locally because that workload is cheap, high-volume, and well served by a small model; generation is hosted because model quality there determines whether the system hallucinates. The `Embedder` and `VertexLLM` interfaces are narrow, so either side can be swapped independently.

---

## Pipeline

**Ingestion**
```
upload → extract text (+page numbers) → chunk (+metadata) → embed → index
```

**Query**
```
question → condense if follow-up → embed → top-k search → RELEVANCE GATE
                                                              ↓
                                        (nothing relevant) → refuse, no model call
                                                              ↓
                                        numbered context → Gemini (JSON schema)
                                                              ↓
                                        validate citations → answer + sources
```

### Chunking

Recursive boundary-aware splitting: separators are tried strongest-first — paragraph → line → sentence → word — and the strongest boundary that fits the size budget is used. A fixed-width split severs sentences mid-clause and produces chunks that embed poorly.

| Parameter | Value | Reasoning |
|---|---|---|
| Chunk size | 1000 chars | Holds a complete idea while keeping each retrieved chunk mostly signal |
| Overlap | 150 chars (15%) | A fact spanning a boundary survives intact in at least one chunk |
| Minimum chunk | 50 chars | Page numbers and orphan headers would otherwise occupy a top-k slot |

Chunks never span pages, which keeps every citation's page number verifiable. Each carries `doc_id`, `filename`, `page_number`, `chunk_index` and `char_start` — the provenance that makes citation possible.

### Conversational memory

Follow-up questions are rewritten into standalone form before retrieval. *"What about its pricing?"* embeds to near-nothing on its own; resolved against recent history it becomes *"What is the pricing of the Enterprise plan?"*, and that is what gets embedded. The rewrite runs only when a question looks context-dependent, so self-contained questions skip the round-trip. The UI displays the rewritten query when it fires.

---

## Running locally

**Prerequisites:** Docker, a GCP project, the `gcloud` CLI.

```bash
gcloud auth application-default login
gcloud services enable aiplatform.googleapis.com

cp .env.example .env          # set GCP_PROJECT
python scripts/verify_vertex.py

docker compose up --build
```

UI at http://localhost:8501 · API docs at http://localhost:8000/docs

> **Windows:** set `GCLOUD_CONFIG_DIR` in `.env` to `C:/Users/<you>/AppData/Roaming/gcloud`. `gcloud` stores credentials under `%APPDATA%` there, so the default mount path would be empty.

<details>
<summary>Without Docker</summary>

```bash
python -m venv .venv && .venv/Scripts/activate    # Windows
# source .venv/bin/activate                        # macOS/Linux
pip install -r requirements.txt

uvicorn app.main:app --reload                      # terminal 1
streamlit run ui/streamlit_app.py                  # terminal 2
```
</details>

---

## Tests

```bash
pytest tests/ -v                  # 55 tests
pytest tests/ -m "not slow"       # skip the ONNX-loading integration tests
```

The model client is mocked throughout, so the suite runs **without credentials, without network access, and at no cost**.

| Suite | Covers |
|---|---|
| `test_rag_grounding.py` | Unanswerable questions refuse **and make zero model calls** |
| `test_citations.py` | Invented citation numbers are stripped and flagged |
| `test_chunker.py` | Overlap, boundary preference, metadata, page isolation |
| `test_extract.py` | PDF/DOCX/TXT handling, table extraction, rejected types |
| `test_memory.py` | History retention and follow-up condensing |
| `test_integration.py` | Real embeddings and vector store; asserts the threshold margin |

---

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/upload` | Ingest a document (multipart) |
| `POST` | `/chat` | Ask a question |
| `GET` | `/documents` | List indexed documents |
| `DELETE` | `/documents/{id}` | Remove a document |
| `DELETE` | `/sessions/{id}` | Clear chat history |
| `GET` | `/health` | Store and Vertex AI status |

`POST /chat` returns:

```json
{
  "answer": "Q3 revenue was $4.2 million.",
  "found": true,
  "verified": true,
  "gated": false,
  "search_query": "What was total revenue in Q3 2024?",
  "citations": [
    {"number": 1, "filename": "report.pdf", "page_number": 7, "text": "..."}
  ]
}
```

`gated: true` means the relevance gate fired and no model call was made. `verified: false` means an invalid citation was removed.

---

## Configuration

All settings live in `.env` (see `.env.example`) and are read through `app/config.py`.

| Variable | Default | Notes |
|---|---|---|
| `GCP_PROJECT` | — | Required |
| `GCP_LOCATION` | `us-central1` | Keep the VM in the same region |
| `LLM_MODEL` | `gemini-2.5-flash` | |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | |
| `TOP_K` | `5` | Chunks retrieved per question |
| `RELEVANCE_THRESHOLD` | `0.80` | The gate; lower is stricter |
| `MAX_UPLOAD_MB` | `20` | Bounds work per request |

---

## Scope

Supported input formats are **PDF, DOCX, TXT and MD**. Image input is not supported in this build — effort was concentrated on the retrieval and grounding core rather than spread across every input format.

Adding it is contained rather than architectural: `gemini-2.5-flash` is multimodal, so images would be sent to the model once at ingest and the extracted text fed into the existing chunking pipeline — roughly 30 lines in `app/extract.py`, with nothing downstream changing. Unsupported uploads return a clear error naming the accepted types, and a scanned PDF with no text layer is rejected explicitly rather than silently indexed as empty.

Other deliberate trade-offs:

| Decision | Accepted cost |
|---|---|
| In-memory chat history | Lost on restart; `app/memory.py` is the only module that would change |
| Embedded ChromaDB | Single-node; a client/server vector DB is the swap for scale |
| Hosted generation | Requires network access; the `VertexLLM` interface keeps a local backend contained |

---

## Layout

```
app/
  config.py      settings and calibration
  extract.py     PDF/DOCX/TXT → pages
  chunker.py     recursive boundary-aware splitting
  embedder.py    local ONNX embeddings
  store.py       ChromaDB wrapper
  llm.py         Vertex AI client
  rag.py         retrieval gate, generation, citation validation
  memory.py      history and follow-up condensing
  main.py        FastAPI application
ui/              Streamlit client
tests/           55 tests
scripts/         Vertex AI preflight check
docker/          API and UI images
```
