# Document-Aware AI Chatbot (RAG)

A chatbot that answers questions **strictly from uploaded documents**, with a page-level citation for every claim — and an explicit refusal when the documents don't contain the answer.

Design rationale: **[execution.md](execution.md)** · Deployment: **[DEPLOY.md](DEPLOY.md)**

---

## Grounding

Preventing hallucination is the central design goal, so it is enforced in **code** rather than requested in a prompt. Three independent layers:

**1. Relevance gate.** After retrieval, chunks clearing neither relevance bar are discarded. If none survive, the API returns a refusal **without invoking the model at all** — a model that is never called cannot hallucinate, and unanswerable questions cost nothing.

A chunk passes on **either** semantic evidence (cosine distance within the threshold) or lexical evidence (a BM25 score clearing both an absolute floor and a share of the query's best lexical score). The disjunction is deliberate: requiring both would discard exactly the cases hybrid retrieval exists to catch.

The threshold is deliberately permissive. Wrongly refusing a question the documents *do* answer is the worse failure — it makes the system look broken — while a marginal chunk reaching the model is caught by the two layers behind it. `scripts/calibrate_threshold.py` measures both populations against your own corpus and recommends a value.

**2. Structured output.** The model responds into a JSON schema — `{answer, citations[], found}` — so citations arrive machine-readable and `found: false` is an unambiguous "not in the documents" signal, rather than something parsed out of prose.

**3. Citation validation.** Every citation the model returns is checked against what was actually retrieved. References to non-existent sources are removed and the answer flagged.

Citation text is always taken from the retrieved chunk, never from the model, so a quoted source cannot be reworded or invented.

---

## Stack

| Layer | Choice |
|---|---|
| Generation | Vertex AI `gemini-2.5-flash` — service-account auth, no API keys |
| Embeddings | Vertex AI `gemini-embedding-001`, task-typed; local ONNX MiniLM as an offline fallback |
| Lexical search | Okapi BM25, implemented in-repo — no dependency |
| Vector store | ChromaDB — persistent, cosine similarity |
| API | FastAPI — interactive docs at `/docs` |
| UI | Streamlit |
| Serving | Docker Compose + Caddy |

Retrieval quality is the ceiling on answer quality — nothing downstream recovers a passage that was never retrieved — so embeddings are hosted by default. `EMBEDDING_PROVIDER=local` switches to in-process ONNX embeddings with no credentials, no cost and no network, which is what the test suite runs on.

---

## Pipeline

**Ingestion**
```
upload → extract text (+page numbers) → chunk (+metadata) → embed → index
```

**Query**
```
question → condense if follow-up ──┬── dense vector search ──┐
                                   │                         ├→ RRF fusion → GATE
                                   └── BM25 lexical search ──┘        │
                                                                      ↓
                                    (nothing relevant) → refuse, no model call
                                                                      ↓
                                    numbered context → Gemini (JSON schema)
                                                                      ↓
                                    validate citations → answer + sources
```

### Hybrid retrieval

Dense and lexical search fail in opposite directions, so both run and their rankings are fused.

| | Dense (embeddings) | Lexical (BM25) |
|---|---|---|
| Good at | Paraphrase, synonyms, concepts | Exact tokens, identifiers, rare words |
| Example it wins | *"time off"* → "vacation policy" | *"QR-88214-ZX"* → the filing mentioning it |

An embedding represents meaning, which is the wrong tool for a token that has none — an invoice number, a SKU, a version string. Those are exactly what users quote when asking simple factual questions about their own documents.

Fusion is **Reciprocal Rank Fusion**: each retriever contributes `1/(60 + rank)`. RRF combines *ranks* rather than scores, which matters because a cosine distance and a BM25 score share no scale and normalising them would need corpus statistics that shift on every upload. A chunk both retrievers rank accumulates both contributions and rises to the top — agreement between independent signals being the strongest evidence available.

### Chunking

Recursive boundary-aware splitting: separators are tried strongest-first — paragraph → line → sentence → word — and the strongest boundary that fits the size budget is used. A fixed-width split severs sentences mid-clause and produces chunks that embed poorly.

| Parameter | Value | Reasoning |
|---|---|---|
| Chunk size | 2000 chars | Keeps a fact together with the context that qualifies it. Smaller chunks raise precision on paper but fragment one idea across several records, so retrieval returns a piece of the answer rather than the answer |
| Overlap | 300 chars (15%) | A fact spanning a boundary survives intact in at least one chunk |
| Minimum chunk | 40 chars | Page numbers and orphan headers would otherwise occupy a top-k slot |
| Top-k | 8 | Generous on purpose — the common failure is an answer that never reached the context, not a distracted model |

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
pytest tests/ -v                  # 98 tests
pytest tests/ -m "not slow"       # skip the ONNX-loading integration tests
```

The model client is mocked throughout, so the suite runs **without credentials, without network access, and at no cost**.

| Suite | Covers |
|---|---|
| `test_rag_grounding.py` | Unanswerable questions refuse **and make zero model calls** |
| `test_citations.py` | Invented citation numbers are stripped and flagged |
| `test_chunker.py` | Overlap, boundary preference, metadata, page isolation |
| `test_retrieval.py` | RRF fusion and the gate's two-signal admission |
| `test_bm25.py` | Lexical ranking, IDF, length normalisation |
| `test_extract.py` | PDF/DOCX/TXT/image handling, scanned-PDF fallback, table extraction |
| `test_memory.py` | History retention and follow-up condensing |
| `test_integration.py` | Real embeddings, real BM25, real fusion — including that an exact identifier is retrievable and off-topic questions are still gated |

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
| `LLM_MODEL` | `gemini-2.5-flash` | Also used for image and scanned-PDF reading |
| `EMBEDDING_PROVIDER` | `vertex` | `local` for offline in-process embeddings |
| `VERTEX_EMBED_MODEL` | `gemini-embedding-001` | |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `2000` / `300` | |
| `TOP_K` | `8` | Chunks passed to the model |
| `RELEVANCE_THRESHOLD` | `1.0` | The semantic bar; lower is stricter |
| `HYBRID_SEARCH` | `true` | Set `false` for dense-only retrieval |
| `MAX_UPLOAD_MB` | `20` | Bounds work per request |

---

## Supported formats

| Format | How it is read |
|---|---|
| PDF (digital) | `pypdf`, per page, preserving real page numbers |
| PDF (scanned) | No usable text layer → re-read with the vision model |
| DOCX | `python-docx` — paragraphs **and table cells** |
| TXT / MD | Direct, with an encoding fallback |
| Images | PNG, JPEG, WEBP, GIF, BMP, TIFF — read by the vision model at upload |

Images are read by the multimodal model rather than a separate OCR engine. That avoids a system dependency and handles what classical OCR does badly — charts, diagrams, screenshots, handwriting — because the model describes structure as well as transcribing glyphs. The transcription prompt forbids commentary: anything the model added would become indexed "source" material a later answer could cite as though it came from the document.

---

## Trade-offs

| Decision | Accepted cost |
|---|---|
| In-memory BM25 index | Rebuilt when the corpus changes and held in process. Fine at POC scale; a larger deployment would push lexical search into the datastore |
| In-memory chat history | Lost on restart; `app/memory.py` is the only module that would change |
| Embedded ChromaDB | Single-node; a client/server vector DB is the swap for scale |
| Hosted embeddings | A network call per batch at ingest, and cost per document — bought deliberately, since retrieval quality caps everything downstream |
| Permissive gate | More marginal chunks reach the model, leaning more weight on the `found` flag and citation validation. Chosen because a false refusal is the more damaging failure |

---

## Layout

```
app/
  config.py      settings and calibration
  extract.py     PDF/DOCX/TXT/images → pages
  chunker.py     recursive boundary-aware splitting
  embedder.py    Vertex and local embedding backends
  bm25.py        Okapi BM25 lexical index
  store.py       ChromaDB wrapper
  retrieval.py   hybrid search, RRF fusion, the relevance gate
  llm.py         Vertex AI client (generation + vision)
  rag.py         prompt assembly, generation, citation validation
  memory.py      history and follow-up condensing
  main.py        FastAPI application
ui/              Streamlit client
tests/           98 tests
scripts/         Vertex preflight, threshold calibration
docker/          API and UI images
```
