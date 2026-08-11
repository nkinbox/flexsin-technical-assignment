# Execution Plan — Document-Aware AI Chatbot (RAG)

**Author:** Nishant
**Assessment:** Build a Document-Aware AI Chatbot using RAG

This document explains *what* was built, *how* the pipeline works, and — most importantly — *why* each design decision was made. The code is intended to be read alongside it.

---

## 1. Problem restated

An LLM has no access to private documents and will confidently invent answers about them. Retrieval-Augmented Generation fixes this by retrieving relevant passages from a user's own documents and constraining the model to answer from those passages only.

The assessment's evaluation criteria are the design brief:

| Criterion | How this build addresses it |
|---|---|
| RAG pipeline understanding | Each stage is its own module (`extract` → `chunker` → `embedder` → `store` → `retrieval` → `rag`) with no hidden framework magic |
| Accuracy of responses | Hybrid retrieval — dense vectors *and* BM25 — over overlapping, boundary-aware chunks |
| **Grounded answers (no hallucination)** | Enforced in **code**, not just prompted — see §6 |
| Proper chunking strategy | Recursive boundary-aware splitting with justified size and overlap — see §4 |

Bonus features (chat memory, multi-document querying, source-citation highlighting) are all implemented.

---

## 2. Architecture

```
                    ┌──────── Caddy (80/443) ─────────────┐
                    │   /      → ui:8501                  │
                    │   /api/* → api:8000                 │
                    └──────────────────┬──────────────────┘
                            ┌──────────┴──────────┐
                         ui (Streamlit)      api (FastAPI)
                                                  │
                                    ┌─────────────┼─────────────┐
                                ChromaDB      BM25 index     Vertex AI
                                (volume)     (in-memory)   embeddings +
                                                            generation
```

**Ingestion**
```
upload → extract text (+page numbers) → chunk (+metadata) → embed → index
              ↑
      images and scanned PDFs are read by the vision model here
```

**Query**
```
question → condense if follow-up ──┬── dense vector search ──┐
                                   │                         ├→ RRF fusion → gate
                                   └── BM25 lexical search ──┘        │
                                                                      ↓
                                    (nothing relevant) → refuse, no model call
                                                                      ↓
                                    numbered context → Gemini (JSON schema)
                                                                      ↓
                                    validate citation numbers → answer + sources
```

---

## 3. Technology choices and rationale

| Layer | Choice | Why |
|---|---|---|
| Generation | **Vertex AI `gemini-2.5-flash`** | Grounded refusal — knowing when *not* to answer — is the hardest behaviour to get right and exactly what is being graded. Also multimodal, which is what makes image ingestion a few lines rather than a second subsystem. |
| Embeddings | **Vertex AI `gemini-embedding-001`** | Retrieval quality is the ceiling on answer quality: nothing downstream can recover a passage that was never retrieved. Supports task-typed embeddings (§5). |
| Lexical search | **Okapi BM25, implemented in-repo** | ~80 lines, no dependency, and it covers precisely what embeddings are worst at (§5). |
| Vector DB | **ChromaDB (persistent, embedded)** | Real vector database semantics — cosine, metadata filtering — with no extra container and no server to operate. |
| API | **FastAPI** | Async, typed, automatic OpenAPI docs at `/docs` — useful for demonstrating the pipeline without the UI. |
| UI | **Streamlit** | Fastest path to a chat interface with file upload; keeps attention on the pipeline rather than frontend plumbing. |
| Serving | **Docker Compose + Caddy** | One command locally and in the cloud. |

### Embeddings are pluggable

`app/embedder.py` defines an `Embedder` interface with two implementations:

- **`VertexEmbedder`** (default) — better recall, and the reason the system can answer questions phrased quite differently from the source text.
- **`LocalEmbedder`** — in-process ONNX MiniLM. No credentials, no cost, no network. This is what the test suite runs on, so the whole suite works offline and for free.

Switching providers changes both the vector space and the dimensionality (768 vs 384), and vectors from different models are not comparable. `COLLECTION_NAME` therefore encodes the embedding configuration, so a provider switch starts a clean collection rather than silently querying one model's index with another model's vectors.

---

## 4. Chunking strategy

Chunking is explicitly graded, so the reasoning matters as much as the result.

**Approach: recursive boundary-aware splitting.** Separators are tried in descending order of semantic strength:

```
"\n\n" (paragraph)  →  "\n" (line)  →  ". " (sentence)  →  " " (word)
```

The splitter uses the strongest boundary that still fits the size budget. A fixed-width split severs sentences mid-clause and produces chunks that embed poorly; splitting on paragraphs keeps each chunk semantically self-contained.

**Parameters and their justification:**

| Parameter | Value | Reasoning |
|---|---|---|
| Chunk size | **2000 chars** (~500 tokens) | Sized so a complete idea stays intact *with the context that qualifies it*. Small chunks are the more tempting default — they raise precision on paper — but they fragment a single fact across several records, and retrieval then returns a piece of the answer rather than the answer. A policy sentence separated from the paragraph stating its exception is worse than useless. |
| Overlap | **300 chars (15%)** | A fact spanning a boundary would otherwise be split across two chunks and complete in neither. Overlap guarantees it survives intact in at least one. |
| Minimum chunk | **40 chars** | Fragments below this — page numbers, isolated headers, footer artifacts — carry no retrievable meaning and would occupy a top-k slot a real chunk could use. |
| Top-k | **8** | Generous on purpose. With hybrid retrieval the extra candidates cost little, and the dominant failure mode in practice is not a distracted model but an answer that never reached the context. |

Chunks never span pages, which keeps `page_number` unambiguous and every citation verifiable.

**Metadata carried on every chunk:** `doc_id`, `filename`, `page_number`, `chunk_index`, `char_start`. This is what makes real citation possible — without page-level provenance the system can only cite a filename, which is not a citation a user can check.

---

## 5. Hybrid retrieval

Dense and lexical retrieval fail in opposite directions, and using only one leaves a whole class of questions unanswerable.

| | Dense (embeddings) | Lexical (BM25) |
|---|---|---|
| Good at | Paraphrase, synonyms, conceptual similarity | Exact tokens, identifiers, rare words |
| Bad at | Opaque strings with no semantics to compress | Anything not sharing vocabulary with the source |
| Example it wins | *"time off"* → "vacation policy" | *"QR-88214-ZX"* → the filing that mentions it |

An embedding represents meaning, which is exactly the wrong tool for a token that has none to represent — an invoice number, a SKU, a version string, a surname. Those are precisely the terms a user quotes verbatim when asking a simple factual question about their own document, so lexical matching is not a fallback but a complement.

**Fusion: Reciprocal Rank Fusion.** Each retriever contributes `1/(k + rank)` for every chunk it ranks, with `k = 60`.

RRF combines *ranks*, not scores — which is what makes it usable here at all. A cosine distance and a BM25 score share no scale, and normalising them would need corpus statistics that shift on every upload. Ranks are directly comparable and need no tuning. A chunk both retrievers rank accumulates both contributions and rises to the top, which is the desired behaviour: agreement between two independent signals is the strongest evidence available.

**Task-typed embeddings.** Vertex embeds passages as `RETRIEVAL_DOCUMENT` and questions as `RETRIEVAL_QUERY`. A question and the passage answering it are different kinds of text — encoding them identically pushes short questions towards other short questions instead of towards the long passages that answer them.

---

## 6. Grounding: how hallucination is actually prevented

The criterion the whole design is oriented around. Three independent layers.

### Layer 1 — The relevance gate

After retrieval, chunks clearing neither relevance bar are discarded. **If none survive, the API returns a refusal without calling the model at all.**

*A model that is never invoked cannot hallucinate.* Most RAG implementations pass whatever came back from the vector store — however irrelevant, since a vector store always returns its nearest neighbours — and then ask the model nicely not to invent. Here the guarantee is structural. It also means unanswerable questions cost nothing in tokens.

A chunk passes on **either** kind of evidence:

- **semantic** — cosine distance within `RELEVANCE_THRESHOLD`, or
- **lexical** — a BM25 score clearing both a small absolute floor and a share of the best lexical score for that query

The disjunction is deliberate. Requiring both would discard exactly the cases hybrid retrieval exists to catch. The lexical bar is *relative* because BM25 scores scale with corpus size and IDF — a fixed threshold tuned on a large corpus rejects everything in a small one.

**The threshold is deliberately permissive.** Wrongly refusing a question the documents *do* answer is the worse failure: it makes the system look broken, while a marginal chunk reaching the model is caught by layers 2 and 3. The gate's job is to exclude the clearly unrelated, not to adjudicate borderline cases.

`scripts/calibrate_threshold.py` measures both populations against your own corpus and recommends a value, because the right threshold depends on the embedding model and the documents.

### Layer 2 — Structured output with an explicit `found` flag

The model is constrained to a JSON schema:

```json
{"answer": "string", "citations": [1, 3], "found": true}
```

Citations arrive machine-readable — no regex, no formatting drift — and `found: false` is an unambiguous "the sources don't contain this" signal rather than a magic token to string-match. When `found` is false the response is normalised into the same refusal shape the gate produces, so callers see one consistent behaviour.

### Layer 3 — Citation validation

Every citation number the model returns is checked against the chunks actually retrieved. A number outside that set is a fabrication: it is dropped and the response flagged `verified: false`.

This closes the last gap — a model can invent a citation index even when its prose is faithful. Citation *text* is always taken from the retrieved chunk, never from the model, so a quoted source cannot be reworded or invented.

---

## 7. Input formats

| Format | How it is read |
|---|---|
| PDF (digital) | `pypdf`, per page, preserving real page numbers |
| PDF (scanned) | No usable text layer → re-read with the vision model, page markers preserved |
| DOCX | `python-docx` — paragraphs **and table cells** |
| TXT / MD | Direct, with an encoding fallback |
| Images | PNG, JPEG, WEBP, GIF, BMP, TIFF — read by the vision model |

**Images are read by the multimodal model rather than a separate OCR engine.** Beyond avoiding a system dependency, this handles what classical OCR does badly — charts, diagrams, screenshots, handwriting — because the model describes structure as well as transcribing glyphs. A chart becomes searchable prose rather than a row of stray axis labels.

The transcription prompt explicitly forbids commentary. Anything the model adds at this stage becomes indexed "source" material that a later answer could cite as though it came from the document, which would quietly undermine the grounding guarantees above.

DOCX table extraction is similarly deliberate: a paragraph-only read silently drops tables, which in business documents hold exactly the facts people ask about — prices, dates, specifications.

---

## 8. Bonus features

**Chat memory with follow-up condensing.** History is kept per `session_id`. The part that matters for RAG is *retrieval-aware* memory: a follow-up like *"what about its pricing?"* embeds terribly on its own — "its" carries the whole meaning and the vector is close to meaningless. Before retrieving, such questions are rewritten into standalone form using recent history, and *that* is what gets embedded. The rewrite is gated to questions that look context-dependent, so self-contained questions skip the round-trip.

**Multi-document querying.** All documents are searched by default; the UI can scope to a subset. Implemented as a store-level metadata filter applied at search time — not post-filtering — so top-k stays meaningful within the selected scope. The BM25 index is scoped to match.

**Source citation highlighting.** Each answer renders an expandable panel per cited source: filename, page number, and the chunk text.

---

## 9. Trade-offs

| Decision | Accepted cost |
|---|---|
| In-memory BM25 index | Rebuilt when the corpus changes and held in process. Fine at POC scale; a larger deployment would push lexical search into the datastore rather than keeping a second copy of the corpus in memory. |
| In-memory chat history | Lost on restart. A single-user POC does not need Redis; `app/memory.py` is the only module that would change. |
| Embedded ChromaDB | Single-node. Correct here; a client/server vector DB is the swap for scale. |
| Hosted embeddings | A network call per batch at ingest, and cost per document. Bought deliberately: retrieval quality caps everything downstream. `EMBEDDING_PROVIDER=local` reverses it. |
| Permissive gate | More marginal chunks reach the model than a strict threshold would allow, leaning more weight on layers 2 and 3. Chosen because a false refusal is the more damaging failure. |

---

## 10. Running it

Full instructions in `README.md` (local) and `DEPLOY.md` (GCP). Briefly:

```bash
gcloud auth application-default login
cp .env.example .env            # set GCP_PROJECT
python scripts/verify_vertex.py # credentials, model, JSON, embeddings, vision
docker compose up --build

pytest tests/ -v                # 98 tests, no credentials or network needed
```

---

## 11. Verification checklist

The behaviours worth checking, in the order they matter:

1. **A question the documents answer gets answered** — including one phrased differently from the source text, and one quoting an exact identifier. These exercise the two halves of hybrid retrieval.
2. **An unanswerable question is refused, and no model call is made.**
3. Every answer's citation resolves to the correct page.
4. An uploaded image is transcribed and becomes answerable.
5. A pronoun follow-up produces a sensible standalone query.
6. A query scoped to one document returns no other document.
7. Restarting the stack preserves indexed documents.
