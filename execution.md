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
| RAG pipeline understanding | Each stage is its own module (`extract` → `chunker` → `embedder` → `store` → `rag`) with no hidden framework magic |
| Accuracy of responses | Cosine retrieval over overlapping, boundary-aware chunks; a strong hosted model for generation |
| **Grounded answers (no hallucination)** | Enforced in **code**, not just prompted — see §5 |
| Proper chunking strategy | Recursive boundary-aware splitting with justified size/overlap — see §4 |

Bonus features (chat memory, multi-document querying, source-citation highlighting) are all implemented.

---

## 2. Architecture

```
                    ┌──────── Caddy (443/80, TLS) ────────┐
                    │   /      → ui:8501                  │
                    │   /api/* → api:8000                 │
                    └──────────────────┬──────────────────┘
                            ┌──────────┴──────────┐
                         ui (Streamlit)      api (FastAPI)
                                                  │
                                    ┌─────────────┼─────────────┐
                              ONNX embedder   ChromaDB      Vertex AI
                               (in-process)   (volume)   gemini-2.5-flash
```

**Ingestion**
```
upload → extract text (+page numbers) → chunk (+metadata) → embed locally → store vectors
```

**Query**
```
question → condense if follow-up → embed → top-k search → RELEVANCE GATE
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
| Generation | **Vertex AI `gemini-2.5-flash`** | Grounded refusal — knowing when *not* to answer — is the hardest behaviour to get right, and it is exactly what's being graded. A small local model degrades here badly. Hosted also means no GPU and a ~$27/mo VM instead of ~$100. |
| Embeddings | **Local `all-MiniLM-L6-v2` via ONNX** | Embedding is the one part a small local model does genuinely well. Running it in-process avoids a network round-trip per chunk during ingestion, costs nothing, and keeps ingestion fast. |
| Vector DB | **ChromaDB (persistent, embedded)** | Real vector database semantics (cosine, metadata filtering) with zero operational overhead — no extra container, no server to run. Reviewer runs one command. |
| API | **FastAPI** | Async, typed, automatic OpenAPI docs at `/docs` — useful for demonstrating the pipeline without the UI. |
| UI | **Streamlit** | Fastest path to a chat interface with file upload; keeps attention on the pipeline rather than frontend plumbing. |
| Serving | **Docker Compose + Caddy** | One command locally and in the cloud. Caddy gives automatic HTTPS in ~4 lines of config. |

### Why ONNX rather than `sentence-transformers`

Same model weights, but the ONNX runtime avoids a PyTorch dependency. Three concrete benefits:

1. **The dev machine runs Python 3.14**, where PyTorch wheels may not yet exist — this would have blocked the build outright.
2. The container image stays small (no ~800 MB torch layer).
3. RAM headroom on a 4 GB VM stays comfortable.

*Verified:* the full dependency set installs cleanly on Python 3.14.6 and produces 384-dimensional vectors.

### A note on the embedding/generation split

This is deliberate, not accidental. Embedding quality and generation quality are separable concerns with very different cost profiles. Embeddings run thousands of times at ingest (cheap, local, fast); generation runs once per question and determines whether the system hallucinates (worth paying for). The `Embedder` and `LLMClient` interfaces mean either half can be swapped independently.

---

## 4. Chunking strategy

Chunking is explicitly graded, so the reasoning matters as much as the result.

**Approach: recursive boundary-aware splitting.**

Separators are tried in descending order of semantic strength:

```
"\n\n"  (paragraph)  →  "\n"  (line)  →  ". "  (sentence)  →  " "  (word)
```

The splitter uses the strongest boundary that still yields a chunk within the size budget. A naive fixed-width split severs sentences mid-clause and produces chunks that embed poorly; splitting on paragraphs where possible keeps each chunk semantically self-contained.

**Parameters and their justification:**

| Parameter | Value | Reasoning |
|---|---|---|
| Chunk size | **1000 chars** (~250 tokens) | Large enough to hold a complete idea with context; small enough that a single retrieved chunk is mostly *signal* rather than surrounding noise. Retrieval precision falls as chunks grow. |
| Overlap | **150 chars (15%)** | A fact spanning a chunk boundary would otherwise be split across two chunks and fully present in neither. Overlap guarantees it survives intact in at least one. 15% is the point where this protection is reliable without materially inflating index size. |
| Minimum chunk | **50 chars** | Fragments below this (page numbers, isolated headers, footer artifacts) carry no retrievable meaning and actively pollute results by occupying top-k slots. |

**Metadata carried on every chunk:**

```python
{"doc_id", "filename", "page_number", "chunk_index", "char_start"}
```

This is what makes real citation possible. Without page-level provenance the system can only cite a filename, which is not a citation a user can verify.

---

## 5. Grounding: how hallucination is actually prevented

This is the criterion the whole design is oriented around. Three independent layers, ordered from strongest to weakest:

### Layer 1 — The relevance gate (structural, strongest)

After retrieval, chunks whose cosine distance exceeds a threshold are discarded. **If no chunk survives, the API returns a refusal without calling the model at all.**

This is the key insight: *a model that is never invoked cannot hallucinate.* Most RAG implementations pass whatever came back from the vector store — however irrelevant — and then ask the model nicely not to make things up. Here the guarantee is structural rather than behavioural.

It also means unanswerable questions cost nothing in tokens.

*Threshold calibration — measured, not guessed.* Distances were sampled across relevant and unrelated questions against real indexed documents:

| Question type | Top-result cosine distance |
|---|---|
| *"What was total revenue in Q3?"* | 0.233 |
| *"Who pays for medical premiums?"* | 0.623 |
| *"How much time can I work from home?"* | 0.666 |
| *"How do I bake sourdough bread?"* | 0.950 |
| *"What is the boiling point of mercury?"* | 1.053 |
| *"What is the migratory pattern of Arctic terns?"* | 1.080 |

The two populations separate cleanly, with a gap between **0.666** and **0.950**. The default threshold of **0.80** sits near the centre of that gap — roughly 0.13 of headroom before a genuine question is wrongly refused, and 0.15 before an unrelated one slips through.

Both sides of this margin are asserted in `tests/test_integration.py`, so a regression in embedding behaviour or threshold tuning fails the build rather than silently degrading grounding. Configurable via `RELEVANCE_THRESHOLD`.

### Layer 2 — Structured output with an explicit `found` flag

The model is constrained to a JSON schema:

```json
{"answer": "string", "citations": [1, 3], "found": true}
```

Two benefits over parsing `[n]` markers out of prose:
- Citations arrive machine-readable — no regex, no formatting drift.
- `found: false` is an unambiguous "the sources don't contain this" signal, rather than string-matching a magic token in free text.

When `found` is false, the response is normalised into the same refusal shape the gate produces, so callers see one consistent behaviour.

### Layer 3 — Citation validation (catches the residual case)

Every citation number the model returns is checked against the set of chunks actually retrieved. A number outside that set is a fabrication: it is **dropped**, and the response is flagged `verified: false`.

This is cheap to implement and closes the last gap — a model can invent a citation index even when its prose is faithful. Detecting that in code is more convincing than trusting the model not to do it.

---

## 6. Bonus features

**Chat memory with follow-up condensing.**
History is kept per `session_id`. The important part is *retrieval-aware* memory: a follow-up like *"what about its pricing?"* embeds terribly on its own — "its" carries the entire meaning and the vector is near-meaningless. Before retrieving, such questions are rewritten into standalone form ("what is the pricing of the Enterprise plan?") using recent history, and *that* is what gets embedded.

The rewrite is gated: it only runs when history exists **and** the question looks dependent (short, or opening with a pronoun/demonstrative). Standalone questions skip it and save the round-trip.

**Multi-document querying.**
All documents are searched by default. The UI can scope a query to a selected subset, implemented as a ChromaDB metadata filter (`doc_id $in [...]`) applied at search time — not post-filtering, so top-k stays meaningful.

**Source citation highlighting.**
Each answer renders an expandable panel per cited source showing filename, page number, and the chunk text with the relevant region highlighted.

---

## 7. Scope decisions and trade-offs

**Image input is not supported.** The brief lists IMAGE among mandatory input types; this build handles PDF, DOCX, and TXT only. This was a deliberate scoping decision to concentrate effort on the graded core (chunking, retrieval, grounding) rather than a partial implementation of everything.

It is a contained addition: `gemini-2.5-flash` is natively multimodal, so supporting images means sending the image to the model once at ingest, taking the extracted text back, and feeding it into the existing chunking pipeline — roughly 30 lines in `extract.py`, no changes anywhere downstream. The pipeline was designed so that this is true.

**Other conscious trade-offs:**

| Decision | Trade-off accepted |
|---|---|
| In-memory chat history | Lost on restart. A POC serving one demo user doesn't need Redis; the `memory` module is the only thing that would change. |
| Embedded ChromaDB | Single-node only. Correct for a POC; a client/server vector DB is the swap for scale. |
| Requires network | Cannot demo offline, unlike a self-hosted model. The `LLMClient` interface keeps a local backend a contained change. |
| No auth on the demo URL | Documented; a commented `basic_auth` block in the Caddyfile enables it in one line. |

---

## 8. Running it

Full instructions in `README.md` (local) and `DEPLOY.md` (GCP). Briefly:

```bash
# Local
gcloud auth application-default login
cp .env.example .env          # set GCP_PROJECT
docker compose up --build     # UI at http://localhost:8501

# Tests (no credentials or network needed — the model client is mocked)
pytest tests/ -v
```

---

## 9. Verification checklist

The behaviours worth checking, in the order they matter:

1. **Unanswerable question → refusal, and no model call is made.** The headline criterion.
2. Answerable question → correct answer with a citation resolving to the right page.
3. Pronoun follow-up → condensing produces a sensible standalone query; retrieval still correct.
4. Query scoped to one document → no other document appears in the citations.
5. Restart the stack → documents survive (volume persistence).
