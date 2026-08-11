"""FastAPI application — the RAG pipeline over HTTP.

Endpoints:
    POST   /upload            ingest a document
    POST   /chat              ask a question
    GET    /documents         list indexed documents
    DELETE /documents/{id}    remove a document
    DELETE /sessions/{id}     clear a chat session
    GET    /health            readiness and dependency status

Interactive docs at /docs — useful for demonstrating the pipeline without the UI.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app import memory
from app.chunker import chunk_pages
from app.config import MAX_UPLOAD_BYTES, MAX_UPLOAD_MB, SUPPORTED_EXTENSIONS
from app.extract import ExtractionError, UnsupportedFileError, extract
from app.llm import LLMError, get_llm
from app.rag import answer_question
from app.store import get_store

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("rag")

app = FastAPI(
    title="Document-Aware RAG Chatbot",
    description=(
        "Answers questions strictly from uploaded documents, using hybrid "
        "retrieval (dense vectors + BM25). Grounding is enforced in code: if "
        "nothing relevant is retrieved, the model is never called."
    ),
    version="1.0.0",
)


# --- Schemas -----------------------------------------------------------------


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="default", max_length=128)
    # Optional scope for multi-document querying. None/empty searches everything.
    doc_ids: list[str] | None = None


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunks: int


# --- Endpoints ---------------------------------------------------------------


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    """Ingest a document: extract, chunk, embed, index."""
    data = await file.read()

    # This can be publicly reachable, so bound the work a single request can
    # cause rather than trusting callers.
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_MB} MB limit.",
        )

    if not data:
        raise HTTPException(status_code=400, detail="File is empty.")

    try:
        pages = extract(file.filename or "unnamed", data)
    except UnsupportedFileError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ExtractionError as exc:
        # A supported file we could not read -- an unreadable image, or a
        # scanned PDF when the vision model was unreachable.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Extraction failed for %s", file.filename)
        raise HTTPException(
            status_code=400, detail=f"Could not read the document: {exc}"
        ) from exc

    if not pages:
        raise HTTPException(
            status_code=422,
            detail=(
                "No text could be extracted from this file. If it is a scanned "
                "document, check that Vertex AI is reachable — scans are read "
                "with the vision model."
            ),
        )

    doc_id = str(uuid.uuid4())
    chunks = chunk_pages(pages, doc_id=doc_id, filename=file.filename or "unnamed")

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="The document produced no indexable text.",
        )

    try:
        get_store().add_chunks(chunks)
    except LLMError as exc:
        # Indexing embeds every chunk, so a credentials or quota problem
        # surfaces here. Report it as a dependency failure with the actual
        # cause rather than as an opaque 500.
        logger.error("Embedding failed while indexing %s: %s", file.filename, exc)
        raise HTTPException(
            status_code=503,
            detail=(
                f"Could not embed the document: {exc} "
                "Set EMBEDDING_PROVIDER=local to embed in-process without "
                "credentials."
            ),
        ) from exc

    logger.info(
        "Indexed %s: %d pages -> %d chunks", file.filename, len(pages), len(chunks)
    )

    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename or "unnamed",
        pages=len(pages),
        chunks=len(chunks),
    )


@app.post("/chat")
async def chat(request: ChatRequest) -> dict:
    """Answer a question from the indexed documents."""
    session_id = request.session_id
    history = memory.get_history(session_id)

    # Rewrite dependent follow-ups into standalone retrieval queries.
    try:
        search_query = memory.condense(request.question, history)
    except LLMError:
        search_query = request.question

    try:
        result = answer_question(
            question=search_query,
            doc_ids=request.doc_ids or None,
            history=history,
        )
    except LLMError as exc:
        logger.error("LLM call failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    memory.add_turn(session_id, "user", request.question)
    memory.add_turn(session_id, "assistant", result.answer)

    payload = result.to_dict()
    # Exposed so the condensing step is observable during a demo.
    payload["search_query"] = search_query
    return payload


@app.get("/documents")
async def list_documents() -> dict:
    """List indexed documents."""
    return {"documents": get_store().list_documents()}


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str) -> dict:
    """Remove a document and all of its chunks."""
    get_store().delete_document(doc_id)
    return {"deleted": doc_id}


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str) -> dict:
    """Clear a chat session's history."""
    memory.clear(session_id)
    return {"cleared": session_id}


@app.get("/health")
async def health() -> dict:
    """Readiness plus dependency status.

    Reports the vector store and Vertex AI separately: the store working while
    Vertex fails is the signature of a credentials problem, which is the most
    likely difference between local and deployed environments.
    """
    try:
        chunks = get_store().count()
        store_status = {"ok": True, "chunks": chunks}
    except Exception as exc:  # noqa: BLE001
        store_status = {"ok": False, "error": str(exc)}

    try:
        llm_status = get_llm().health()
    except LLMError as exc:
        llm_status = {"ok": False, "error": str(exc)}

    return {
        "status": "ok" if store_status["ok"] and llm_status["ok"] else "degraded",
        "store": store_status,
        "llm": llm_status,
        "supported_types": sorted(SUPPORTED_EXTENSIONS),
    }
