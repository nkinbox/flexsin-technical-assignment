"""Streamlit UI — upload, chat, and citation inspection.

Talks to the FastAPI backend over HTTP; it holds no pipeline logic of its own,
so the API remains the real interface and the UI is a thin client over it.
"""

from __future__ import annotations

import os
import uuid

import httpx
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
TIMEOUT = httpx.Timeout(120.0, connect=10.0)

st.set_page_config(page_title="Document RAG Chatbot", page_icon="📄", layout="wide")


# --- API helpers -------------------------------------------------------------


def api_get(path: str) -> dict | None:
    try:
        response = httpx.get(f"{API_BASE}{path}", timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def api_post(path: str, **kwargs) -> tuple[dict | None, str | None]:
    """POST, returning (payload, error_message)."""
    try:
        response = httpx.post(f"{API_BASE}{path}", timeout=TIMEOUT, **kwargs)
        if response.status_code >= 400:
            try:
                return None, response.json().get("detail", response.text)
            except Exception:
                return None, response.text
        return response.json(), None
    except Exception as exc:  # noqa: BLE001
        return None, f"Could not reach the API at {API_BASE}: {exc}"


# --- Session state -----------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded" not in st.session_state:
    st.session_state.uploaded = set()


# --- Sidebar -----------------------------------------------------------------

with st.sidebar:
    st.title("📄 Documents")

    health = api_get("/health")
    if health is None:
        st.error(f"API unreachable at `{API_BASE}`")
    elif health["status"] == "ok":
        st.success("Backend ready")
    else:
        st.warning("Backend degraded")
        if not health.get("llm", {}).get("ok"):
            with st.expander("Vertex AI error"):
                st.code(health["llm"].get("error", "unknown"), language=None)

    st.divider()

    uploads = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="PDF, DOCX, TXT and MD. Image input is not supported in this build.",
    )

    for upload in uploads or []:
        # Guard against Streamlit's rerun-on-every-interaction model
        # re-uploading the same file repeatedly.
        key = f"{upload.name}:{upload.size}"
        if key in st.session_state.uploaded:
            continue

        with st.spinner(f"Indexing {upload.name}…"):
            payload, error = api_post(
                "/upload",
                files={"file": (upload.name, upload.getvalue())},
            )

        if error:
            st.error(f"{upload.name}: {error}")
        else:
            st.session_state.uploaded.add(key)
            st.success(
                f"{payload['filename']} — {payload['pages']} pages, "
                f"{payload['chunks']} chunks"
            )

    st.divider()

    # Multi-document querying: selected documents scope the search.
    documents = (api_get("/documents") or {}).get("documents", [])
    selected: list[str] = []

    if documents:
        st.subheader("Search scope")
        st.caption("Leave all ticked to search everything.")

        for document in documents:
            row, remove = st.columns([5, 1])
            with row:
                if st.checkbox(
                    f"{document['filename']} ({document['chunks']})",
                    value=True,
                    key=f"sel_{document['doc_id']}",
                ):
                    selected.append(document["doc_id"])
            with remove:
                if st.button("🗑", key=f"del_{document['doc_id']}"):
                    httpx.delete(
                        f"{API_BASE}/documents/{document['doc_id']}", timeout=TIMEOUT
                    )
                    st.rerun()
    else:
        st.info("Upload a document to begin.")

    st.divider()
    if st.button("Clear chat"):
        httpx.delete(
            f"{API_BASE}/sessions/{st.session_state.session_id}", timeout=TIMEOUT
        )
        st.session_state.messages = []
        st.rerun()


# --- Main panel --------------------------------------------------------------

st.title("Document-Aware AI Chatbot")
st.caption(
    "Answers come only from your uploaded documents. "
    "When nothing relevant is found, the model is not called at all."
)


def render_sources(message: dict) -> None:
    """Show citations, plus the grounding flags behind the answer."""
    if message.get("gated"):
        st.caption(
            "🛡️ Relevance gate: nothing in the documents was close enough to "
            "this question, so no model call was made."
        )
        return

    citations = message.get("citations") or []
    if not citations:
        return

    if not message.get("verified", True):
        st.warning(
            "⚠️ The model referenced a source that was not retrieved. "
            "The invalid reference was removed; the sources below are genuine."
        )

    st.caption(f"**Sources** ({len(citations)})")
    for citation in citations:
        label = (
            f"[{citation['number']}] {citation['filename']} "
            f"— page {citation['page_number']}"
        )
        with st.expander(label):
            st.markdown(
                f"<div style='background:#fff8dc;border-left:3px solid #f0c14b;"
                f"padding:0.75rem;border-radius:4px;font-size:0.9rem;"
                f"white-space:pre-wrap'>{_escape(citation['text'])}</div>",
                unsafe_allow_html=True,
            )


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message)


if question := st.chat_input("Ask something about your documents…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents…"):
            payload, error = api_post(
                "/chat",
                json={
                    "question": question,
                    "session_id": st.session_state.session_id,
                    # None means "search everything"; a subset scopes the query.
                    "doc_ids": selected if selected and documents else None,
                },
            )

        if error:
            st.error(error)
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Error: {error}"}
            )
        else:
            st.markdown(payload["answer"])

            # Surface the condensed query so follow-up rewriting is visible.
            if payload.get("search_query", question) != question:
                st.caption(f"🔎 Searched for: *{payload['search_query']}*")

            render_sources(payload)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": payload["answer"],
                    "citations": payload.get("citations", []),
                    "verified": payload.get("verified", True),
                    "gated": payload.get("gated", False),
                }
            )
