"""Verify Vertex AI connectivity before running the app.

Run this once after setting up GCP. It isolates the credentials/model/region
questions from the rest of the pipeline, so a failure here points at exactly
one thing rather than surfacing later as a confusing runtime error.

    python scripts/verify_vertex.py

Checks, in order:
  1. Credentials resolve (ADC locally, metadata server on a VM)
  2. The configured model exists in the configured region and responds
  3. Structured JSON output works — the mechanism grounding depends on
  4. Embeddings work, when Vertex is the configured embedding provider
  5. Vision works — image and scanned-PDF ingestion depend on it
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import GCP_LOCATION, GCP_PROJECT, LLM_MODEL  # noqa: E402


def main() -> int:
    print("Vertex AI configuration check")
    print("-" * 60)
    print(f"  project  : {GCP_PROJECT or '(not set)'}")
    print(f"  location : {GCP_LOCATION}")
    print(f"  model    : {LLM_MODEL}")
    print("-" * 60)

    if not GCP_PROJECT:
        print("\nFAIL: GCP_PROJECT is not set.")
        print("  cp .env.example .env   # then set GCP_PROJECT")
        return 1

    # 1. Credentials
    print("\n[1/5] Resolving credentials…")
    try:
        import google.auth

        credentials, detected_project = google.auth.default()
        source = type(credentials).__module__.split(".")[-1]
        print(f"      OK — resolved via {source}")
        if detected_project and detected_project != GCP_PROJECT:
            print(
                f"      NOTE: credentials default to project "
                f"'{detected_project}', but GCP_PROJECT is '{GCP_PROJECT}'. "
                "Requests will use GCP_PROJECT."
            )
    except Exception as exc:  # noqa: BLE001
        print(f"      FAIL: {exc}")
        print("\n  Locally:  gcloud auth application-default login")
        print("  On a VM:  attach a service account with roles/aiplatform.user")
        return 1

    # 2. Model reachability
    print(f"\n[2/5] Calling {LLM_MODEL}…")
    try:
        from app.llm import VertexLLM

        llm = VertexLLM()
        reply = llm.generate_text(
            system_instruction="Reply with exactly one word.",
            prompt="Say OK.",
            max_output_tokens=16,
        )
        print(f"      OK — model replied: {reply!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"      FAIL: {exc}")
        print("\n  Common causes:")
        print("    - Vertex AI API not enabled:")
        print("        gcloud services enable aiplatform.googleapis.com")
        print(f"    - Model '{LLM_MODEL}' unavailable in '{GCP_LOCATION}'.")
        print("      Try another region, or set LLM_MODEL in .env to a model")
        print("      listed in the Vertex AI Model Garden for your project.")
        print("    - The service account lacks roles/aiplatform.user")
        return 1

    # 3. Structured output — the mechanism grounding depends on
    print("\n[3/5] Checking structured JSON output…")
    try:
        result = llm.generate_json(
            system_instruction="Answer from the source only.",
            prompt="SOURCE:\n[1] The sky is blue.\n\nQUESTION: What colour is the sky?",
            response_schema={
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "citations": {"type": "array", "items": {"type": "integer"}},
                    "found": {"type": "boolean"},
                },
                "required": ["answer", "citations", "found"],
            },
        )
        assert isinstance(result, dict) and "found" in result
        print(f"      OK — structured response: {result}")
    except Exception as exc:  # noqa: BLE001
        print(f"      FAIL: {exc}")
        print("\n  The model responded but could not produce schema-constrained")
        print("  JSON. Try a different model id in .env.")
        return 1

    # 4. Embeddings — only when Vertex is the configured provider
    from app.config import EMBEDDING_PROVIDER, VERTEX_EMBED_MODEL

    if EMBEDDING_PROVIDER == "vertex":
        print(f"\n[4/5] Embedding with {VERTEX_EMBED_MODEL}…")
        try:
            from app.embedder import VertexEmbedder

            embedder = VertexEmbedder()
            vectors = embedder.embed_documents(["a test passage about revenue"])
            query = embedder.embed_query("what was revenue?")
            assert len(vectors[0]) == len(query) == embedder.dimension
            print(f"      OK — {embedder.dimension}-dimensional vectors")
        except Exception as exc:  # noqa: BLE001
            print(f"      FAIL: {exc}")
            print(f"\n  Check that '{VERTEX_EMBED_MODEL}' is available in your")
            print("  region, or set EMBEDDING_PROVIDER=local to embed in-process.")
            return 1
    else:
        print(f"\n[4/5] Embeddings: using local provider, nothing to check")

    # 5. Vision — image ingestion depends on it
    print("\n[5/5] Checking vision (image ingestion)…")
    try:
        # A 1x1 PNG is enough to confirm the model accepts image input.
        import base64

        pixel = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        )
        llm.extract_from_media(pixel, "image/png", "Describe this image briefly.")
        print("      OK — the model accepts image input")
    except Exception as exc:  # noqa: BLE001
        print(f"      FAIL: {exc}")
        print("\n  Image and scanned-PDF ingestion will not work.")
        print(f"  Check that VISION_MODEL is multimodal (current: {llm.model}).")
        return 1

    print("\n" + "=" * 60)
    print("All checks passed. Start the app with:  docker compose up --build")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
