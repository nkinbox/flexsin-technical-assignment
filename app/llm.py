"""Stage 5a — LLM client (Vertex AI).

A deliberately narrow wrapper over `google-genai` in Vertex mode. `rag.py`
depends only on `generate_json` / `generate_text` / `stream_text`, so swapping
in a different backend (a local Ollama server, another hosted provider) means
reimplementing this one class and nothing else.

Authentication is Application Default Credentials throughout:
  - Local dev:  `gcloud auth application-default login`
  - On the VM:  the attached service account, via the metadata server

No API keys, and no service-account JSON key file anywhere. A downloaded key
is the most common way demo projects leak credentials.
"""

from __future__ import annotations

import json
from typing import Iterator

from google import genai
from google.genai import types

from app.config import (
    GCP_LOCATION,
    GCP_PROJECT,
    LLM_MAX_OUTPUT_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
)


class LLMError(RuntimeError):
    """Raised when the model is unreachable or misconfigured."""


class VertexLLM:
    """Thin Vertex AI client."""

    def __init__(
        self,
        project: str = GCP_PROJECT,
        location: str = GCP_LOCATION,
        model: str = LLM_MODEL,
    ):
        if not project:
            raise LLMError(
                "GCP_PROJECT is not set. Copy .env.example to .env and set your "
                "project id, then authenticate with:\n"
                "  gcloud auth application-default login"
            )

        self.model = model
        self._project = project
        self._location = location

        try:
            self._client = genai.Client(
                vertexai=True, project=project, location=location
            )
        except Exception as exc:  # noqa: BLE001 - surface actionable guidance
            raise LLMError(
                f"Could not initialise the Vertex AI client: {exc}\n"
                "Check that:\n"
                "  1. Application Default Credentials exist "
                "(`gcloud auth application-default login`), or the VM has an "
                "attached service account\n"
                "  2. The Vertex AI API is enabled "
                "(`gcloud services enable aiplatform.googleapis.com`)\n"
                f"  3. Project '{project}' is correct"
            ) from exc

    def _config(
        self,
        system_instruction: str,
        response_schema: dict | None = None,
        max_output_tokens: int = LLM_MAX_OUTPUT_TOKENS,
    ) -> types.GenerateContentConfig:
        """Build a generation config shared by every call path."""
        kwargs = {
            "system_instruction": system_instruction,
            "temperature": LLM_TEMPERATURE,
            "max_output_tokens": max_output_tokens,
            # This is grounded extraction over a handful of short passages, not
            # open-ended reasoning. Disabling thinking removes latency and cost
            # the task does not benefit from.
            "thinking_config": types.ThinkingConfig(thinking_budget=0),
        }

        if response_schema is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = response_schema

        return types.GenerateContentConfig(**kwargs)

    def generate_json(
        self,
        system_instruction: str,
        prompt: str,
        response_schema: dict,
    ) -> dict:
        """Generate a response constrained to a JSON schema.

        The schema constraint is what makes citations machine-readable instead
        of something to regex out of prose. See execution.md §5, layer 2.

        Raises:
            LLMError: The call failed, or returned unparseable JSON.
        """
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=self._config(system_instruction, response_schema),
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Vertex AI request failed: {exc}") from exc

        text = (response.text or "").strip()
        if not text:
            raise LLMError("Vertex AI returned an empty response.")

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            # Should not happen with a schema applied, but a malformed payload
            # must not surface as a raw traceback to the user.
            raise LLMError(f"Model returned malformed JSON: {text[:200]}") from exc

    def generate_text(
        self,
        system_instruction: str,
        prompt: str,
        max_output_tokens: int = LLM_MAX_OUTPUT_TOKENS,
    ) -> str:
        """Generate plain text. Used for query condensing."""
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=self._config(
                    system_instruction, max_output_tokens=max_output_tokens
                ),
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Vertex AI request failed: {exc}") from exc

        return (response.text or "").strip()

    def stream_text(self, system_instruction: str, prompt: str) -> Iterator[str]:
        """Yield text chunks as they are generated."""
        try:
            stream = self._client.models.generate_content_stream(
                model=self.model,
                contents=prompt,
                config=self._config(system_instruction),
            )
            for event in stream:
                if event.text:
                    yield event.text
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Vertex AI streaming request failed: {exc}") from exc

    def health(self) -> dict:
        """Confirm credentials resolve and the model responds.

        Called by `/health` so a misconfiguration surfaces as a clear status
        rather than as a failure on the user's first question.
        """
        try:
            self._client.models.generate_content(
                model=self.model,
                contents="ping",
                config=types.GenerateContentConfig(
                    max_output_tokens=8,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return {
                "ok": True,
                "model": self.model,
                "project": self._project,
                "location": self._location,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "model": self.model,
                "project": self._project,
                "location": self._location,
                "error": str(exc),
            }


_llm: VertexLLM | None = None


def get_llm() -> VertexLLM:
    """Return the process-wide client, constructing it on first use."""
    global _llm
    if _llm is None:
        _llm = VertexLLM()
    return _llm
