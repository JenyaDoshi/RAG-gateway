"""Stage 6 - LLM generation client.

Input : a Prompt.
Output: the answer text (str).

The rest of the app only knows the tiny `Generator` interface below. Gemini is
one implementation. In the gateway milestone, a second provider will be another
implementation of the same interface, and nothing upstream will change.
"""

from __future__ import annotations

from typing import Protocol

from .errors import GenerationError
from .prompt import Prompt


class Generator(Protocol):
    name: str

    def generate(self, prompt: Prompt) -> str:
        """Return the answer text, or raise GenerationError."""


def _explain_api_error(exc) -> str:
    """Turn a Gemini API error into a message that says what to do next."""
    code = getattr(exc, "code", None)
    detail = getattr(exc, "message", None) or str(exc)
    if code == 429:
        return ("Gemini rate limit or quota reached (HTTP 429). Wait a minute and retry, "
                "or switch GEMINI_MODEL. Free-tier limits are low. This is exactly the "
                f"situation the LLM Gateway will handle. Details: {detail}")
    if code in (401, 403) or (code == 400 and "api key" in detail.lower()):
        return ("Gemini rejected the API key (authentication/permission error). Check "
                f"GEMINI_API_KEY in your .env file. Details: {detail}")
    if code == 404:
        return (f"Gemini model not found (HTTP 404). Check GEMINI_MODEL in your .env. Details: {detail}")
    if code == 400:
        return f"Gemini rejected the request as invalid (HTTP 400). Details: {detail}"
    if code is not None and code >= 500:
        return f"Gemini is temporarily unavailable (HTTP {code}). Try again shortly. Details: {detail}"
    return f"Gemini API error (HTTP {code}): {detail}"


class GeminiGenerator:
    name = "gemini"

    def __init__(self, api_key: str, model: str, timeout_s: int = 60, client=None):
        self.model = model
        if client is None:
            from google import genai
            from google.genai import types

            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=timeout_s * 1000),  # SDK expects milliseconds
            )
        self._client = client

    def generate(self, prompt: Prompt) -> str:
        from google.genai import errors, types

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt.user,
                config=types.GenerateContentConfig(system_instruction=prompt.system),
            )
        except errors.APIError as exc:
            raise GenerationError(
                _explain_api_error(exc), status_code=getattr(exc, "code", None), provider=self.name
            ) from exc
        except Exception as exc:  # network down, timeout, DNS, ...
            raise GenerationError(
                f"Could not reach the Gemini API ({type(exc).__name__}: {exc}). "
                "Check your internet connection and LLM_TIMEOUT_S.",
                provider=self.name,
            ) from exc

        text = (getattr(response, "text", None) or "").strip()
        if not text:
            reason = ""
            candidates = getattr(response, "candidates", None) or []
            if candidates and getattr(candidates[0], "finish_reason", None):
                reason = f" (finish reason: {candidates[0].finish_reason})"
            raise GenerationError(
                f"Gemini returned an empty answer{reason}. The response may have been "
                "blocked by a safety filter. Try rephrasing the question.",
                provider=self.name,
            )
        return text
