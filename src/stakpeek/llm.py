from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import openai
from openai import OpenAI

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"


class LLMError(RuntimeError):
    """A problem talking to the model, with a message safe to show the user."""


class LLM(Protocol):
    def stream(self, system: str, user: str) -> Iterator[str]:
        """Yield the model's answer piece by piece."""
        ...


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    model: str
    api_key: str

    @classmethod
    def from_env(cls) -> LLMConfig:
        key = os.environ.get("STAKPEEK_API_KEY") or os.environ.get("GROQ_API_KEY")
        if not key:
            raise LLMError("Set GROQ_API_KEY (or STAKPEEK_API_KEY) in your environment.")
        return cls(
            base_url=os.environ.get("STAKPEEK_BASE_URL", DEFAULT_BASE_URL),
            model=os.environ.get("STAKPEEK_MODEL", DEFAULT_MODEL),
            api_key=key,
        )


class OpenAICompatLLM:
    """Any OpenAI-compatible chat endpoint (Groq, OpenRouter, Gemini, Ollama...)."""

    def __init__(self, config: LLMConfig) -> None:
        self._model = config.model
        self._client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    def stream(self, system: str, user: str) -> Iterator[str]:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                stream=True,
            )
            for event in response:
                if not event.choices:
                    continue
                piece = event.choices[0].delta.content
                if piece:
                    yield piece
        except openai.RateLimitError as err:
            raise LLMError("Rate limit reached. Wait a minute and try again.") from err
        except openai.AuthenticationError as err:
            raise LLMError("The API key was rejected. Check GROQ_API_KEY.") from err
        except openai.APIConnectionError as err:
            raise LLMError("Could not reach the model server.") from err
