"""LLM layer with automatic fallback.

Providers:
  * gemini  - Google Gemini (GEMINI_API_KEY, free tier at https://aistudio.google.com/apikey)
  * groq    - Free hosted open-source LLMs like Llama 3 (GROQ_API_KEY, https://console.groq.com)
  * ollama  - 100% free local LLM (install https://ollama.com, then `ollama pull llama3.2`)
  * mock    - offline echo bot, used by tests / demo with no keys

`LLMRouter.chat()` tries providers in order with retries; if all fail it raises LLMError
with a combined friendly message instead of crashing the app.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from core.config import Settings
from core.errors import ConfigError, LLMError, friendly_llm_error
from core.logger import get_logger

log = get_logger("llm")

PROVIDER_LABELS = {
    "gemini": "Google Gemini",
    "groq": "Groq (free Llama)",
    "ollama": "Ollama (free local)",
    "mock": "Offline demo",
}


class MockChatModel:
    """Tiny stand-in for a LangChain chat model. No network, deterministic."""

    def invoke(self, messages: list[BaseMessage]):
        last_user = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        system = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
        has_data = "DATA CONTEXT" in system
        return AIMessage(
            content=(
                "[Offline demo mode - koi real LLM connect nahi hai]\n\n"
                f"Aapka sawal: {last_user[:300]}\n"
                f"Uploaded data context mila: {'haan' if has_data else 'nahi'}.\n"
                "Real jawab ke liye .env me GEMINI_API_KEY ya GROQ_API_KEY daalein, ya Ollama chalayein."
            )
        )


def _content_to_text(content) -> str:
    """Gemini sometimes returns a list of content parts instead of a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return "".join(parts)
    return str(content or "")


def build_model(provider: str, settings: Settings):
    """Create a chat model for one provider. Raises ConfigError if it can't be configured."""
    if provider == "gemini":
        if not settings.gemini_api_key:
            raise ConfigError("Gemini: GEMINI_API_KEY set nahi hai.")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ConfigError("Gemini: `pip install langchain-google-genai` chalayein.") from exc
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=settings.temperature,
            timeout=settings.request_timeout,
            max_retries=0,  # we handle retries ourselves for consistent behaviour
        )

    if provider == "groq":
        if not settings.groq_api_key:
            raise ConfigError("Groq: GROQ_API_KEY set nahi hai.")
        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:
            raise ConfigError("Groq: `pip install langchain-groq` chalayein.") from exc
        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=settings.temperature,
            timeout=settings.request_timeout,
            max_retries=0,
        )

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise ConfigError("Ollama: `pip install langchain-ollama` chalayein.") from exc
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.temperature,
            client_kwargs={"timeout": settings.request_timeout},
        )

    if provider == "mock":
        return MockChatModel()

    raise ConfigError(f"Unknown provider '{provider}'.")


def _is_retryable(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(k in text for k in ("429", "rate", "timeout", "timed out", "503", "500", "unavailable", "overloaded"))


@dataclass
class LLMReply:
    text: str
    provider: str


class LLMRouter:
    def __init__(self, settings: Settings, providers: Iterable[str] | None = None):
        self.settings = settings
        self.providers = list(providers or settings.providers())
        self._models: dict[str, object] = {}

    def _model(self, provider: str):
        if provider not in self._models:
            self._models[provider] = build_model(provider, self.settings)
        return self._models[provider]

    def chat(self, messages: list[BaseMessage]) -> LLMReply:
        if not messages:
            raise LLMError("Khali message bheja nahi ja sakta.")

        problems: list[str] = []
        for provider in self.providers:
            label = PROVIDER_LABELS.get(provider, provider)
            try:
                model = self._model(provider)
            except ConfigError as exc:
                problems.append(exc.user_message)
                continue

            attempts = max(1, self.settings.max_retries + 1)
            for attempt in range(1, attempts + 1):
                try:
                    response = model.invoke(messages)
                    text = _content_to_text(getattr(response, "content", response)).strip()
                    if not text:
                        raise LLMError(f"{label}: khali jawab mila.")
                    return LLMReply(text=text, provider=label)
                except Exception as exc:  # noqa: BLE001 - every SDK raises different types
                    log.warning("Provider %s attempt %s failed: %r", provider, attempt, exc)
                    if attempt < attempts and _is_retryable(exc):
                        time.sleep(min(2 ** attempt, 8))
                        continue
                    problems.append(exc.user_message if isinstance(exc, LLMError) else friendly_llm_error(label, exc))
                    break

        raise LLMError(
            "Koi bhi LLM provider jawab nahi de paya:\n- " + "\n- ".join(problems or ["Koi provider configure nahi hai."])
        )
