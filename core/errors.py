"""Custom errors + a translator that turns any provider exception into a friendly message."""
from __future__ import annotations


class ChatbotError(Exception):
    """Base class. `user_message` is always safe to show in the UI."""

    def __init__(self, user_message: str, *, detail: str = ""):
        super().__init__(detail or user_message)
        self.user_message = user_message
        self.detail = detail


class ConfigError(ChatbotError):
    pass


class LLMError(ChatbotError):
    pass


class DataLoadError(ChatbotError):
    pass


class BookingError(ChatbotError):
    pass


def friendly_llm_error(provider: str, exc: Exception) -> str:
    """Map raw SDK errors (Gemini / Groq / Ollama) to short, actionable text."""
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(k in text for k in ("api key not valid", "api_key_invalid", "invalid api key", "401", "unauthenticated", "permission_denied", "403")):
        return f"{provider}: API key invalid ya permission nahi hai. .env me key check karein."
    if any(k in text for k in ("429", "rate limit", "resource_exhausted", "quota")):
        return f"{provider}: Rate limit / free quota khatam. Thodi der baad try karein ya dusra provider use hoga."
    if any(k in text for k in ("connection refused", "connecterror", "failed to establish", "name or service not known", "max retries exceeded")):
        return f"{provider}: Server se connect nahi ho paya (internet ya Ollama chal raha hai check karein)."
    if any(k in text for k in ("timeout", "timed out", "deadline")):
        return f"{provider}: Request timeout ho gayi."
    if any(k in text for k in ("not found", "404", "model_not_found", "does not exist")):
        return f"{provider}: Model name galat hai ya available nahi hai (.env me model check karein)."
    if any(k in text for k in ("safety", "blocked", "finish_reason")):
        return f"{provider}: Response safety filter ne block kar diya. Sawal ko alag tarah se poochein."
    if any(k in text for k in ("500", "503", "unavailable", "internal", "overloaded")):
        return f"{provider}: Provider server abhi busy/down hai."
    return f"{provider}: Unexpected error ({type(exc).__name__})."
