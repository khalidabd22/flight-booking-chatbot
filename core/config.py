"""Central configuration. Values come from (in order): sidebar overrides -> .env / environment
-> Streamlit secrets -> defaults. Nothing here ever raises: missing keys are reported, not crashed on."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _secret(name: str, default: str = "") -> str:
    """Read from env first, then Streamlit secrets (for Streamlit Cloud), never raise."""
    value = os.getenv(name)
    if value:
        return value.strip()
    try:
        import streamlit as st  # imported lazily so tests run without a Streamlit context

        return str(st.secrets.get(name, default)).strip()
    except Exception:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(_secret(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    gemini_api_key: str = field(default_factory=lambda: _secret("GEMINI_API_KEY") or _secret("GOOGLE_API_KEY"))
    gemini_model: str = field(default_factory=lambda: _secret("GEMINI_MODEL", "gemini-2.5-flash"))

    groq_api_key: str = field(default_factory=lambda: _secret("GROQ_API_KEY"))
    groq_model: str = field(default_factory=lambda: _secret("GROQ_MODEL", "llama-3.3-70b-versatile"))

    ollama_base_url: str = field(default_factory=lambda: _secret("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _secret("OLLAMA_MODEL", "llama3.2"))

    # Order in which providers are tried. First one that works answers.
    provider_order: str = field(default_factory=lambda: _secret("LLM_PROVIDER_ORDER", "gemini,groq,ollama"))

    temperature: float = 0.3
    request_timeout: int = field(default_factory=lambda: _int("LLM_TIMEOUT_SECONDS", 60))
    max_retries: int = field(default_factory=lambda: _int("LLM_MAX_RETRIES", 2))
    max_history_turns: int = field(default_factory=lambda: _int("MAX_HISTORY_TURNS", 10))
    max_context_chars: int = field(default_factory=lambda: _int("MAX_CONTEXT_CHARS", 12000))
    max_upload_mb: int = field(default_factory=lambda: _int("MAX_UPLOAD_MB", 20))

    db_path: str = field(default_factory=lambda: _secret("BOOKING_DB_PATH", str(BASE_DIR / "data" / "bookings.db")))

    def providers(self) -> list[str]:
        valid = {"gemini", "groq", "ollama", "mock"}
        order = [p.strip().lower() for p in self.provider_order.split(",") if p.strip()]
        return [p for p in order if p in valid] or ["gemini", "groq", "ollama"]


def get_settings(**overrides) -> Settings:
    settings = Settings()
    for key, value in overrides.items():
        if value not in (None, "") and hasattr(settings, key):
            setattr(settings, key, value)
    return settings
