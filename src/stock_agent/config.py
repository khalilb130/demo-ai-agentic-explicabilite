from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    openai_api_key: str | None
    groq_api_key: str | None
    groq_model: str
    mistral_api_key: str | None
    mistral_model: str
    newsapi_key: str | None
    alpha_vantage_api_key: str | None
    ollama_model: str
    ollama_base_url: str
    enable_ollama_fallback: bool
    openai_model: str
    news_articles_limit: int

    @property
    def llm_enabled(self) -> bool:
        return (
            bool(self.groq_api_key)
            or bool(self.mistral_api_key)
            or bool(self.openai_api_key)
            or self.enable_ollama_fallback
        )

    @property
    def newsapi_enabled(self) -> bool:
        return bool(self.newsapi_key)

    @property
    def alpha_vantage_enabled(self) -> bool:
        return bool(self.alpha_vantage_api_key)

    @property
    def ollama_enabled(self) -> bool:
        return self.enable_ollama_fallback


def _read_optional_env(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def load_config() -> AppConfig:
    raw_limit = os.getenv("NEWS_ARTICLES_LIMIT", "5").strip()
    try:
        news_limit = max(1, min(15, int(raw_limit)))
    except ValueError:
        news_limit = 5

    ollama_toggle = os.getenv("ENABLE_OLLAMA_FALLBACK", "false").strip().lower()
    enable_ollama = ollama_toggle in {"1", "true", "yes", "on"}

    return AppConfig(
        openai_api_key=_read_optional_env("OPENAI_API_KEY"),
        groq_api_key=_read_optional_env("GROQ_API_KEY"),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip(),
        mistral_api_key=_read_optional_env("MISTRAL_API_KEY"),
        mistral_model=os.getenv("MISTRAL_MODEL", "mistral-small-latest").strip(),
        newsapi_key=_read_optional_env("NEWSAPI_KEY"),
        alpha_vantage_api_key=_read_optional_env("ALPHA_VANTAGE_API_KEY"),
        ollama_model=os.getenv("OLLAMA_MODEL", "mistral:latest").strip(),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip(),
        enable_ollama_fallback=enable_ollama,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
        news_articles_limit=news_limit,
    )
