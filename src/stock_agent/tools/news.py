from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import requests
import yfinance as yf
from langchain_core.tools import tool
from pydantic import BaseModel, Field


POSITIVE_TERMS = {
    "growth",
    "beat",
    "record",
    "upgrade",
    "profit",
    "surge",
    "strong",
    "hausse",
    "croissance",
    "record",
}
NEGATIVE_TERMS = {
    "downgrade",
    "miss",
    "lawsuit",
    "decline",
    "loss",
    "weak",
    "drop",
    "baisse",
    "risque",
    "warning",
}


class NewsInput(BaseModel):
    ticker: str = Field(description="Ticker boursier, ex: AAPL.")
    max_articles: int = Field(
        default=5,
        ge=1,
        le=15,
        description="Nombre max d'articles récents à analyser.",
    )


def _headline_score(text: str) -> float:
    tokens = set((text or "").lower().replace(",", " ").replace(".", " ").split())
    if not tokens:
        return 0.0
    pos = len(tokens.intersection(POSITIVE_TERMS))
    neg = len(tokens.intersection(NEGATIVE_TERMS))
    return float(pos - neg)


def _label_from_score(score: float) -> str:
    if score > 0.2:
        return "positif"
    if score < -0.2:
        return "négatif"
    return "neutre"


def _news_from_newsapi(symbol: str, max_articles: int, api_key: str) -> list[dict[str, Any]]:
    response = requests.get(
        "https://newsapi.org/v2/everything",
        params={
            "q": f"{symbol} stock",
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": max_articles,
            "apiKey": api_key,
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    raw_articles = payload.get("articles", [])
    articles: list[dict[str, Any]] = []
    for article in raw_articles[:max_articles]:
        articles.append(
            {
                "title": article.get("title"),
                "description": article.get("description"),
                "source": (article.get("source") or {}).get("name"),
                "published_at": article.get("publishedAt"),
                "url": article.get("url"),
            }
        )
    return articles


def _news_from_yfinance(symbol: str, max_articles: int) -> list[dict[str, Any]]:
    news_items = yf.Ticker(symbol).news or []
    articles: list[dict[str, Any]] = []
    for item in news_items[:max_articles]:
        provider_time = item.get("providerPublishTime")
        published_at = None
        if provider_time:
            published_at = datetime.fromtimestamp(provider_time, tz=timezone.utc).isoformat()
        articles.append(
            {
                "title": item.get("title"),
                "description": item.get("summary"),
                "source": item.get("publisher"),
                "published_at": published_at,
                "url": item.get("link"),
            }
        )
    return articles


@tool(args_schema=NewsInput)
def get_news_context(ticker: str, max_articles: int = 5) -> dict[str, Any]:
    """Récupère le contexte news et estime un sentiment simple (positif/neutre/négatif).

    Priorité: NewsAPI (si clé) puis fallback gratuit yfinance.
    """

    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("Ticker vide.")

    source_used = "yfinance"
    key_available = bool(os.getenv("NEWSAPI_KEY", "").strip())
    articles: list[dict[str, Any]]

    if key_available:
        try:
            source_used = "newsapi"
            articles = _news_from_newsapi(symbol, max_articles, os.getenv("NEWSAPI_KEY", "").strip())
        except Exception:
            source_used = "yfinance"
            articles = _news_from_yfinance(symbol, max_articles)
    else:
        articles = _news_from_yfinance(symbol, max_articles)

    scores: list[float] = []
    for article in articles:
        text = f"{article.get('title', '')} {article.get('description', '')}".strip()
        scores.append(_headline_score(text))

    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "status": "ok" if articles else "no_data",
        "source": source_used,
        "ticker": symbol,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "articles_count": len(articles),
        "sentiment_score": round(avg_score, 3),
        "sentiment_label": _label_from_score(avg_score),
        "articles": articles,
        "newsapi_key_present": key_available,
    }

