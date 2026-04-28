from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import requests
import yfinance as yf
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class FundamentalsInput(BaseModel):
    ticker: str = Field(description="Ticker boursier à analyser.")


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


@tool(args_schema=FundamentalsInput)
def get_fundamentals_context(ticker: str) -> dict[str, Any]:
    """Récupère des fondamentaux d'entreprise via yfinance (PE, croissance, marges, dette).

    Utiliser cet outil pour comparer valorisation et santé financière à la tendance marché.
    """

    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("Ticker vide.")

    info = yf.Ticker(symbol).info or {}
    if not info:
        raise ValueError(f"Impossible de récupérer les fondamentaux pour {symbol}.")

    trailing_pe = _safe_float(info.get("trailingPE"))
    profit_margin = _safe_float(info.get("profitMargins"))
    debt_to_equity = _safe_float(info.get("debtToEquity"))
    revenue_growth = _safe_float(info.get("revenueGrowth"))

    valuation_signal = "neutre"
    if trailing_pe is not None:
        if trailing_pe < 20:
            valuation_signal = "valorisation modérée"
        elif trailing_pe > 35:
            valuation_signal = "valorisation élevée"

    return {
        "status": "ok",
        "source": "yfinance",
        "ticker": symbol,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": _safe_float(info.get("marketCap")),
        "trailing_pe": trailing_pe,
        "forward_pe": _safe_float(info.get("forwardPE")),
        "profit_margins": profit_margin,
        "debt_to_equity": debt_to_equity,
        "revenue_growth": revenue_growth,
        "beta": _safe_float(info.get("beta")),
        "valuation_signal": valuation_signal,
    }


@tool(args_schema=FundamentalsInput)
def get_alpha_vantage_overview(ticker: str) -> dict[str, Any]:
    """Récupère un complément fondamental via Alpha Vantage (si clé API configurée).

    Si ALPHA_VANTAGE_API_KEY est absente, l'outil retourne un statut "skipped".
    """

    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("Ticker vide.")

    if not api_key:
        return {
            "status": "skipped",
            "source": "alpha_vantage",
            "ticker": symbol,
            "message": "ALPHA_VANTAGE_API_KEY absente. Source optionnelle ignorée.",
        }

    response = requests.get(
        "https://www.alphavantage.co/query",
        params={"function": "OVERVIEW", "symbol": symbol, "apikey": api_key},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()

    if not payload or "Symbol" not in payload:
        note = payload.get("Note") if isinstance(payload, dict) else None
        return {
            "status": "unavailable",
            "source": "alpha_vantage",
            "ticker": symbol,
            "message": note or "Aucun résultat Alpha Vantage.",
        }

    return {
        "status": "ok",
        "source": "alpha_vantage",
        "ticker": symbol,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "name": payload.get("Name"),
        "analyst_target_price": _safe_float(payload.get("AnalystTargetPrice")),
        "pe_ratio": _safe_float(payload.get("PERatio")),
        "peg_ratio": _safe_float(payload.get("PEGRatio")),
        "eps": _safe_float(payload.get("EPS")),
        "dividend_yield": _safe_float(payload.get("DividendYield")),
        "profit_margin": _safe_float(payload.get("ProfitMargin")),
    }

