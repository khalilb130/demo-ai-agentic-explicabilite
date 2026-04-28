from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import yfinance as yf
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class MarketSnapshotInput(BaseModel):
    ticker: str = Field(description="Ticker boursier, ex: AAPL, MSFT, AIR.PA.")
    period: str = Field(
        default="6mo",
        description="Fenêtre historique yfinance (ex: 3mo, 6mo, 1y).",
    )


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _compute_rsi(close_series, window: int = 14) -> float | None:
    if close_series is None or len(close_series) < window + 1:
        return None
    delta = close_series.diff()
    gains = delta.clip(lower=0).rolling(window=window).mean()
    losses = (-delta.clip(upper=0)).rolling(window=window).mean()
    last_loss = losses.iloc[-1]
    if last_loss is None:
        return None
    if float(last_loss) == 0.0:
        return 100.0
    rs = gains.iloc[-1] / last_loss
    if rs is None:
        return None
    return float(100 - (100 / (1 + rs)))


def _trend_label(price: float | None, sma20: float | None, sma50: float | None) -> str:
    if price is None or sma20 is None or sma50 is None:
        return "indéterminé"
    if price > sma20 > sma50:
        return "haussier"
    if price < sma20 < sma50:
        return "baissier"
    return "neutre"


@tool(args_schema=MarketSnapshotInput)
def get_market_snapshot(ticker: str, period: str = "6mo") -> dict[str, Any]:
    """Récupère le contexte marché: prix courant, variation, moyennes mobiles, RSI et volatilité.

    Utiliser cet outil pour établir la base technique d'une analyse d'action.
    """

    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("Ticker vide.")

    history = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=False)
    if history is None or history.empty:
        raise ValueError(f"Aucune donnée de marché disponible pour {symbol}.")

    closes = history["Close"].dropna()
    if closes.empty:
        raise ValueError(f"Données de clôture manquantes pour {symbol}.")

    current_price = _to_float(closes.iloc[-1])
    previous_close = _to_float(closes.iloc[-2]) if len(closes) > 1 else None
    daily_change_pct = None
    if current_price is not None and previous_close not in (None, 0):
        daily_change_pct = ((current_price - previous_close) / previous_close) * 100

    sma20 = _to_float(closes.rolling(window=20).mean().iloc[-1]) if len(closes) >= 20 else None
    sma50 = _to_float(closes.rolling(window=50).mean().iloc[-1]) if len(closes) >= 50 else None
    rsi14 = _compute_rsi(closes, window=14)

    returns = closes.pct_change().dropna()
    volatility_30d = None
    if len(returns) >= 30:
        volatility_30d = _to_float(returns.tail(30).std() * (252**0.5) * 100)

    return {
        "status": "ok",
        "source": "yfinance",
        "ticker": symbol,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "current_price": round(current_price, 4) if current_price is not None else None,
        "previous_close": round(previous_close, 4) if previous_close is not None else None,
        "daily_change_pct": round(daily_change_pct, 3) if daily_change_pct is not None else None,
        "sma20": round(sma20, 4) if sma20 is not None else None,
        "sma50": round(sma50, 4) if sma50 is not None else None,
        "rsi14": round(rsi14, 2) if rsi14 is not None else None,
        "volatility_30d_annualized_pct": round(volatility_30d, 2) if volatility_30d is not None else None,
        "trend": _trend_label(current_price, sma20, sma50),
    }

