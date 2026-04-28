from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from .config import AppConfig
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .tools import (
    get_alpha_vantage_overview,
    get_fundamentals_context,
    get_market_snapshot,
    get_news_context,
)

DISCLAIMER = "Contenu éducatif uniquement. Ceci n'est pas un conseil financier."
VALID_TICKER_PATTERN = re.compile(r"^[A-Za-z0-9.\-]{1,15}$")


@dataclass
class ActionLog:
    tool: str
    status: str
    detail: str
    output: dict[str, Any]


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _invoke_tool(tool_obj, payload: dict[str, Any]) -> ActionLog:
    try:
        output = tool_obj.invoke(payload)
        status = str(output.get("status", "ok")) if isinstance(output, dict) else "ok"
        detail = f"Appel réussi ({tool_obj.name})"
        if isinstance(output, dict) and output.get("message"):
            detail = str(output["message"])
        if isinstance(output, dict) and output.get("status") == "skipped":
            detail = str(output.get("message", "Outil ignoré"))
    except Exception as exc:
        output = {"status": "error", "message": str(exc)}
        status = "error"
        detail = str(exc)

    return ActionLog(tool=tool_obj.name, status=status, detail=detail, output=output)


def _collect_perception(ticker: str, config: AppConfig) -> tuple[dict[str, Any], list[ActionLog]]:
    actions: list[ActionLog] = []
    perception: dict[str, Any] = {}

    market_action = _invoke_tool(get_market_snapshot, {"ticker": ticker})
    fundamentals_action = _invoke_tool(get_fundamentals_context, {"ticker": ticker})
    news_action = _invoke_tool(
        get_news_context,
        {"ticker": ticker, "max_articles": config.news_articles_limit},
    )

    actions.extend([market_action, fundamentals_action, news_action])
    perception["market"] = market_action.output
    perception["fundamentals"] = fundamentals_action.output
    perception["news"] = news_action.output

    if config.alpha_vantage_enabled:
        alpha_action = _invoke_tool(get_alpha_vantage_overview, {"ticker": ticker})
        actions.append(alpha_action)
        perception["alpha_vantage"] = alpha_action.output
    else:
        perception["alpha_vantage"] = {
            "status": "skipped",
            "message": "ALPHA_VANTAGE_API_KEY absente.",
        }

    return perception, actions


@dataclass
class ScoreStep:
    signal: str
    value: str
    delta: int
    reason: str


def _score_recommendation(perception: dict[str, Any]) -> dict[str, Any]:
    score = 0
    steps: list[ScoreStep] = []
    comparisons: list[str] = []
    caveats: list[str] = []

    market = perception.get("market", {})
    fundamentals = perception.get("fundamentals", {})
    news = perception.get("news", {})

    trend = market.get("trend")
    if trend == "haussier":
        score += 2
        steps.append(ScoreStep("Tendance (SMA20/50)", trend, +2, "Prix > SMA20 > SMA50 → momentum haussier."))
    elif trend == "baissier":
        score -= 2
        steps.append(ScoreStep("Tendance (SMA20/50)", trend, -2, "Prix < SMA20 < SMA50 → momentum baissier."))
    else:
        steps.append(ScoreStep("Tendance (SMA20/50)", trend or "indéterminé", 0, "Signal technique neutre."))
        caveats.append("Tendance technique peu claire (signal neutre/indéterminé).")

    daily_change = _to_float(market.get("daily_change_pct"))
    if daily_change is not None:
        if daily_change > 2:
            score += 1
            steps.append(ScoreStep("Variation 1j", f"+{daily_change:.2f}%", +1, "Hausse journalière > 2%."))
        elif daily_change < -2:
            score -= 1
            steps.append(ScoreStep("Variation 1j", f"{daily_change:.2f}%", -1, "Baisse journalière > 2%."))
        else:
            steps.append(ScoreStep("Variation 1j", f"{daily_change:.2f}%", 0, "Variation dans la norme (±2%)."))

    rsi = _to_float(market.get("rsi14"))
    if rsi is not None:
        if rsi < 35:
            score += 1
            steps.append(ScoreStep("RSI 14j", f"{rsi:.1f}", +1, "RSI < 35: zone de survente, rebond possible."))
            comparisons.append("RSI bas: possible zone de rebond (survente).")
        elif rsi > 70:
            score -= 1
            steps.append(ScoreStep("RSI 14j", f"{rsi:.1f}", -1, "RSI > 70: zone de surachat, risque de correction."))
            comparisons.append("RSI élevé: risque de surchauffe court terme.")
        else:
            steps.append(ScoreStep("RSI 14j", f"{rsi:.1f}", 0, "RSI neutre (35–70)."))

    trailing_pe = _to_float(fundamentals.get("trailing_pe"))
    if trailing_pe is not None:
        if trailing_pe < 20:
            score += 1
            steps.append(ScoreStep("P/E", f"{trailing_pe:.1f}", +1, "Valorisation modérée (PE < 20)."))
        elif trailing_pe > 35:
            score -= 1
            steps.append(ScoreStep("P/E", f"{trailing_pe:.1f}", -1, "Valorisation élevée (PE > 35)."))
            comparisons.append("Valorisation élevée (PE > 35).")
        else:
            steps.append(ScoreStep("P/E", f"{trailing_pe:.1f}", 0, "Valorisation dans la moyenne (20–35)."))
    else:
        steps.append(ScoreStep("P/E", "N/D", 0, "Donnée non disponible."))
        caveats.append("PE indisponible pour affiner la valorisation.")

    profit_margins = _to_float(fundamentals.get("profit_margins"))
    if profit_margins is not None:
        pct = profit_margins * 100
        if profit_margins > 0.15:
            score += 1
            steps.append(ScoreStep("Marge nette", f"{pct:.1f}%", +1, "Marges solides (> 15%)."))
        elif profit_margins < 0.05:
            score -= 1
            steps.append(ScoreStep("Marge nette", f"{pct:.1f}%", -1, "Marges faibles (< 5%)."))
            comparisons.append("Marges faibles, pression potentielle sur la rentabilité.")
        else:
            steps.append(ScoreStep("Marge nette", f"{pct:.1f}%", 0, "Marges acceptables (5–15%)."))

    debt_to_equity = _to_float(fundamentals.get("debt_to_equity"))
    if debt_to_equity is not None:
        if debt_to_equity > 150:
            score -= 1
            steps.append(ScoreStep("Dette/Capitaux", f"{debt_to_equity:.0f}%", -1, "Levier élevé (D/E > 150%)."))
            comparisons.append("Levier élevé (dette/capitaux propres).")
        elif debt_to_equity < 80:
            score += 1
            steps.append(ScoreStep("Dette/Capitaux", f"{debt_to_equity:.0f}%", +1, "Bilan sain (D/E < 80%)."))
        else:
            steps.append(ScoreStep("Dette/Capitaux", f"{debt_to_equity:.0f}%", 0, "Levier modéré (80–150%)."))

    revenue_growth = _to_float(fundamentals.get("revenue_growth"))
    if revenue_growth is not None:
        pct = revenue_growth * 100
        if revenue_growth > 0.1:
            score += 1
            steps.append(ScoreStep("Croissance CA", f"+{pct:.1f}%", +1, "Croissance du chiffre d'affaires > 10%."))
        elif revenue_growth < 0:
            score -= 1
            steps.append(ScoreStep("Croissance CA", f"{pct:.1f}%", -1, "Chiffre d'affaires en recul."))
            comparisons.append("Croissance du chiffre d'affaires négative.")
        else:
            steps.append(ScoreStep("Croissance CA", f"+{pct:.1f}%", 0, "Croissance modeste (0–10%)."))

    sentiment = news.get("sentiment_label")
    sentiment_score_val = _to_float(news.get("sentiment_score"))
    if sentiment == "positif":
        score += 1
        steps.append(ScoreStep("Sentiment news", f"positif ({sentiment_score_val:+.2f})" if sentiment_score_val is not None else "positif", +1, "Couverture médiatique favorable."))
    elif sentiment == "négatif":
        score -= 1
        steps.append(ScoreStep("Sentiment news", f"négatif ({sentiment_score_val:+.2f})" if sentiment_score_val is not None else "négatif", -1, "Couverture médiatique défavorable."))
        comparisons.append("Sentiment news défavorable.")
    else:
        steps.append(ScoreStep("Sentiment news", "neutre", 0, "Pas de signal fort dans les titres récents."))
        caveats.append("Sentiment news neutre ou peu représentatif.")

    if score >= 3:
        recommendation = "Achat"
    elif score <= -3:
        recommendation = "Vente"
    else:
        recommendation = "Conserver"

    data_blocks = 0
    for key in ("market", "fundamentals", "news"):
        if perception.get(key, {}).get("status") in ("ok", "no_data"):
            data_blocks += 1
    confidence = min(90, max(45, int(50 + abs(score) * 8 + data_blocks * 6)))

    if news.get("status") == "no_data":
        caveats.append("Peu de news exploitables sur la fenêtre récente.")
    if perception.get("alpha_vantage", {}).get("status") == "skipped":
        caveats.append("Source Alpha Vantage non activée (optionnel).")

    if not comparisons:
        comparisons.append("Signaux globalement équilibrés, sans divergence majeure.")

    return {
        "score": score,
        "recommendation": recommendation,
        "confidence": confidence,
        "comparisons": comparisons,
        "caveats": caveats,
        "disclaimer": DISCLAIMER,
        "score_steps": [asdict(s) for s in steps],
    }


def _build_plan(config: AppConfig) -> list[str]:
    plan = [
        "Collecter les signaux techniques (prix, variation, SMA, RSI).",
        "Ajouter les fondamentaux pour estimer valorisation et robustesse financière.",
        "Intégrer le contexte news/sentiment et identifier convergences/divergences.",
        "Pondérer les signaux puis produire Achat / Conserver / Vente avec confiance.",
    ]
    if not config.llm_enabled:
        plan.append("Mode dégradé: synthèse générée sans LLM (règles explicites).")
    return plan


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return str(content)


def _llm_commentary(
    ticker: str,
    perception: dict[str, Any],
    heuristic: dict[str, Any],
    config: AppConfig,
) -> tuple[str | None, str | None, str | None]:
    if not config.llm_enabled:
        return None, "Aucun backend LLM activé: commentaire LLM non généré.", None

    model = None
    backend = None
    if config.groq_api_key:
        try:
            from langchain_groq import ChatGroq

            backend = "groq"
            model = ChatGroq(
                model=config.groq_model,
                temperature=0.1,
                api_key=config.groq_api_key,
            )
        except Exception as exc:
            return None, f"Groq indisponible: {exc}", None
    elif config.mistral_api_key:
        try:
            from langchain_mistralai import ChatMistralAI

            backend = "mistral"
            model = ChatMistralAI(
                model=config.mistral_model,
                temperature=0.1,
                api_key=config.mistral_api_key,
            )
        except Exception as exc:
            return None, f"Mistral AI indisponible: {exc}", None
    elif config.openai_api_key:
        backend = "openai"
        model = ChatOpenAI(
            model=config.openai_model,
            temperature=0.1,
            api_key=config.openai_api_key,
        )
    elif config.ollama_enabled:
        try:
            from langchain_ollama import ChatOllama

            backend = "ollama"
            model = ChatOllama(
                model=config.ollama_model,
                base_url=config.ollama_base_url,
                temperature=0.1,
            )
        except Exception as exc:
            return None, f"Ollama indisponible: {exc}", None
    else:
        return None, "Aucun LLM configuré (GROQ_API_KEY, OPENAI_API_KEY ou Ollama requis).", None

    try:
        agent = create_react_agent(
            model=model,
            tools=[
                get_market_snapshot,
                get_fundamentals_context,
                get_news_context,
                get_alpha_vantage_overview,
            ],
            prompt=SYSTEM_PROMPT,
        )
        result = agent.invoke(
            {"messages": [{"role": "user", "content": build_user_prompt(ticker, perception, heuristic)}]},
            config={"recursion_limit": 6},
        )
        message = result["messages"][-1]
        return _extract_text(getattr(message, "content", "")), None, backend
    except Exception as exc:
        return None, f"LLM indisponible ({backend}): {exc}", backend


def analyze_stock(ticker: str, config: AppConfig) -> dict[str, Any]:
    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("Ticker obligatoire.")
    if not VALID_TICKER_PATTERN.match(symbol):
        raise ValueError("Format ticker invalide (caractères autorisés: lettres, chiffres, point, tiret).")

    perception, actions = _collect_perception(symbol, config)
    heuristic = _score_recommendation(perception)
    llm_text, llm_error, llm_backend = _llm_commentary(symbol, perception, heuristic, config)
    warnings = []
    if llm_error:
        warnings.append(llm_error)
    if llm_backend == "groq":
        warnings.append(f"LLM actif: Groq ({config.groq_model}) — open-source, gratuit.")
    elif llm_backend == "mistral":
        warnings.append(f"LLM actif: Mistral AI ({config.mistral_model}) — open-source, gratuit.")
    elif llm_backend == "ollama":
        warnings.append(f"LLM actif: Ollama ({config.ollama_model}) — mode local gratuit.")
    elif llm_backend == "openai":
        warnings.append(f"LLM actif: OpenAI ({config.openai_model}).")
    if not config.newsapi_enabled:
        warnings.append("NEWSAPI_KEY absente: fallback automatique sur les news yfinance.")

    return {
        "ticker": symbol,
        "perceive": perception,
        "plan": _build_plan(config),
        "act": [asdict(action) for action in actions],
        "observe": {
            "recommendation": heuristic["recommendation"],
            "confidence": heuristic["confidence"],
            "score": heuristic["score"],
            "comparisons": heuristic["comparisons"],
            "caveats": heuristic["caveats"],
            "score_steps": heuristic["score_steps"],
        },
        "llm_commentary": llm_text,
        "llm_backend": llm_backend,
        "warnings": warnings,
    }

