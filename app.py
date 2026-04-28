from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stock_agent.agent import DISCLAIMER, analyze_stock
from stock_agent.config import load_config


def _status_badge(is_available: bool) -> str:
    return "✅ disponible" if is_available else "⚠️ non configuré"


def _render_perceive(result_data: dict) -> None:
    market = result_data.get("perceive", {}).get("market", {})
    fundamentals = result_data.get("perceive", {}).get("fundamentals", {})
    news = result_data.get("perceive", {}).get("news", {})

    st.subheader("Perceive")
    c1, c2, c3 = st.columns(3)
    c1.metric("Prix actuel", f"{market.get('current_price', 'N/A')}")
    c2.metric("Variation 1j (%)", f"{market.get('daily_change_pct', 'N/A')}")
    c3.metric("Signal tendance", market.get("trend", "N/A"))

    st.write("Résumé des signaux collectés :")
    st.json(
        {
            "market": market,
            "fundamentals": fundamentals,
            "news": news,
            "alpha_vantage": result_data.get("perceive", {}).get("alpha_vantage"),
        }
    )


def _render_plan(result_data: dict) -> None:
    st.subheader("Plan")
    for step in result_data.get("plan", []):
        st.markdown(f"- {step}")
    llm_text = result_data.get("llm_commentary")
    if llm_text:
        st.info(llm_text)


def _render_act(result_data: dict) -> None:
    st.subheader("Act")
    actions = result_data.get("act", [])
    if not actions:
        st.warning("Aucune action outil n'a été enregistrée.")
        return

    st.table(
        [
            {
                "outil": action["tool"],
                "statut": action["status"],
                "détail": action["detail"],
            }
            for action in actions
        ]
    )
    with st.expander("Voir les sorties détaillées des outils"):
        for action in actions:
            st.markdown(f"**{action['tool']}** ({action['status']})")
            st.json(action.get("output", {}))


def _render_observe(result_data: dict) -> None:
    observe = result_data.get("observe", {})
    recommendation = observe.get("recommendation", "Conserver")
    confidence = observe.get("confidence", 0)
    score = observe.get("score", 0)

    st.subheader("Observe")
    if recommendation == "Achat":
        st.success(f"Opinion finale : **{recommendation}**")
    elif recommendation == "Vente":
        st.error(f"Opinion finale : **{recommendation}**")
    else:
        st.warning(f"Opinion finale : **{recommendation}**")

    c1, c2 = st.columns(2)
    c1.metric("Confiance estimée", f"{confidence}%")
    c2.metric("Score agrégé", f"{score:+d}", delta=score, delta_color="normal")

    steps = observe.get("score_steps", [])
    if steps:
        st.write("**Trace de scoring signal par signal :**")
        rows = []
        for s in steps:
            delta = s.get("delta", 0)
            emoji = "🟢" if delta > 0 else ("🔴" if delta < 0 else "⚪")
            rows.append(
                {
                    " ": emoji,
                    "Signal": s.get("signal", ""),
                    "Valeur observée": s.get("value", ""),
                    "Δ score": f"{delta:+d}" if delta != 0 else "0",
                    "Interprétation": s.get("reason", ""),
                }
            )
        st.table(rows)

    st.write("Convergences / divergences notables :")
    for line in observe.get("comparisons", []):
        st.markdown(f"- {line}")

    st.write("Incertitudes / caveats :")
    for caveat in observe.get("caveats", []):
        st.markdown(f"- {caveat}")

    st.caption(DISCLAIMER)


def main() -> None:
    st.set_page_config(page_title="Agent boursier pédagogique", layout="wide")
    config = load_config()

    st.title("Démo Streamlit — Agent pédagogique d'analyse d'action")
    st.write(
        "Flux agentique visible : **Perceive → Plan → Act → Observe**. "
        "Les données viennent de sources gratuites en priorité."
    )

    with st.sidebar:
        st.header("Configuration")
        st.subheader("LLM (priorité)")
        st.write(f"1. Groq (open-source) : {_status_badge(bool(config.groq_api_key))}")
        if config.groq_api_key:
            st.caption(f"Modèle: {config.groq_model}")
        st.write(f"2. Mistral AI (open-source) : {_status_badge(bool(config.mistral_api_key))}")
        if config.mistral_api_key:
            st.caption(f"Modèle: {config.mistral_model}")
        st.write(f"3. OpenAI : {_status_badge(bool(config.openai_api_key))}")
        st.write(f"4. Ollama (local) : {_status_badge(config.ollama_enabled)}")
        if config.ollama_enabled:
            st.caption(f"Modèle local: {config.ollama_model}")
        st.subheader("Données optionnelles")
        st.write(f"NEWSAPI_KEY : {_status_badge(config.newsapi_enabled)}")
        st.write(f"ALPHA_VANTAGE_API_KEY : {_status_badge(config.alpha_vantage_enabled)}")
        st.caption("Sans LLM, l'app utilise les règles heuristiques (mode dégradé).")

    ticker = st.text_input("Ticker (ex: AAPL, MSFT, AIR.PA)", value="AAPL").upper().strip()
    launch = st.button("Analyser", type="primary")

    if not launch:
        st.stop()

    try:
        with st.spinner(f"Analyse de {ticker} en cours..."):
            result = analyze_stock(ticker=ticker, config=config)
    except ValueError as exc:
        st.error(f"Entrée invalide : {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Erreur lors de l'analyse : {exc}")
        st.stop()

    if result.get("warnings"):
        for warning in result["warnings"]:
            st.info(warning)

    _render_perceive(result)
    _render_plan(result)
    _render_act(result)
    _render_observe(result)


if __name__ == "__main__":
    main()
