from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = """Tu es un analyste financier pédagogique.
Objectif:
- Expliquer clairement les signaux de marché, fondamentaux et news.
- Comparer les signaux convergents/divergents.
- Donner une opinion finale: Achat, Conserver ou Vente.
- Donner un niveau de confiance en pourcentage et des caveats.
Contraintes:
- Réponds en français.
- Reste prudent, factuel, et transparent sur les incertitudes.
- Termine toujours par: "Ceci est un contenu éducatif, pas un conseil financier."
"""


def build_user_prompt(ticker: str, perceived_data: dict[str, Any], heuristic: dict[str, Any]) -> str:
    payload = {
        "ticker": ticker,
        "perceived_data": perceived_data,
        "heuristic": heuristic,
    }
    return (
        "Analyse ce contexte d'action boursière et reformule une synthèse pédagogique.\n"
        "Format attendu:\n"
        "1) Plan d'analyse (3-5 puces)\n"
        "2) Comparaison des signaux (technique, fondamentaux, news)\n"
        "3) Opinion finale (Achat/Conserver/Vente), confiance, caveats\n\n"
        f"Contexte JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}"
    )

