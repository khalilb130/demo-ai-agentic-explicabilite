# Demo AI agentic explicabilite (Streamlit)

Demo pedagogique d'un agent d'analyse d'actions avec un flux explicable en 4 etapes :
**Perceive -> Plan -> Act -> Observe**.

## Ce que fait l'app

- Collecte des signaux marche/fondamentaux/news (sources gratuites en priorite)
- Expose chaque action outil et chaque signal de scoring
- Produit une recommandation finale : **Achat / Conserver / Vente** avec confiance
- Fonctionne en mode degrade sans LLM (heuristique uniquement)

## Stack

- Streamlit
- LangChain / LangGraph
- yfinance (gratuit)
- NewsAPI / Alpha Vantage (optionnels)
- LLM optionnels : Groq, Mistral, OpenAI, Ollama local

## Installation locale

```bash
pip install -r requirements.txt
```

## Configuration

1. Copier l'exemple :

```bash
cp .env.example .env
```

2. Renseigner uniquement les cles que vous voulez utiliser (tout est optionnel sauf selon votre cas d'usage).

## Lancement

```bash
streamlit run app.py
```

## Deploiement Streamlit Community Cloud

1. Pousser ce repo sur GitHub (deja fait)
2. Aller sur https://share.streamlit.io
3. **New app** -> selectionner le repo, branche `main`, fichier `app.py`
4. Ajouter les secrets dans **App settings -> Secrets** (ne pas les mettre dans le code)
5. Deploy

## Securite des cles API

- Ne jamais commiter de cles API
- `.env` et `.env.*` sont ignores par `.gitignore` (`.env.example` reste versionne)
- Pour la prod/cloud, utiliser les secrets de la plateforme (Streamlit Secrets)

