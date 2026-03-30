# Enterprise AI Demo (Production-Oriented Starter)

This project is a runnable FastAPI + browser chat app that uses your modular framework:

- `src/platform_ai/client.py`
- `src/tools/registry.py`
- `src/safety/pipeline.py`
- `src/retrieval/fusion.py`
- `src/prompts/registry.py`
- `src/telemetry/tracing.py`
- `src/infra/secrets.py`

It uses **real Azure/OpenAI integrations only**. There are **no mock or fake responses** in the codebase.

## What you need

- Python 3.12+
- An Azure OpenAI / Microsoft Foundry deployment with a model deployment name
- Optional but supported:
  - Azure AI Search service + index
  - Azure Cosmos DB for NoSQL database + container
  - Application Insights connection string

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` with your real Azure values.

## Run

```bash
python main.py
```

Open:

```text
http://127.0.0.1:8000
```

## Required environment variables

At minimum, set:

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_BASE_URL`
- `AZURE_OPENAI_CHAT_MODEL`

Example:

```env
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_BASE_URL=https://YOUR-RESOURCE.openai.azure.com/openai/v1/
AZURE_OPENAI_CHAT_MODEL=gpt-4o-mini
```

## Optional retrieval

If you also set Azure AI Search and Cosmos variables, the app will pull real retrieval context from those services.

If you leave them blank, the app still runs, but retrieval returns empty lists rather than fake data.

## Deploy

This is suitable to deploy to:

- Azure App Service (recommended first)
- AKS (if you need Kubernetes control)

