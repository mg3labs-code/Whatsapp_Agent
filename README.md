# WASA — WhatsApp AI Sales Agent

B2B pharmaceutical export assistant for New Life Medicare. Handles pricing, FAQ, orders, lead qualification, and human escalation over WhatsApp.

## Stack

FastAPI · LangGraph · GPT-4o / GPT-4o-mini · Pinecone · PostgreSQL · Redis · Railway

## Documentation (source of truth)

| Doc | Purpose |
|-----|---------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Message pipeline, session, escalation, deployment |
| [docs/AGENTS.md](docs/AGENTS.md) | Per-agent behavior, guardrails, router rules |
| [docs/SCENARIOS.md](docs/SCENARIOS.md) | UAT / integration test scenarios |
| [docs/PRODUCT_IMPORT.md](docs/PRODUCT_IMPORT.md) | Catalog import (`products` table) |

> **Runtime behavior:** always verify against `app/` code. Docs are updated to match the codebase; if they diverge, the code wins.

## Quick start

```bash
cd wasa
pip install -r requirements.txt
# Set DATABASE_URL, REDIS_URL, OPENAI_API_KEY, PINECONE_API_KEY, WhatsApp tokens in .env
alembic upgrade head
uvicorn app.main:app --reload
```

## Key env vars

- `LEADS_ALERT_PHONE_NUMBERS` — escalation / lead WhatsApp alerts
- `ORDER_ALERT_PHONE_NUMBERS` — new order alerts
- `FAQ_PINECONE_MIN_SCORE` — FAQ similarity floor (default 0.41)

## Tests

```bash
pytest tests/
```
