# WASA Architecture Reference
# WhatsApp AI Sales Agent — New Life Medicare

This document reflects the **live codebase** (`app/orchestrator/graph.py`, agents, guardrails). When in doubt, trust the code.

## FULL MESSAGE PIPELINE

```
WhatsApp Buyer
     │
     ▼
Meta Cloud API
     │  POST /webhook (JSON payload)
     ▼
FastAPI webhook handler (app/webhook/router.py)
     │  Return HTTP 200 immediately
     │  BackgroundTasks → process_message
     ▼
parse_meta_payload() → phone, text, message_id
     │  Status-only payloads → drop
     ▼
Deduplication (Redis)
     │  wasa:processed_ids — drop retries
     ▼
LangGraph compiled_graph.ainvoke(state)
     │
     ├── load_session_node
     │     Redis GET session:{phone}; hydrate from Postgres if needed
     │
     ├── greeting_node
     │     First visit: session.greeted = True → welcome/disclosure on send
     │
     ├── pre_guardrails_node (no LLM)
     │     check_pre_guardrails(message, session)
     │     Blocks: disqualified lead, shipment-excluded country in session
     │     If blocked → final_reply = refusal, log guardrail_logs → send_reply
     │
     ├── router_node
     │     human_active + resume phrase → clear handoff
     │     disqualified session → intent escalate
     │     main menu / order mid-flow / qual mid-flow overrides
     │     else classify_intent() (GPT-4o-mini JSON + rules)
     │
     ├── Conditional routing (_route_to_agent)
     │     guardrail_blocked → send_reply
     │     human_active → human_active node
     │     intent → pricing | faq | order | qualify | escalate | menu_refresh
     │
     ├── [Agent node]
     │     Sets agent_response + updated session
     │     qualify / faq may set intent for same-turn handoff
     │
     ├── qualify_agent → post_guardrails OR re-route to order/pricing/faq/escalate
     ├── faq_agent → post_guardrails OR escalation_agent (2nd consecutive miss)
     ├── pricing_agent → post_guardrails OR escalate (repeated catalog miss)
     ├── order/escalation → post_guardrails
     │
     ├── post_guardrails_node
     │     check_post_guardrails(agent_response) — clinical dosing rules
     │     Blocked → final_reply = refusal; else final_reply = agent_response
     │
     └── send_reply_node
           send_message(phone, final_reply)
           save_session(phone, session) → Redis
           persist conversation summaries → Postgres
           → END

human_active node → sends hold + Continue with Bot buttons → END (no agent)
menu_refresh → resend main menu → send_reply
```

## AGENT RESPONSIBILITIES

| Agent | Intent | LLM | Data source | Multi-turn |
|-------|--------|-----|-------------|------------|
| Pricing | pricing | GPT-4o (tool calling) | PostgreSQL `products` | No |
| FAQ/RAG | faq | GPT-4o-mini | Pinecone `wasa-faq` | No (miss counter in session) |
| Order | order | GPT-4o-mini optional (`ORDER_AGENT_USE_LLM`); rule fallback | PostgreSQL orders/products/shipping | Yes |
| Qualification | qualify | None (rule-based + UI lists) | PostgreSQL `leads` | Yes (2 steps) |
| Escalation | escalate | None | WhatsApp team alerts | No |

## TEAM ALERTS

Escalations and new leads → **WhatsApp DMs** to `LEADS_ALERT_PHONE_NUMBERS` (fallback: `ESCALATION_PHONE_NUMBERS`).

New orders → `ORDER_ALERT_PHONE_NUMBERS`.

Implementation: `app/integrations/alerts.py` (not Slack).

## SESSION SCHEMA (Redis JSON, 24h TTL)

Key fields (not exhaustive):

```json
{
  "lead_qualified": true,
  "lead_score": 75,
  "lead_category": "warm",
  "company": "Acme Pharma",
  "country": "Kenya",
  "business_type": "distributor",
  "buyer_type": "distributor",
  "human_active": false,
  "escalation_reason": null,
  "faq_miss_count": 0,
  "pricing_miss_count": 0,
  "clarification_count": 0,
  "pending_intent": "pricing",
  "pending_query": "price for metformin",
  "qual_state": "COLLECT_BIZ_TYPE",
  "order_state": "COLLECT_SKU",
  "order_cart": [],
  "greeted": true,
  "last_agent": "faq"
}
```

## ESCALATION TRIGGERS (→ escalation_agent)

1. **Speak to team** / HUMAN_KEYWORDS / discount request (`router.classify_intent`)
2. **FAQ 2nd consecutive miss** — no Pinecone chunks, soft LLM no-answer, empty reply, or infra error (`faq_no_match_repeated`)
2b. **Pricing 2nd consecutive catalog miss** — product not found after suggestions (`pricing_no_match_repeated`)
3. **Qualified lead, classifier confidence < 0.45** twice in a row (`clarification_count` ≥ 2)
4. **Hot lead** after qualification — `lead_score >= 80` (`HOT_LEAD_MIN_SCORE`)
5. **Manual review** or **disqualified** paths from qualification scoring
6. **Excluded country** during qualification
7. **Disqualified session** re-contact (`router` → escalate)

When `human_active=True`, router sends to **human_active** node (hold message + resume buttons), not silent drop.

## OFF-HOURS BEHAVIOR

- Business hours: Mon–Sat **10:00–20:00 IST** (env: `BUSINESS_HOURS_START`, `BUSINESS_HOURS_END`, `BUSINESS_TIMEZONE`)
- **Agents still run** 24/7; only the **escalation buyer copy** changes off-hours
- Off-hours escalation: offline message + priority flag; still sets `human_active=True` and sends WhatsApp team alert

## GUARDRAILS (summary)

**Pre-LLM** (`check_pre_guardrails`):
- Disqualified lead
- Shipment-excluded country in session

Restricted products are enforced with two layers:
- **pre-check list** (`restricted_terms`) loaded from client schedule workbook, applied before catalog matching, and
- **catalog flags** (`products.is_restricted`) for rows that exist in sellable catalog data.

**Post-LLM** (`check_post_guardrails`):
- Imperative/frequency dosing (e.g. "take 500mg twice daily")
- OR `BLOCKED_TOPICS` phrase within ±80 chars of `\d+ mg|ml|mcg`
- Topic words alone (e.g. "prescription required", "side effects") **pass**
- Catalog strengths alone (e.g. "Metformin 500mg strips") **pass**

## DEPLOYMENT (Railway)

```
GitHub → Railway (FastAPI)
  ├── PostgreSQL (products, leads, orders, guardrail_logs, conversations)
  ├── Redis / Upstash (sessions, dedup)
  ├── Pinecone (wasa-faq index)
  └── Meta WhatsApp Cloud API webhook
Langfuse — LLM tracing (optional)
```
