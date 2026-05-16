# WASA Architecture Reference
# WhatsApp AI Sales Agent — New Life Medicare

## FULL MESSAGE PIPELINE (Every message goes through this exact sequence)

```
WhatsApp Buyer
     │
     ▼
Meta Cloud API v18
     │  POST /webhook (JSON payload)
     ▼
FastAPI Webhook Handler
     │  Return HTTP 200 INSTANTLY
     │  Then: BackgroundTasks.add_task(process_message)
     ▼
parse_meta_payload()
     │  Extract: phone, text, message_id
     │  If None (status update, not a message) → DROP SILENTLY
     ▼
Deduplication (Redis)
     │  SISMEMBER wasa:processed_ids message_id
     │  If exists → DROP SILENTLY (prevent double-processing on Meta retries)
     │  If new → SADD (24h TTL)
     ▼
LangGraph compiled_graph.ainvoke(state)
     │
     ├── Node 1: load_session_node
     │     Redis GET session:{phone} → state.session (dict, default {})
     │
     ├── Node 2: pre_guardrails_node (NO LLM — pure rule check)
     │     Check: state.session["country"] in SANCTIONED_COUNTRIES → BLOCK
     │     Check: state.message contains HARD_BLOCKED_PRODUCTS → BLOCK
     │     If blocked: state.guardrail_blocked=True, state.final_reply=refusal, log to DB
     │
     ├── Node 3: router_node (GPT-4o-mini for classification)
     │     Step 1: HUMAN_KEYWORDS check → "escalate" (no LLM)
     │     Step 2: Off-hours check → immediate off_hours response
     │     Step 3: LLM intent classification (JSON mode)
     │     Step 4: New lead + pricing/order intent → override to "qualify"
     │     Step 5: Confidence < 0.65 → override to "escalate"
     │
     ├── Conditional routing (route_to_agent function)
     │     guardrail_blocked=True → send_reply
     │     session.human_active=True → human_active (no-op → END)
     │     intent=pricing → pricing_agent_node
     │     intent=faq → faq_agent_node
     │     intent=order → order_agent_node
     │     intent=qualify → qualify_agent_node
     │     intent=escalate → escalation_agent_node
     │
     ├── [One of 5 agent nodes executes]
     │     Sets: state.agent_response = reply string
     │     Sets: state.session = updated session dict
     │
     ├── Node: post_guardrails_node (LLM response safety check)
     │     Check: state.agent_response contains BLOCKED_TOPICS → replace with refusal
     │     Otherwise: state.final_reply = state.agent_response
     │
     └── Node: send_reply_node
           send_message(phone, final_reply) → Meta Cloud API
           save_session(phone, session) → Redis (24h TTL)
           → END
```

## AGENT RESPONSIBILITIES

| Agent | Intent | LLM | Key Data Source | Multi-turn |
|-------|--------|-----|-----------------|------------|
| Pricing | pricing | GPT-4o (tool calling) | PostgreSQL products table | No |
| FAQ/RAG | faq | GPT-4o-mini | Pinecone vector index | No |
| Order | order | None (rule-based) | PostgreSQL orders table | Yes (6 steps) |
| Qualification | qualify | None (rule-based) | PostgreSQL leads table | Yes (5 steps) |
| Escalation | escalate | None | Slack, Meta Business Inbox | No |

## SESSION SCHEMA (Redis JSON, 24h TTL)

```json
{
  "lead_qualified": true,
  "lead_score": 75,
  "company": "Al Noor Pharmaceuticals LLC",
  "country": "UAE",
  "business_type": "distributor",
  "annual_volume_usd": 300000,
  "license_number": "UAE-MOHAP-2024-0491",
  "human_active": false,
  "pending_intent": "pricing",

  "qual_state": "QUAL_COMPLETE",

  "order_state": "COLLECT_QTY",
  "order_sku": "AMX-500-10",
  "order_product_name": "Amoxicillin 500mg",
  "order_qty": null,
  "order_country": null,
  "order_city": null,
  "order_contact": null,
  "order_payment": null
}
```

## ESCALATION TRIGGERS (ANY ONE → escalation_agent)

1. HUMAN_KEYWORDS in message (keyword list, no LLM)
2. Intent classification confidence < 0.65 (for qualified leads only)
3. Lead score > 85 immediately after qualification
4. session.human_active = True (silently drop — no reply at all)
5. FAQ agent returns no relevant context (redirected internally)

## OFF-HOURS BEHAVIOR

- business_hours: Mon–Sat, 10:00–20:00 IST (configurable via env); Sunday limited; AI 24/7
- If message received outside hours:
  1. Do NOT run any agent
  2. Reply with: off-hours message + next business open time
  3. Fire Slack alert to sales team (they may choose to respond manually)
  4. Do NOT set human_active (AI resumes next business day automatically)

## RAILWAY DEPLOYMENT ARCHITECTURE

```
GitHub (main branch)
     │ auto-deploy on push
     ▼
Railway Service (FastAPI)
     │ DATABASE_URL from Railway env
     ├── PostgreSQL Service (Railway managed)
     │     tables: products, leads, orders, guardrail_logs
     │
     │ REDIS_URL from env (external)
     ├── Upstash Redis (external managed)
     │     keys: session:{phone}, wasa:processed_ids
     │
     │ Pinecone (external)
     └── Vector index: wasa-faq (1536 dims, cosine)

Meta Cloud API ←→ Railway Service (/webhook endpoint)
Langfuse ← Railway Service (traces all LLM calls)
Slack ← Railway Service (escalation + order alerts)
```
