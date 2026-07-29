# WASA Agent Reference
# Specifications aligned with live code in `app/agents/` and `app/guardrails/`

## AGENT 1: PRICING (`app/agents/pricing.py`)

**Model:** GPT-4o with function/tool calling  
**Input:** `message`, `session`, `db`  
**Output:** `str` (WhatsApp-ready reply)

### Tool: `get_product_by_name(query)`
- Fuzzy match on `Product.product_name`, `salt_name`, `manufacturing_company` (ILIKE)
- Returns catalog dict with `price_per_strip` (USD), or:
  - `{"error": "product_not_found"}`
  - `{"error": "product_restricted", "name", "schedule_category"}` when `is_restricted=True`
- Max **3** tool calls per turn

### Restricted products
Pre-guardrails do **not** block drug names in free text. Restriction is **catalog-level**: pricing tool returns `product_restricted`; the LLM is instructed to say the product is not available for export via this channel.

### Qualification gate
Order and pricing require `session.lead_qualified=True`. FAQ does not.

---

## AGENT 2: FAQ / RAG (`app/agents/faq.py`)

**Model:** GPT-4o-mini (only when qualifying chunks exist)  
**Input:** `message`, `phone`, `session`  
**Output:** `(reply_text, updated_session, next_intent)` where `next_intent` is `"faq"` or `"escalate"`

### RAG process
1. Optional shortcut: "ship/deliver/send to {country}" → shipping lookup (no Pinecone)
2. Embed message → `text-embedding-3-small`
3. Pinecone query `top_k=3`; use chunks with score **>** `FAQ_PINECONE_MIN_SCORE` (default **0.41**)
4. No qualifying chunks → increment `faq_miss_count`, return `NO_CONTEXT_REPLY` (1st miss) or escalate (2nd miss)
5. With chunks → GPT-4o-mini grounded on chunk text only

### Miss counter (`faq_miss_count`)
Consecutive failures that increment toward escalation (threshold **2**):
- No Pinecone chunks above threshold
- LLM empty reply or soft no-answer (matches markers like "don't have specific information")
- Missing API keys / Pinecone or OpenAI errors

Reset to **0** on a successful grounded answer. Cleared on human handoff resume and main menu (`clear_human_handoff`).

### Buyer copy
**1st miss** (`NO_CONTEXT_REPLY`):
> I don't have specific information on that in our knowledge base. Could you rephrase, or type *speak to team* if you'd like a specialist?

**2nd miss:** empty reply from FAQ; `next_intent="escalate"`, `escalation_reason="faq_no_match_repeated"`. Orchestrator routes to **escalation_agent** (same path as speak-to-team). Escalation owns the handoff message — no duplicate FAQ text.

### System prompt
Live prompt: `FAQ_SYSTEM_PROMPT` in `faq.py` (must match `NO_CONTEXT_REPLY` when context is insufficient).

---

## AGENT 3: ORDER (`app/agents/order.py`)

**Model:** GPT-4o-mini when `ORDER_AGENT_USE_LLM=true` (default); rule-based fallback otherwise  
**Input:** `message`, `session`, `db`  
**Output:** `(reply_text, updated_session)`

### State machine (high level)
```
COLLECT_SKU → COLLECT_SKU_CONFIRM → COLLECT_QTY → CART_MENU
  → (add/edit/remove/qty) → checkout path
COLLECT_COUNTRY → COLLECT_CITY → COLLECT_CONTACT
  → SHIPPING_CHOICE (EMS/LP or PENDING_QUOTE)
  → COLLECT_CHECKOUT → CONFIRM_ORDER → SELECT_PAYMENT → ORDER_COMPLETE
```

Session keys include: `order_state`, `order_cart`, `order_country`, `order_city`, `order_contact`, shipping fields, payment selection.

On commit: `order_ref`, DB rows, `send_order_alert` to order team WhatsApp numbers, clear order session keys.

---

## AGENT 4: QUALIFICATION (`app/agents/qualification.py`)

**Model:** None — rule-based + WhatsApp interactive lists  
**Input:** `message`, `session`, `db`  
**Output:** `(reply_text, updated_session, next_intent)`

### State machine (current — 2 steps)
```
COLLECT_COUNTRY  → country picker / typed country; excluded → escalate
COLLECT_BIZ_TYPE → interactive list or typed biz type
QUAL_COMPLETE    → score_lead(), persist lead, route by score
```

Legacy states (`COLLECT_COMPANY`, `COLLECT_VOLUME`, `COLLECT_LICENSE`) normalize to `COLLECT_COUNTRY`.

### Lead scoring
Uses `app/agents/lead_scoring.score_lead()` (client SOP, 0–100). At `QUAL_COMPLETE`:
- **Disqualified** → escalate (compliance message)
- **manual_review_only** → escalate (`manual_review`)
- **score ≥ 80** (`HOT_LEAD_MIN_SCORE`) → escalate (`hot_lead`)
- Else → `pending_intent` handoff (pricing/order/faq) or main menu prompt

---

## AGENT 5: ESCALATION (`app/agents/escalation.py`)

**Model:** None  
**Input:** `message`, `session`, `reason`, `phone`  
**Output:** `(reply_text, updated_session)`

### Side effects
- `session.human_active = True`
- `session.escalation_reason = reason`
- `send_escalation_alert()` → WhatsApp to `LEADS_ALERT_PHONE_NUMBERS`

### Buyer templates
**In hours:** connecting with sales team, 30–60 min ETA, `exports@newlifemedicare.com`  
**Off hours:** team offline, next open time, query flagged priority

---

## ROUTER (`app/agents/router.py`)

**Classifier:** GPT-4o-mini JSON `{intent, confidence}`

### Early exits (no LLM)
- HUMAN_KEYWORDS / speak-to-team phrases → `escalate`
- Discount request → `escalate` + `escalation_reason=discount_request`
- Menu button `speak` → `escalate`

### Qualified leads
- Classifier confidence **< 0.45** → increment `clarification_count`; route FAQ once, escalate on **2nd** low-confidence turn
- `intent=qualify` from classifier → mapped to `faq` (never re-trap qualified buyers)

### Unqualified leads
- `faq` → allowed immediately
- `pricing` / `order` → `qualify` with `pending_intent`

---

## GUARDRAILS (`app/guardrails/check.py`)

### Pre-LLM (`check_pre_guardrails`)
1. `session.disqualified` or `lifecycle_stage == disqualified` → block
2. `session.country` in shipment-excluded list → block

**Removed:** blanket schedule-drug substring pre-filter on inbound messages. Restriction is catalog-level via pricing only.

`faq_miss_count`, `clarification_count`, and `clarification_attempts` are cleared on human handoff resume, main menu, and order cancel (`clear_human_handoff` / `clear_conversation_counters`).

### Post-LLM (`check_post_guardrails`)
Blocks **clinical dosing advice**:
1. Regex: imperative/frequency dosing (e.g. "take 500mg twice daily") — no topic word required
2. OR `BLOCKED_TOPICS` phrase within ±80 characters of `\d+ mg|ml|mcg`

**Passes:** topic without dose ("prescription required", "side effects include nausea"); product strength only ("Metformin 500mg strips").

### Refusal messages (exact strings in code)
- **Sanctioned / disqualified:** `REFUSAL_SANCTIONED_COUNTRY`
- **Restricted product (pricing path):** `REFUSAL_RESTRICTED_PRODUCT` — used when agents/guardrails surface catalog restriction, not pre-filter
- **Clinical content:** `REFUSAL_CLINICAL_CONTENT`

### Logging
Every block → `guardrail_logs` table; `message_text` capped at 200 chars.

---

## ORCHESTRATOR HANDOFFS (`app/orchestrator/graph.py`)

| From | Condition | To |
|------|-----------|-----|
| `qualify_agent` | `next_intent != continue_qual` | order / pricing / faq / escalate |
| `faq_agent` | `next_intent == escalate` | `escalation_agent` (FAQ reply cleared) |
| `router` | `human_active` | `human_active` node |
| All agents | normal | `post_guardrails` → `send_reply` |

---

## SCENARIO EXAMPLES

### A — New buyer asks price
Unqualified + pricing intent → qualify (country + biz type) → score → pricing agent with DB tool.

### B — Multi-turn order
Qualified → order state machine → cart → shipping → confirm → team order alert.

### C — Hot lead
Qual complete, score ≥ 80 → escalate + `human_active` + team WhatsApp alert.

### D — FAQ miss loop (fixed)
Miss 1 → honest no-context + rephrase hint. Miss 2 → escalation handoff only (no duplicate connect copy).

### E — Restricted product
Buyer asks price for `is_restricted=True` catalog row → pricing tool returns `product_restricted` → agent refuses export via channel (not pre-guardrail block).

### F — Speak to team
Keyword/button → router `escalate` → escalation_agent (no classifier call).
