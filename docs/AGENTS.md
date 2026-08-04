# WASA Agent Reference
# Specifications aligned with live code in `app/agents/` and `app/guardrails/`

## AGENT 1: PRICING (`app/agents/pricing.py`)

**Model:** GPT-4o with function/tool calling (single-product free text)  
**Input:** `message`, `session`, `db`  
**Output:** `(reply_text, updated_session, next_intent)` where `next_intent` is `"pricing"` or `"escalate"`

### Multi-product lists (no LLM)
When the message looks like a bulk/list (`looks_like_bulk_order` — commas, newlines, or `Name - qty`):
- Parse lines with `parse_bulk_order_lines` (same as order agent)
- Resolve each line with order’s `_resolve_product_match` (token fallback)
- Return deterministic USD-per-strip quotes (+ line totals when qty present)
- Restricted → channel refusal note; unmatched → “Couldn't match” + **Did you mean…** suggestions
- Quantity is **optional** for quotes; default is per-strip price only

### Tool: `get_product_by_name(query)` (single-product LLM path)
- Same resolve as order: ILIKE on trade/salt/manufacturer, then token fallback (`lookup_product_ilike` + `_resolve_product_match`)
- Returns catalog dict with `price_per_strip` (USD), or:
  - `{"error": "product_not_found", "query", "suggestions"}`
  - `{"error": "product_restricted", "name", "schedule_category"}` when `is_restricted=True`
- Max **3** tool calls per turn
- Strong system prompt: extract clean search string, never invent prices, surface suggestions on miss

### Miss counter (`pricing_miss_count`)
**1st consecutive catalog miss:** suggest-on-miss reply (no escalate).  
**2nd miss:** empty reply; `next_intent="escalate"`, `escalation_reason="pricing_no_match_repeated"`. Orchestrator routes to **escalation_agent**.  
Cleared on successful quote / restricted catalog hit, menu, handoff resume, and order cancel (`clear_conversation_counters`).

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

### State machine (primary buyer flow)
```
COLLECT_SKU → COLLECT_SKU_CONFIRM → COLLECT_QTY → CART_MENU
  → (add / edit / remove / qty) → checkout
COLLECT_CHECKOUT  → Name, City (WhatsApp phone; reuses session.country)
  → SHIPPING_CHOICE (EMS / LP) or PENDING_QUOTE (desk quote; no wire yet)
  → CONFIRM_ORDER → wire confirm (one bubble: total + T/T details + buttons)
```

**Legacy / LLM fallback only** (not the primary interactive path): `COLLECT_COUNTRY`, `COLLECT_CITY`, `COLLECT_CONTACT`. Prefer `COLLECT_CHECKOUT` + session country.

Session keys include: `order_state`, `order_cart`, `order_country`, `order_city`, `order_contact`, shipping fields, payment selection.

On commit: `order_ref`, DB rows, `send_order_alert` to order team WhatsApp numbers; buyer gets a **single** WhatsApp interactive message (confirm + total + wire + New Order / Order Status / Speak).

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
- **score ≥ 80** (`HOT_LEAD_MIN_SCORE`) → team alert only (`hot_lead`); bot services stay open (not `human_active`)
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

### When the team is alerted (leads / order desk)

| Trigger | Alert | Channel |
|---------|-------|---------|
| Speak to team / human keywords / discount | Yes (via escalation_agent) | Leads |
| Hot lead (score ≥ 80) after qual | Yes (alert only — bot services stay open) | Leads |
| FAQ / pricing 2nd consecutive miss | Yes | Leads |
| Pricing API outage/error | Yes | Leads |
| Hot lead / manual review / disqualified / excluded country | Yes | Leads |
| Order checkout / set shipping to excluded country | Yes | Leads |
| FAQ “ship to X” with no rates | Yes (auto, no speak gate) | Leads |
| FAQ / guardrail excluded-country inquiry | Yes | Leads |
| Order cart shipping rates unavailable | Yes (draft order, **no wire**) | Order desk |
| New order confirmed | Yes | Order desk |
| human_active follow-up (already escalated) | No (hold only) | — |

### Buyer templates
**In hours:** connecting with export team, soft follow-up (no minute/hour SLA), `exports@newlifemedicare.com`  
**Off hours:** team offline, next open time, query flagged priority  
**Reason-specific:** hot lead / manual review / disqualified / pricing outage use one dedicated message (no double-merge with default escalate copy)  
**Speak menu:** selection header only; escalation owns the handoff body (no double “connecting”)

---

## ROUTER (`app/agents/router.py`)

**Layered design:** deterministic short-circuits → optional keyword fallback → LLM free-text classifier (with session context) → business policy.

**Classifier:** GPT-4o-mini JSON `{intent, confidence}` — used for free text only (not bare greetings / buttons).

### Early exits (no LLM)
- HUMAN_KEYWORDS / speak-to-team phrases → `escalate`
- Discount request → `escalate` + `escalation_reason=discount_request`
- Menu button `speak` → `escalate`
- **Pure greeting** (`hi` / `hello` / … with no pricing/order/FAQ substance) → `menu_refresh` (disclosure once per Redis session + main menu; never FAQ miss)
- **Main menu escape** (`main_menu` button or free text `menu` / `main menu` / `show menu` / `options`) → `menu_refresh` even mid-order / mid-qual (clears unfinished `qual_state` so buyers are not re-trapped)
- Mid-qual **Speak to Team** / **FAQs** short-circuit to escalate / faq (not stuck in country picker)
- Mixed greeting + request (e.g. `hi, price for metformin`) → LLM → primary intent
- **Catalog product name** (DB match, e.g. `KLENSMART 60MG`) → `pricing` (or `qualify` + `pending_intent=pricing` if unqualified). Skipped when message is clearly FAQ-process (`ship`/`documents`/…) or clear order request.

### Qualified leads
- Classifier confidence **< 0.45** → increment `clarification_count`; route FAQ once, escalate on **2nd** low-confidence turn
- `intent=qualify` from classifier → mapped to `faq` (never re-trap qualified buyers; greetings already exited to menu)

### Unqualified leads
- `faq` → allowed immediately
- `pricing` / `order` → `qualify` with `pending_intent`
- Pure greeting → `menu_refresh` (menu first; qual starts when they pick pricing/order)

---

## GUARDRAILS (`app/guardrails/check.py`)

### Pre-LLM (`check_pre_guardrails`)
1. `session.disqualified` or `lifecycle_stage == disqualified` → block
2. `session.country` in shipment-excluded list → block
3. Soft inbound clinical: obvious dosing / “how do I take” asks → short clinician / Speak-to-Team refusal (**no RAG**). Skipped when the message looks commercial (price, quote, order, cart, strip, etc.) so “Metformin 500mg price” still reaches pricing/FAQ.

Restricted product control is dual-source:
- pre-check term list (`restricted_terms`, from schedule workbook), and
- catalog row restriction (`products.is_restricted`) when a sellable row exists.

`faq_miss_count`, `pricing_miss_count`, `clarification_count`, and `clarification_attempts` are cleared on human handoff resume, main menu, and order cancel (`clear_human_handoff` / `clear_conversation_counters`).

### Post-LLM (`check_post_guardrails`) — keep these (most important)
Blocks **clinical dosing advice** in outbound replies:
1. Regex: imperative/frequency dosing (e.g. "take 500mg twice daily") — no topic word required
2. OR `BLOCKED_TOPICS` phrase within ±80 characters of `\d+ mg|ml|mcg`

**Passes:** topic without dose ("prescription required", "side effects include nausea"); product strength only ("Metformin 500mg strips").

### Refusal messages (exact strings in code)
- **Sanctioned / disqualified:** `REFUSAL_SANCTIONED_COUNTRY`
- **Restricted product (pricing path):** `REFUSAL_RESTRICTED_PRODUCT` — used when agents/guardrails surface catalog restriction, not pre-filter
- **Clinical outbound:** `REFUSAL_CLINICAL_CONTENT`
- **Clinical inbound (soft):** `REFUSAL_CLINICAL_INBOUND`

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
Qualified → cart → checkout (Name, City) → shipping → confirm → one wire bubble + team order alert.

### C — Hot lead
Qual complete, score ≥ 80 → team WhatsApp alert; buyer keeps pricing / order / FAQ (not locked to human-only).

### D — FAQ miss loop (fixed)
Miss 1 → honest no-context + rephrase hint. Miss 2 → escalation handoff only (no duplicate connect copy).

### E — Restricted product
Buyer asks price for `is_restricted=True` catalog row → pricing tool returns `product_restricted` → agent refuses export via channel (not pre-guardrail block).

### F — Speak to team
Keyword/button → router `escalate` → escalation_agent (no classifier call).
