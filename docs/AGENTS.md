# WASA Agent Reference
# Complete specifications for all 5 agents + guardrails

## AGENT 1: PRICING AGENT (app/agents/pricing.py)
Model: GPT-4o with function/tool calling
Cost tier: High — use only for confirmed pricing queries

### Input
- message: buyer's message text
- session: dict (must contain "company" and "country" or agent asks for them)
- db: SQLAlchemy Session

### Tool: get_product_by_name(query: str)
- Fuzzy match on `Product.product_name`, `Product.salt_name`, and `Product.manufacturing_company` (ILIKE)
- Returns: product dict with `price_per_strip` (USD) OR `{"error": "product_not_found"}` OR `{"error": "product_restricted"}`
- NEVER call this more than 3 times per turn

### System prompt
The live system prompt is in `app/agents/pricing.py` (`PRICING_SYSTEM_PROMPT`). It instructs GPT-4o to call the tool first and quote the **single catalog USD price per strip** (no multi-tier DB fields).

### Output: formatted string ready to send to WhatsApp

---

## AGENT 2: FAQ/RAG AGENT (app/agents/faq.py)
Model: GPT-4o-mini
Cost tier: Low — cheap and fast for FAQ lookups

### Input
- message: buyer's question text

### RAG Process
1. Embed message → text-embedding-3-small → 1536-dim vector
2. Pinecone query: top_k=3, include_metadata=True
3. Filter: only use chunks with score > FAQ_PINECONE_MIN_SCORE (default **0.41**; tune **0.40–0.42** per `scripts/analyze_faq_thresholds` on your index)
4. If no chunks above threshold → return escalation message (do NOT call LLM)
5. Build context: join chunk texts with "\n\n"

### System Prompt (production-ready)
```
You are a helpful assistant for New Life Medicare pharmaceutical exports.
Answer the buyer's question using ONLY the provided context. Rules:
1. If the answer is not in the context, say: "I'll need to check on that and get back to you.
   Let me connect you with our team for this." — then stop, do not improvise.
2. Never fabricate shipping timelines, document requirements, or regulatory claims.
3. Use *asterisks* for bold (WhatsApp format).
4. Keep answers under 400 words.
5. End with a clarifying question if appropriate.
Context: {context}
```

### Output: grounded answer string OR escalation fallback message

---

## AGENT 3: ORDER COLLECTION AGENT (app/agents/order.py)
Model: GPT-4o-mini + function tools (rule-based fallback if `OPENAI_API_KEY` missing)
Session keys used: order_state, order_cart (list of lines), order_sku/order_product_name/order_moq
                   (while adding a line), order_country, order_city, order_contact, order_payment

### State Machine (COLLECT IN EXACT ORDER)
```
COLLECT_SKU → COLLECT_QTY → CART_MENU (repeat add / edit / remove / qty commands)
  CART_MENU: *add* another product | *done* checkout | *edit* | *remove 2* | *qty 1 500*

COLLECT_SKU
  Question: "I'll help you place your order! Which product would you like to add? (name or SKU)"
  Validate: product exists in DB (fuzzy search)
  On valid: save pending line → COLLECT_QTY

COLLECT_QTY
  Question: "How many units of {product_name}?"
  On valid: append/merge into order_cart → CART_MENU

COLLECT_COUNTRY
  Question: "Which country should we ship to?"
  Validate: country NOT in SANCTIONED_COUNTRIES (case-insensitive check)
  On sanctioned: return guardrail refusal, reset ALL order_* session keys
  On valid: save country → advance to COLLECT_CITY

COLLECT_CITY
  Question: "Which city or port of entry in {country}?"
  Validate: len(message.strip()) >= 2
  On valid: save city → advance to COLLECT_CONTACT

COLLECT_CONTACT
  Question: "Your name and company name for this order?"
  Validate: len(message.strip()) >= 3
  On valid: save contact → advance to COLLECT_PAYMENT

COLLECT_PAYMENT
  Question: "Preferred payment terms? (T/T Advance, Letter of Credit, or 30-day net)"
  On valid: save payment_terms → CONFIRM_ORDER (review summary)

CONFIRM_ORDER
  Show full cart + shipping + payment. User must reply *CONFIRM* (or yes/ok).
  *edit* returns to CART_MENU. On CONFIRM → commit.

ORDER_COMPLETE / commit
  1. order_ref = ORD-YYYYMMDD-####; one DB row per cart line (order_ref-L01, L02, …)
  2. Write Order rows + send_order_alert (all lines)
  3. Clear order_* session keys
  4. Return confirmation:
     "✅ *Order Confirmed!*
     📋 *Order Summary:*
     • Product: {product_name}
     • Quantity: {qty} units
     • Ship to: {city}, {country}
     • Contact: {contact}
     • Payment: {payment_terms}
     • Order Ref: {order_ref}
     Our sales team will contact you within 24 hours with the proforma invoice. Thank you!"
```

---

## AGENT 4: QUALIFICATION AGENT (app/agents/qualification.py)
Model: None — rule-based state machine with NLP extraction
Session keys: qual_state, company, country, business_type, annual_volume_usd, license_number

### State Machine (COLLECT IN EXACT ORDER)
```
COLLECT_COMPANY
  Question: "Welcome to New Life Medicare! To provide accurate pricing and ensure compliance
  with export regulations, could I get your company name and country?"
  Extract: company name from response (take full response if one line, else first sentence)
  Advance immediately to COLLECT_COUNTRY

COLLECT_COUNTRY
  Question: "Thank you! And which country are you based in?"
  Extract: country name (accept any non-empty response)
  Advance to COLLECT_BIZ_TYPE

COLLECT_BIZ_TYPE
  Question: "What type of business are you? (hospital, pharmaceutical distributor, pharmacy chain,
  independent pharmacy, or other)"
  Extract business type using keyword matching:
  - "hospital" or "clinic" or "medical center" → "hospital"
  - "distributor" or "wholesale" or "wholesaler" → "distributor"
  - "pharmacy chain" or "chain pharmacy" or "retail chain" → "pharmacy_chain"
  - "pharmacy" or "chemist" or "drugstore" → "pharmacy"
  - anything else → "other"
  Advance to COLLECT_VOLUME

COLLECT_VOLUME
  Question: "What is your approximate annual pharmaceutical purchase volume in USD?
  (For example: $50,000, $500,000, or $2 million)"
  Extract number using these patterns:
  - "$2 million" or "2m" or "2 mil" → 2000000
  - "$500k" or "500,000" or "500 thousand" → 500000
  - "$50,000" or "50000" → 50000
  - bare integer → use as-is
  On extraction failure: "Could you share the approximate USD amount? (e.g., $100,000)"
  Advance to COLLECT_LICENSE

COLLECT_LICENSE
  Question: "Do you hold a valid pharmaceutical import/distribution license? If yes, please
  share the license number. (This is optional)"
  If response contains alphanumeric code (length > 3): license_number = response
  If response is "no" / "don't have" / "not yet": license_number = None
  Advance to QUAL_COMPLETE

QUAL_COMPLETE
  1. Calculate lead_score using calculate_lead_score(session)
  2. Write Lead to DB
  3. Set session["lead_qualified"] = True, session["lead_score"] = score
  4. Clear qual_state from session
  5. If score > 85 → return next_intent = "escalate"
     Transition message: "Thank you for the information! Based on your business profile,
     I'd like to connect you with our Senior Export Manager directly..."
  6. If score <= 85 → return next_intent = session.pop("pending_intent", "faq")
     Transition message: "Thank you! I have everything I need. Let me now help with your query..."
```

### Lead Scoring (EXACT — do not change weights)
```python
score = 0
# Business type (max 30)
hospital → 30, distributor → 25, pharmacy_chain → 20, pharmacy → 10, other → 5

# Annual volume (max 30)
≥ $500,000 → 30
≥ $100,000 → 20
≥ $25,000  → 10
< $25,000  → 5

# License present (20)
license_number is not None → +20

# Country tier (max 20)
tier1 = [UAE, KSA, Saudi Arabia, Germany, UK, United Kingdom, USA, Singapore, Australia, Canada, France, Netherlands] → +20
tier2 = [Kenya, Nigeria, Tanzania, Bangladesh, Pakistan, Ghana, Ethiopia, Uganda, South Africa] → +10
other → +5

# Cap at 100
return min(score, 100)
```

---

## AGENT 5: ESCALATION AGENT (app/agents/escalation.py)
Model: None — rule-based
Side effects: Slack alert, session.human_active = True

### Triggers (ANY ONE)
- HUMAN_KEYWORDS detected (no LLM)
- Intent confidence < 0.65 (qualified leads only)
- lead_score > 85 after qualification
- Complaint/frustration keyword

### Response Templates

IN HOURS:
```
I'm connecting you with our sales team right now, {company or ""}!

Our team will reach out to you directly within the next 30–60 minutes.
For urgent matters, you can also reach us at exports@newlifemedicare.com

Reference your phone number when contacting us. 🙏
```

OFF HOURS:
```
Thank you for reaching out to New Life Medicare!

Our team is currently offline (Business hours: Mon–Sat, 10 AM – 8 PM IST). Our AI assistant is available 24/7.

Your query has been flagged as a priority and we'll get back to you on {next_business_open}.

For urgent inquiries: exports@newlifemedicare.com
```

### After sending reply
- session["human_active"] = True
- Send Slack alert with: phone (hashed for privacy), company, country, lead_score, reason
- Save session

---

## GUARDRAILS (app/guardrails/check.py)

### Pre-LLM Checks (run BEFORE any agent, zero API cost)
1. Country in SANCTIONED_COUNTRIES → BLOCK
2. Message contains HARD_BLOCKED_PRODUCTS keywords → BLOCK

### Post-LLM Checks (run AFTER agent response, catch clinical content)
1. Response contains BLOCKED_TOPICS words → REPLACE with refusal

### Refusal Messages (use these EXACTLY)
Sanctioned country: "I'm sorry, we're unable to process orders for that destination due to export compliance requirements. Please contact our compliance team directly."
Restricted product: "I'm unable to assist with that product query through this channel. Please contact our medical compliance team directly."
Clinical content: "I can't assist with that query. For medical or clinical questions, please consult a qualified healthcare professional."

### Logging
Every trigger → GuardrailLog entry in DB.
message_text field: truncated to 200 chars max.

---

## REAL-TIME SCENARIO EXAMPLES

### Scenario A: New buyer asks for price
Turn 1: "Hi, I need price for Amoxicillin 500mg, 10,000 strips"
  → session: {} → qualify_agent
  → Qual asks: company name + country

Turn 2: "Al Noor Pharmaceuticals LLC — Dubai, UAE"
  → qual collects company + country

Turn 3-5: business type, volume, license answers

Turn 6: QUAL_COMPLETE → lead_score: 75 → qualified → pricing_agent
  → Pricing Agent queries DB → returns formatted quote using USD per strip from catalog

### Scenario B: Multi-turn order placement
Turn 1: "I want to order"
  → intent: order → order_agent → asks for product name

Turn 2: "Metformin 500mg"
  → validates, saves SKU, asks for quantity

Turn 3: "0"
  → validates: not a positive integer → rejects, asks for a positive number

Turn 4: "2000"
  → valid → saves qty → asks for country

Turn 5-7: country (Kenya), city (Nairobi), contact (Priya Sharma, MedEx)

Turn 8: "T/T advance"
  → ORDER_COMPLETE → writes to DB → Slack alert → confirmation sent

### Scenario C: Immediate escalation (high value)
Turn 1-5: Qualification → "NHS Supply Chain — UK" + hospital + $2M volume
  → lead_score: 92 → ESCALATE
  → "Connecting you with our Senior Export Manager..."
  → Slack alert fired
  → session.human_active = True
