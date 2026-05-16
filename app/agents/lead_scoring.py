"""Client SOP: Lead Scoring & Buyer Prioritization (0–100).

Six core parameters (30+20+15+15+10+10) plus bonuses/penalties and overrides.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.business.countries import (
    classify_country,
    country_priority_points,
    is_priority_market,
    is_shipment_excluded_country,
)

# P6 → manual review only
P6_KEYWORDS = [
    "schedule x",
    "schedule h",
    "ndps",
    "controlled substance",
    "narcotic",
    "psychotropic",
    "tramadol",
    "buprenorphine",
    "ketamine",
]

P1_KEYWORDS = [
    "ed ",
    "erectile",
    "viagra",
    "sildenafil",
    "tadalafil",
    "hair loss",
    "finasteride",
    "weight loss",
    "semaglutide",
    "fertility",
    "cardiac",
    "blood pressure",
    "cholesterol",
    "diabetes",
    "metformin",
    "thyroid",
    "levothyroxine",
    "veterinary",
]

P2_KEYWORDS = [
    "antibiotic",
    "amoxicillin",
    "ciprofloxacin",
    "gastro",
    "omeprazole",
    "pain",
    "ibuprofen",
    "arthritis",
    "respiratory",
    "allergy",
    "eye ",
    "ent",
    "neuro",
]

P3_KEYWORDS = [
    "vitamin",
    "supplement",
    "nutraceutical",
    "otc",
    "ayurvedic",
    "herbal",
    "steroid",
    "peptide",
    "nootropic",
]

P5_KEYWORDS = [
    "cold chain",
    "temperature",
    "injectable",
    "ampoule",
    "vial",
    "oncology",
    "cancer",
    "hiv",
    "hepatitis",
    "critical care",
    "rare medicine",
]

RISKY_PRODUCT_KEYWORDS = P6_KEYWORDS + [
    "tramadol",
    "codeine",
    "opioid",
    "benzodiazepine",
    "xanax",
]

@dataclass
class LeadScoreResult:
    score: int
    category: str  # hot | warm | low_priority | ignore
    manual_review_only: bool
    disqualified: bool
    breakdown: dict[str, int] = field(default_factory=dict)


def classify_lead_score(score: int) -> str:
    if score >= 80:
        return "hot"
    if score >= 60:
        return "warm"
    if score >= 40:
        return "low_priority"
    return "ignore"


def map_business_type_to_buyer_type(business_type: str | None, message: str = "") -> str:
    """Map WASA qualification business_type + keywords to SOP buyer type."""
    text = f"{business_type or ''} {message}".lower()
    if any(k in text for k in ("suspicious", "scam", "fake")):
        return "suspicious"
    if any(k in text for k in ("repeat customer", "ordered before", "previous order")):
        return "repeat_customer"
    if any(
        k in text
        for k in ("distributor", "wholesale", "wholesaler", "resell", "bulk", "my clients")
    ):
        return "distributor"
    if any(k in text for k in ("doctor", "physician", "prescriber", "medical professional")):
        return "doctor"
    if any(
        k in text
        for k in (
            "pharmacy",
            "clinic",
            "hospital",
            "chemist",
            "drugstore",
            "medical center",
            "pharmacy chain",
        )
    ):
        return "pharmacy_clinic"
    return "new_individual"


def score_buyer_type(buyer_type: str) -> int:
    weights = {
        "distributor": 30,
        "pharmacy_clinic": 25,
        "doctor": 20,
        "repeat_customer": 15,
        "new_individual": 10,
        "suspicious": 0,
    }
    return weights.get(buyer_type, 10)


def score_order_value_usd(value: float | None) -> int:
    if value is None or value <= 0:
        return 5
    if value > 500:
        return 20
    if value >= 200:
        return 18
    if value >= 100:
        return 15
    if value >= 50:
        return 10
    return 5


def annual_volume_to_order_value_estimate(annual_usd: float | None) -> float | None:
    """Proxy typical order value from annual purchase volume when per-order not collected."""
    if not annual_usd or annual_usd <= 0:
        return None
    if annual_usd >= 2_000_000:
        return 600.0
    if annual_usd >= 500_000:
        return 400.0
    if annual_usd >= 100_000:
        return 150.0
    if annual_usd >= 50_000:
        return 75.0
    if annual_usd >= 25_000:
        return 50.0
    return 25.0


def score_country(country: str | None) -> tuple[int, bool, bool]:
    """Returns (points, is_shipment_excluded, is_compliance_risk_market)."""
    tier = classify_country(country)
    if tier == "excluded":
        return 0, True, True
    if tier == "missing":
        return 0, False, False
    points = country_priority_points(country)
    compliance_risk = tier in ("secondary", "other")
    return points, False, compliance_risk


def detect_product_tiers(text: str) -> list[str]:
    """Return list of tier codes P1–P6 found in text."""
    lowered = (text or "").lower()
    if not lowered:
        return []

    tiers: list[str] = []
    if any(k in lowered for k in P6_KEYWORDS):
        tiers.append("P6")
    if any(k in lowered for k in P5_KEYWORDS):
        tiers.append("P5")
    if any(k in lowered for k in P1_KEYWORDS):
        tiers.append("P1")
    if any(k in lowered for k in P2_KEYWORDS):
        tiers.append("P2")
    if any(k in lowered for k in P3_KEYWORDS):
        tiers.append("P3")
    return tiers


def score_product_category(text: str) -> tuple[int, bool, str | None]:
    """Returns (points, manual_review_only, dominant_tier)."""
    tiers = detect_product_tiers(text)
    if "P6" in tiers:
        return 0, True, "P6"

    tier_points = {"P1": 15, "P2": 12, "P3": 8, "P5": 15}
    found = [t for t in tiers if t in tier_points]
    if not found:
        return 8, False, None

    best = max(tier_points[t] for t in found)
    dominant = max(found, key=lambda t: tier_points[t])
    has_risky = any(t in tiers for t in ("P3",)) and any(
        k in (text or "").lower() for k in RISKY_PRODUCT_KEYWORDS
    )
    if len(found) > 1 and has_risky:
        best = max(5, best - 2)
    elif len(found) > 1:
        risky_in_mix = any(k in (text or "").lower() for k in RISKY_PRODUCT_KEYWORDS)
        if risky_in_mix:
            best = max(5, best - 2)

    return best, False, dominant


def score_response_quality(session: dict) -> int:
    has_company = bool(session.get("company"))
    has_country = bool(session.get("country"))
    has_biz = bool(session.get("business_type") or session.get("buyer_type"))
    has_value = bool(
        session.get("order_value_usd") or session.get("annual_volume_usd")
    )

    if has_company and has_country and has_biz and has_value:
        return 10
    if has_company and has_country and (has_biz or has_value):
        return 5
    return 0


def score_buying_intent(session: dict, message: str = "") -> int:
    combined = f"{message} {session.get('pending_intent', '')} {session.get('last_message', '')}".lower()
    if any(
        k in combined
        for k in (
            "payment details",
            "send payment",
            "ready to pay",
            "bank details",
            "wire transfer",
            "how to pay",
        )
    ):
        return 10
    if any(k in combined for k in ("final quote", "confirm price", "proforma", "invoice")):
        return 7
    if any(k in combined for k in ("price", "pricing", "quote", "order", "cost")):
        return 3
    return 0


def apply_bonuses_penalties(
    session: dict,
    base: int,
    *,
    country_high_risk: bool,
    product_manual: bool,
) -> tuple[int, bool]:
    score = base
    manual_review = product_manual

    if session.get("is_repeat_customer"):
        score += 10
    annual = float(session.get("annual_volume_usd") or 0)
    if annual >= 500_000 or session.get("bulk_quantity"):
        score += 5
    if session.get("fast_response"):
        score += 5
    if session.get("payment_proof_shared"):
        score += 15
    elif session.get("asked_payment_details"):
        score += 10
    if session.get("has_paid_before"):
        score += 20

    if session.get("suspicious_behavior"):
        score -= 20
    if session.get("incomplete_after_retries"):
        score -= 10
    if product_manual and country_high_risk:
        score -= 10
        manual_review = True

    return max(0, min(score, 100)), manual_review


def calculate_lead_score(lead: dict[str, Any]) -> int:
    """Backward-compatible: return final score only."""
    return score_lead(lead).score


def score_lead(session: dict[str, Any], message: str = "") -> LeadScoreResult:
    """Full client SOP scoring from session context."""
    session = session or {}
    text = " ".join(
        filter(
            None,
            [
                message,
                session.get("last_message", ""),
                session.get("product_query", ""),
                session.get("pending_intent", ""),
            ],
        )
    )

    buyer_type = session.get("buyer_type") or map_business_type_to_buyer_type(
        session.get("business_type"), text
    )
    buyer_pts = score_buyer_type(buyer_type)

    order_val = session.get("order_value_usd")
    if order_val is None:
        order_val = annual_volume_to_order_value_estimate(
            float(session.get("annual_volume_usd") or 0) or None
        )
    order_pts = score_order_value_usd(float(order_val) if order_val else None)

    country_pts, restricted, country_high_risk = score_country(session.get("country"))
    product_pts, product_manual, product_tier = score_product_category(text)
    response_pts = score_response_quality(session)
    intent_pts = score_buying_intent(session, message)

    breakdown = {
        "buyer_type": buyer_pts,
        "order_value": order_pts,
        "country": country_pts,
        "product_category": product_pts,
        "response_quality": response_pts,
        "buying_intent": intent_pts,
    }

    base = sum(breakdown.values())
    manual_review = product_manual or restricted
    disqualified = restricted or session.get("disqualified", False)

    if session.get("is_suspicious"):
        manual_review = True

    final, manual_review = apply_bonuses_penalties(
        session,
        base,
        country_high_risk=country_high_risk,
        product_manual=product_manual,
    )

    if disqualified:
        manual_review = True
        final = min(final, 39)

    category = classify_lead_score(final)
    if manual_review and not disqualified:
        category = "warm" if final >= 60 else "low_priority"

    return LeadScoreResult(
        score=final,
        category=category,
        manual_review_only=manual_review,
        disqualified=disqualified,
        breakdown=breakdown,
    )


def extract_order_value_from_text(text: str) -> float | None:
    """Parse per-order USD value from buyer message."""
    raw = (text or "").strip().lower()
    if not raw:
        return None

    money = re.search(r"\$?\s*(\d[\d,]*(?:\.\d+)?)", raw.replace(",", ""))
    if not money:
        return None
    value = float(money.group(1).replace(",", ""))
    if "k" in raw and value < 1000:
        value *= 1000
    return value


def enrich_session_from_message(session: dict, message: str) -> dict:
    """Update session scoring signals from the latest inbound message."""
    session = dict(session or {})
    session["last_message"] = message
    lowered = message.lower()

    if any(k in lowered for k in ("bulk", "large quantity", "container", "pallets")):
        session["bulk_quantity"] = True
    if any(k in lowered for k in ("payment details", "how to pay", "bank account")):
        session["asked_payment_details"] = True
    if any(k in lowered for k in ("screenshot", "payment proof", "paid")):
        session["payment_proof_shared"] = True
    if any(k in lowered for k in ("scam", "fake", "fraud")):
        session["suspicious_behavior"] = True

    order_val = extract_order_value_from_text(message)
    if order_val is not None:
        session["order_value_usd"] = order_val

    tiers = detect_product_tiers(message)
    if tiers:
        session["product_query"] = message
        if "P6" in tiers:
            session["manual_review_only"] = True

    return session
