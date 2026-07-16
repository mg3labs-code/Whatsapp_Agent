"""Pre/post LLM guardrail checks and GuardrailLog persistence."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from app.business.countries import (
    SHIPMENT_EXCLUDED_NAMES,
    is_shipment_excluded_country,
)
from app.db.database import SessionLocal
from app.db.models import GuardrailLog
from app.utils.security import user_ref

logger = logging.getLogger(__name__)

MESSAGE_TEXT_MAX_LEN = 200

# Shipment-excluded destinations (same list as order/qual agents).
SANCTIONED_COUNTRIES: tuple[str, ...] = SHIPMENT_EXCLUDED_NAMES

BLOCKED_TOPICS: tuple[str, ...] = (
    "dosage",
    "prescription",
    "clinical trial",
    "treatment plan",
    "side effects",
    "contraindication",
    "medical advice",
    "self-medicate",
    "administer",
)

# Generic compliance keywords (.cursorrules)
_GENERIC_HARD_BLOCKED: tuple[str, ...] = (
    "schedule h",
    "schedule h1",
    "schedule x",
    "narcotic",
    "psychotropic",
    "controlled substance",
)

# Hardcoded from data/import_samples/schedule_hx.xlsx (import with --schedule-xlsx)
SCHEDULE_X_PRODUCTS: tuple[str, ...] = (
    "amobarbital",
    "amphetamine",
    "barbital",
    "cyclobarbital",
    "dexamphetamine",
    "ethchlorvynol",
    "glutethimide",
    "ketamine",
    "meprobamate",
    "methamphetamine",
    "methylphenidate",
    "methylphenobarbital",
    "pentobarbital",
    "phencyclidine",
    "phenmetrazine",
    "secobarbital",
)

SCHEDULE_H_PRODUCTS: tuple[str, ...] = (
    "alprazolam",
    "chlordiazepoxide",
    "clobazam",
    "clonazepam",
    "codeine",
    "diazepam",
    "diphenoxylate",
    "ethionamide",
    "lorazepam",
    "midazolam",
    "nitrazepam",
    "pentazocine",
    "tramadol hydrochloride",
    "zolpidem",
)

SCHEDULE_H1_PRODUCTS: tuple[str, ...] = (
    "alprazolam",
    "buprenorphine",
    "chlordiazepoxide",
    "codeine",
    "diazepam",
    "diphenoxylate",
    "midazolam",
    "nitrazepam",
    "pentazocine",
    "tramadol",
    "zolpidem",
)

# Longest phrases first so e.g. "tramadol hydrochloride" matches before "tramadol"
_HARD_BLOCKED_MERGED = (
    _GENERIC_HARD_BLOCKED
    + SCHEDULE_X_PRODUCTS
    + SCHEDULE_H_PRODUCTS
    + SCHEDULE_H1_PRODUCTS
)
HARD_BLOCKED_PRODUCTS: tuple[str, ...] = tuple(
    sorted(set(_HARD_BLOCKED_MERGED), key=len, reverse=True)
)

REFUSAL_SANCTIONED_COUNTRY = (
    "I'm sorry, we're unable to process orders for that destination due to export "
    "compliance requirements. Please contact our compliance team directly."
)

REFUSAL_RESTRICTED_PRODUCT = (
    "I'm unable to assist with that product query through this channel. "
    "Please contact our medical compliance team directly."
)

REFUSAL_CLINICAL_CONTENT = (
    "I can't assist with that query. For medical or clinical questions, "
    "please consult a qualified healthcare professional."
)


@dataclass
class GuardrailResult:
    blocked: bool
    reason: str = ""
    refusal_message: str = ""


def _contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    lowered = (text or "").lower()
    if not lowered:
        return False
    return any(phrase in lowered for phrase in phrases)


def check_pre_guardrails(message: str, session: dict) -> GuardrailResult:
    """Run before any agent/LLM call."""
    session = session or {}

    if session.get("disqualified") or session.get("lifecycle_stage") == "disqualified":
        return GuardrailResult(
            blocked=True,
            reason="disqualified_lead",
            refusal_message=REFUSAL_SANCTIONED_COUNTRY,
        )

    country = session.get("country")
    if country and is_shipment_excluded_country(country):
        return GuardrailResult(
            blocked=True,
            reason="sanctioned_country",
            refusal_message=REFUSAL_SANCTIONED_COUNTRY,
        )

    if _contains_phrase(message, HARD_BLOCKED_PRODUCTS):
        return GuardrailResult(
            blocked=True,
            reason="restricted_product",
            refusal_message=REFUSAL_RESTRICTED_PRODUCT,
        )

    return GuardrailResult(blocked=False)


def check_post_guardrails(response: str) -> GuardrailResult:
    """Run on agent output before sending to the buyer."""
    if _contains_phrase(response, BLOCKED_TOPICS):
        return GuardrailResult(
            blocked=True,
            reason="clinical_content",
            refusal_message=REFUSAL_CLINICAL_CONTENT,
        )
    return GuardrailResult(blocked=False)


def _write_guardrail_log(
    phone: str,
    reason: str,
    stage: str,
    message_text: str,
) -> None:
    db = SessionLocal()
    try:
        # SECURITY: cap stored message_text at 200 chars (PII minimization)
        truncated = (message_text or "")[:MESSAGE_TEXT_MAX_LEN]
        entry = GuardrailLog(
            phone=phone,
            trigger_type=stage,
            reason=reason,
            message_text=truncated or None,
        )
        db.add(entry)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


async def log_guardrail(
    phone: str,
    reason: str,
    stage: str,
    message_text: str = "",
) -> None:
    """Persist a guardrail trigger; logs on failure — never swallows errors silently."""
    try:
        await asyncio.to_thread(
            _write_guardrail_log,
            phone,
            reason,
            stage,
            message_text,
        )
    except Exception:
        # SECURITY: hashed user ref in logs — not raw phone
        logger.exception(
            "Failed to write guardrail_logs entry user_ref=%s reason=%s stage=%s",
            user_ref(phone),
            reason,
            stage,
        )
        raise
