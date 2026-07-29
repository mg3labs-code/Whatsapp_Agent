"""Pre/post LLM guardrail checks and GuardrailLog persistence."""

from __future__ import annotations

import asyncio
import logging
import re
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

    return GuardrailResult(blocked=False)


# Numeric strength / dose amount (mg, ml, mcg).
_DOSE_AMOUNT_RE = re.compile(r"\d+\s*(?:mg|ml|mcg)\b", re.IGNORECASE)

# Imperative / instructional dosing without requiring a BLOCKED_TOPICS word —
# e.g. "take 500mg twice daily" — while allowing catalog lines like "Metformin 500mg".
_DOSING_ADVICE_RE = re.compile(
    r"(?:"
    r"\b(?:take|administer|dose(?:d|s)?(?:\s+of)?|give|inject|consume)\s+"
    r"\d+\s*(?:mg|ml|mcg)\b"
    r"|"
    r"\b\d+\s*(?:mg|ml|mcg)\b.{0,40}\b"
    r"(?:twice|thrice|daily|per\s+day|every\s+\d+|times?\s+(?:a|per)\s+day|"
    r"once\s+(?:a|per)\s+day|morning|evening|with\s+meals?)\b"
    r"|"
    r"\b(?:twice|thrice|daily|per\s+day|every\s+\d+|times?\s+(?:a|per)\s+day|"
    r"once\s+(?:a|per)\s+day)\b.{0,40}\b\d+\s*(?:mg|ml|mcg)\b"
    r")",
    re.IGNORECASE | re.DOTALL,
)


def check_post_guardrails(response: str) -> GuardrailResult:
    """Run on agent output before sending to the buyer.

    Blocks clinical dosing advice: a BLOCKED_TOPICS phrase near a numeric dose,
    or imperative / frequency-linked dosing (e.g. "take 500mg twice daily") even
    without those topic words. Catalog product strengths alone still pass.
    """
    lowered = (response or "").lower()
    if not lowered:
        return GuardrailResult(blocked=False)

    if _DOSING_ADVICE_RE.search(lowered):
        return GuardrailResult(
            blocked=True,
            reason="clinical_content",
            refusal_message=REFUSAL_CLINICAL_CONTENT,
        )

    near_window = 80
    for phrase in BLOCKED_TOPICS:
        start = 0
        while True:
            idx = lowered.find(phrase, start)
            if idx == -1:
                break
            window_start = max(0, idx - near_window)
            window_end = min(len(lowered), idx + len(phrase) + near_window)
            if _DOSE_AMOUNT_RE.search(lowered[window_start:window_end]):
                return GuardrailResult(
                    blocked=True,
                    reason="clinical_content",
                    refusal_message=REFUSAL_CLINICAL_CONTENT,
                )
            start = idx + 1

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
