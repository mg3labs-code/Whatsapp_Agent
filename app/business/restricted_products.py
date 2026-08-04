"""Restricted product term pre-checks (independent from sellable catalog rows)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from time import monotonic

from sqlalchemy.orm import Session

from app.db.models import RestrictedTerm

RESTRICTED_TERMS_CACHE_TTL_SECONDS = 300
_MIN_TERM_LEN = 4
_SHORT_TOKEN_RE = re.compile(r"^[a-z0-9]{1,6}$")
_NON_WORD_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class RestrictedTermHit:
    term: str
    schedule_category: str | None


_CACHE_LOADED_AT: float = 0.0
_CACHE_TERMS: list[tuple[str, str, str | None]] = []


def normalize_term(text: str) -> str:
    """Normalize free text and imported terms for deterministic substring matching."""
    lowered = (text or "").strip().lower()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def clear_restricted_terms_cache() -> None:
    global _CACHE_LOADED_AT, _CACHE_TERMS
    _CACHE_LOADED_AT = 0.0
    _CACHE_TERMS = []


def _load_cached_terms(db: Session) -> list[tuple[str, str, str | None]]:
    global _CACHE_LOADED_AT, _CACHE_TERMS
    now = monotonic()
    if _CACHE_TERMS and (now - _CACHE_LOADED_AT) < RESTRICTED_TERMS_CACHE_TTL_SECONDS:
        return _CACHE_TERMS

    rows = db.query(RestrictedTerm).all()
    terms: list[tuple[str, str, str | None]] = []
    for row in rows:
        normalized = normalize_term(row.normalized_term or row.term or "")
        if len(normalized) < _MIN_TERM_LEN:
            continue
        terms.append((normalized, row.term, row.schedule_category))

    # Longest first to prefer specific salt names over short fragments.
    terms.sort(key=lambda item: len(item[0]), reverse=True)
    _CACHE_TERMS = terms
    _CACHE_LOADED_AT = now
    return terms


def match_restricted_term(query: str, db: Session) -> RestrictedTermHit | None:
    """Return first restricted hit for buyer query, else None."""
    text = normalize_term(query)
    if len(text) < _MIN_TERM_LEN:
        return None

    padded = f" {text} "
    for normalized, original, category in _load_cached_terms(db):
        # For short alnum tokens, enforce token boundaries to avoid false positives.
        if _SHORT_TOKEN_RE.fullmatch(normalized):
            boundary = f" {normalized} "
            if boundary in padded:
                return RestrictedTermHit(term=original, schedule_category=category)
            continue

        # Generic boundary check: replace punctuation with spaces for stable containment.
        normalized_query = f" {_NON_WORD_RE.sub(' ', text)} "
        normalized_term = f" {_NON_WORD_RE.sub(' ', normalized)} "
        if normalized_term in normalized_query:
            return RestrictedTermHit(term=original, schedule_category=category)
    return None
