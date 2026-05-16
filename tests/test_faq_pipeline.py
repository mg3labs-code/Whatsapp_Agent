"""Unit tests for the FAQ ingestion pipeline (normalize_chunks)."""

from __future__ import annotations

from scripts.normalize_chunks import (
    _checksum,
    _chunk_id,
    _detect_topic,
    _normalize_answer,
    _normalize_question,
    _normalize_question_for_dedupe,
    _split_long_answer,
    normalize,
)


# ---------------------------------------------------------------------------
# Normalization primitives
# ---------------------------------------------------------------------------


def test_normalize_question_adds_question_mark():
    assert _normalize_question("What is your company name") == "What is your company name?"


def test_normalize_question_preserves_existing_question_mark():
    assert _normalize_question("What is your company name?") == "What is your company name?"


def test_normalize_question_strips_leading_number():
    assert _normalize_question("1. Company Overview") == "Company Overview?"


def test_normalize_question_strips_emoji_and_whitespace():
    assert _normalize_question("  📦  How do you ship?  ") == "How do you ship?"


def test_normalize_answer_collapses_blank_lines():
    answer = "Line 1\n\n\n\nLine 2"
    assert _normalize_answer(answer) == "Line 1\n\nLine 2"


def test_normalize_answer_strips_emoji():
    answer = "We ship globally. 🌍 🚚 Reliable."
    assert "🌍" not in _normalize_answer(answer)
    assert "🚚" not in _normalize_answer(answer)


def test_normalize_question_for_dedupe_ignores_punctuation_and_case():
    a = _normalize_question_for_dedupe("Do you ship to UAE?")
    b = _normalize_question_for_dedupe("do you ship to UAE")
    c = _normalize_question_for_dedupe("Do you ship to UAE!")
    assert a == b == c


# ---------------------------------------------------------------------------
# Topic detection (priority: section > question > answer)
# ---------------------------------------------------------------------------


def test_topic_detection_prefers_section_header():
    topic = _detect_topic(
        question="What about delivery times?",
        answer="See our partner list.",
        section="9. Shipping process and tracking",
    )
    assert topic == "shipping"


def test_topic_detection_falls_back_to_question_keywords():
    topic = _detect_topic(
        question="How does the L/C payment work?",
        answer="It is processed via SWIFT.",
        section="",
    )
    assert topic == "payment"


def test_topic_detection_uses_answer_only_as_last_resort():
    topic = _detect_topic(
        question="Tell me more",
        answer="We are WHO-GMP certified and hold a valid drug license in India.",
        section="",
    )
    assert topic == "regulatory"


def test_topic_detection_general_when_nothing_matches():
    topic = _detect_topic(question="hello there", answer="nice to meet you", section="")
    assert topic == "general"


def test_topic_detection_word_boundary_does_not_substring_match():
    # 'ship' shouldn't match 'shipment' inside an answer that's actually about pricing.
    topic = _detect_topic(
        question="Do you offer tier pricing?",
        answer="Tier pricing applies to every shipment over 200 units.",
        section="",
    )
    assert topic == "pricing"


# ---------------------------------------------------------------------------
# Deterministic IDs and checksums
# ---------------------------------------------------------------------------


def test_chunk_id_is_deterministic_and_topic_prefixed():
    a = _chunk_id("shipping", "Do you ship to UAE?")
    b = _chunk_id("shipping", "Do you ship to UAE?")
    c = _chunk_id("payment", "Do you ship to UAE?")
    assert a == b
    assert a != c
    assert a.startswith("faq-shipping-")


def test_checksum_changes_when_text_changes():
    a = _checksum("Q: foo\nA: bar")
    b = _checksum("Q: foo\nA: BAZ")
    assert a != b
    assert a.startswith("sha1:")


# ---------------------------------------------------------------------------
# Oversize answer splitting
# ---------------------------------------------------------------------------


def test_split_long_answer_passthrough_for_short_text():
    text = "Short answer."
    assert _split_long_answer(text) == [text]


def test_split_long_answer_splits_on_paragraph_boundary():
    paragraphs = ["This is paragraph " + str(i) + ". " + "x" * 800 for i in range(5)]
    long_answer = "\n\n".join(paragraphs)
    pieces = _split_long_answer(long_answer)
    assert len(pieces) >= 2
    assert all(len(p) <= 2200 for p in pieces)


# ---------------------------------------------------------------------------
# End-to-end normalization
# ---------------------------------------------------------------------------


def _raw(question, answer, source="x.docx", section="", page=None, kind="qa"):
    return {
        "question": question,
        "answer": answer,
        "source": source,
        "section": section,
        "page": page,
        "kind": kind,
    }


def test_normalize_full_pipeline_assigns_schema_fields():
    chunks = normalize([
        _raw(
            "Do you ship to UAE",
            "Yes, air freight to UAE takes 5–7 business days.",
            section="9. Shipping process and tracking",
        ),
    ])
    assert len(chunks) == 1
    c = chunks[0]
    assert set(c.keys()) >= {
        "id", "source", "source_page", "topic", "section", "question", "answer", "text", "version", "checksum",
    }
    assert c["topic"] == "shipping"
    assert c["question"] == "Do you ship to UAE?"
    assert c["id"].startswith("faq-shipping-")
    assert c["text"].startswith("Q: Do you ship to UAE?")
    assert c["checksum"].startswith("sha1:")


def test_normalize_dedupes_same_question_keeps_longer_answer():
    longer_answer = "Yes, very long detailed answer about shipping to UAE " * 5
    chunks = normalize([
        _raw("Do you ship to UAE?", "Yes.", section="shipping"),
        _raw("Do you ship to UAE?", longer_answer, section="shipping"),
    ])
    assert len(chunks) == 1
    assert chunks[0]["answer"].startswith("Yes, very long detailed answer")


def test_normalize_drops_short_answers():
    chunks = normalize([
        _raw("Is something?", "yes"),  # 3 chars < 10 threshold
        _raw("Is something else?", "Yes it definitely is."),
    ])
    assert len(chunks) == 1
    assert chunks[0]["question"] == "Is something else?"


def test_normalize_dedupes_case_insensitive():
    chunks = normalize([
        _raw("DO YOU SHIP TO UAE?", "Answer one about shipping to UAE."),
        _raw("do you ship to uae", "Answer two about shipping to UAE that is much longer."),
    ])
    assert len(chunks) == 1
    assert "longer" in chunks[0]["answer"]


def test_normalize_qa_kind_wins_over_section_kind_on_duplicate():
    chunks = normalize([
        _raw("Company Overview?", "Section content here is quite long.", kind="section"),
        _raw("Company Overview?", "Q/A content is here.", kind="qa"),
    ])
    assert len(chunks) == 1
    assert chunks[0]["answer"] == "Q/A content is here."


def test_normalize_assigns_deterministic_ids_across_runs():
    raw = [_raw("Hello?", "World, with at least ten chars.", section="company")]
    a = normalize(raw)
    b = normalize(raw)
    assert a[0]["id"] == b[0]["id"]
    assert a[0]["checksum"] == b[0]["checksum"]
