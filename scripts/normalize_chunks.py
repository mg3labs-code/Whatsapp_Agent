"""Normalize raw extracted Q/A pairs into the canonical FAQ chunk schema.

Input:  data/faq_extracted/*.jsonl   (any number of raw pair files)
Output: docs/faq_chunks.json         (single sorted, deduped, topic-tagged array)

Canonical schema (one JSON object per chunk):
    {
      "id":          "faq-<topic>-<sha1[:8]>",       # deterministic, idempotent
      "source":      "<original filename>",
      "source_page": <int|null>,
      "topic":       "shipping" | "payment" | ...,
      "section":     "<section header from source>",
      "question":    "<cleaned question or section title>",
      "answer":      "<cleaned answer body>",
      "text":        "Q: <question>\\nA: <answer>",   # what gets embedded
      "version":     "<YYYY-MM>",
      "checksum":    "sha1:<hex>"                     # of text; re-embed if changed
    }

Behaviour:
  - Whitespace: collapse multi-blank-lines to single \\n, trim trailing junk.
  - Drop chunks with empty Q AND empty A, or A shorter than 10 chars.
  - Dedupe by normalized-question hash (keep latest version seen, prefer
    Q/A over section chunks if duplicate).
  - Assign topic via keyword dictionary (first-match wins; fallback "general").
  - If answer >800 tokens (~3200 chars) split into ~400-token chunks with Q
    repeated. Keeps single-vector retrieval workable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

EXTRACTED_DIR = Path("data/faq_extracted")
OUT_PATH = Path("docs/faq_chunks.json")

DEFAULT_VERSION = datetime.now(timezone.utc).strftime("%Y-%m")
MAX_ANSWER_CHARS = 3200
SPLIT_CHARS = 1600

# Topic keyword hierarchy.
# Each entry is a list of keywords matched with word-boundary regex.
# Detection priority (most reliable first):
#   1. SECTION_HINTS hit on the section header
#   2. STRONG_KEYWORDS hit on the question (specific, low false-positive)
#   3. STRONG_KEYWORDS hit on the answer (fallback)
#   4. "general" if nothing matches
SECTION_HINTS: dict[str, list[str]] = {
    "shipping": ["shipping", "tracking", "transit", "dispatch", "courier", "international shipping"],
    "products": ["product", "categories", "stock", "availability", "substitution", "alternatives"],
    "pricing": ["pricing", "bulk deals", "discount", "recurring orders"],
    "order": ["order flow", "order placement", "standard order"],
    "company": [
        "company overview", "credibility", "reputation", "trust", "background",
        "mission", "vision", "first-time buyer",
    ],
    "regulatory": [
        "compliance", "customs", "regulatory", "legal status", "duties",
        "product restrictions", "liability",
    ],
    "documentation": ["documents", "paperwork", "kyc", "certifications"],
    "payment": ["payment", "billing", "transactions"],
    "quality": ["quality", "authenticity", "origin", "storage", "packaging"],
    "returns": ["return", "reship", "lost parcel", "refund", "return-to-origin"],
    "ai_rules": ["ai response", "ai behaviour", "ai escalation", "master rule", "escalation rules"],
    "support": ["customer support", "contact", "support"],
    "prescription": ["prescription", "medical use"],
}

STRONG_KEYWORDS: dict[str, list[str]] = {
    "shipping": [
        "shipping", "freight", "courier", "delivery", "dispatch", "transit time",
        "lead time", "dhl", "fedex", "air freight", "sea freight", "tracking",
    ],
    "payment": [
        "payment", "t/t", "telegraphic transfer", "l/c", "letter of credit",
        "advance payment", "net 30", "wire transfer", "remittance",
    ],
    "regulatory": [
        "who-gmp", "regulatory", "license", "licensed", "compliance",
        "sfda", "mohap", "registration", "sanctioned", "embassy attestation",
        "controlled", "schedule h", "schedule x", "narcotic", "psychotropic",
        "import permit", "import license",
    ],
    "documentation": [
        "coa", "copp", "certificate of analysis", "certificate of origin",
        "commercial invoice", "packing list", "bill of lading", "airway bill",
        "free sale certificate", "attestation",
    ],
    "quality": [
        "batch", "shelf life", "stability", "pharmacopoeia", "assay",
        "dissolution", "expiry", "ip/bp/usp", "who labeling", "cold chain",
    ],
    "company": [
        "newlife medicare", "head office", "company name", "experience",
        "founded", "established", "mission", "vision",
    ],
    "products": [
        "sku", "category", "categories", "generic medicine", "branded medicine",
        "tablet", "capsule", "injectable", "antibiotic", "antidiabetic",
        "analgesic", "product portfolio", "product range",
    ],
    "pricing": [
        "price", "pricing", "quote", "discount", "bulk pricing", "tier pricing",
        "minimum order quantity",
    ],
    "order": [
        "place an order", "place order", "proforma", "purchase order",
        "confirm order", "order ref", "order confirmation",
    ],
    "sample": ["sample request", "product sample", "trial order"],
    "returns": [
        "return policy", "claim", "discrepancy", "defective", "damage",
        "replacement", "refund", "shortage", "reship",
    ],
    "restrictions": [
        "restricted", "prohibited", "not permitted", "controlled substance",
        "sanctioned",
    ],
    "escalation": [
        "escalation", "escalate", "human agent", "speak to", "talk to",
        "manual support", "senior export manager",
    ],
    "support": [
        "customer service", "business hours", "support team",
    ],
    "ai_rules": [
        "ai agent", "ai must", "ai should", "master rule", "ai safe",
        "the bot", "the assistant",
    ],
    "prescription": [
        "prescription", "valid prescription",
    ],
}


def _build_keyword_pattern(keywords: list[str]) -> re.Pattern[str]:
    parts = [re.escape(k) for k in keywords]
    return re.compile(r"(?<![a-z])(" + "|".join(parts) + r")(?![a-z])", re.IGNORECASE)


_SECTION_PATTERNS = {topic: _build_keyword_pattern(kws) for topic, kws in SECTION_HINTS.items()}
_KEYWORD_PATTERNS = {topic: _build_keyword_pattern(kws) for topic, kws in STRONG_KEYWORDS.items()}

EMOJI_OR_SYMBOL_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\u2600-\u27BF"
    "\u2700-\u27BF"
    "]+",
    flags=re.UNICODE,
)


def _strip_noise(text: str) -> str:
    text = EMOJI_OR_SYMBOL_RE.sub("", text)
    text = text.replace("•", "-").replace("●", "-").replace("◦", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_question(q: str) -> str:
    q = unicodedata.normalize("NFKC", q).strip()
    q = _strip_noise(q)
    q = re.sub(r"^\d+\.\s*", "", q)
    stripped = q.strip(" .?:")
    if not stripped:
        return ""
    return stripped + "?"


def _normalize_question_for_dedupe(q: str) -> str:
    base = unicodedata.normalize("NFKC", q).lower()
    base = re.sub(r"[^a-z0-9]+", " ", base)
    return " ".join(base.split())


def _normalize_answer(a: str) -> str:
    a = unicodedata.normalize("NFKC", a)
    a = _strip_noise(a)
    a = re.sub(r"\n[ \t]+", "\n", a)
    return a.strip()


def _detect_topic(question: str, answer: str, section: str) -> str:
    """Three-tier topic detection: section > question > answer.

    Word-boundary regex avoids substring traps like 'ship' matching 'shipment'
    inside an answer that's actually about pricing.
    """
    section_l = section.lower()
    for topic, pat in _SECTION_PATTERNS.items():
        if pat.search(section_l):
            return topic

    question_l = question.lower()
    q_hits: dict[str, int] = {}
    for topic, pat in _KEYWORD_PATTERNS.items():
        n = len(pat.findall(question_l))
        if n:
            q_hits[topic] = n
    if q_hits:
        return max(q_hits, key=q_hits.get)

    answer_l = answer.lower()
    a_hits: dict[str, int] = {}
    for topic, pat in _KEYWORD_PATTERNS.items():
        n = len(pat.findall(answer_l))
        if n:
            a_hits[topic] = n
    if a_hits:
        return max(a_hits, key=a_hits.get)

    return "general"


def _make_text(question: str, answer: str) -> str:
    return f"Q: {question}\nA: {answer}"


def _chunk_id(topic: str, question: str) -> str:
    h = hashlib.sha1(question.encode("utf-8")).hexdigest()[:8]
    return f"faq-{topic}-{h}"


def _checksum(text: str) -> str:
    return "sha1:" + hashlib.sha1(text.encode("utf-8")).hexdigest()


def _split_long_answer(answer: str) -> list[str]:
    """Split an oversize answer into ~SPLIT_CHARS chunks, on paragraph boundaries."""
    if len(answer) <= MAX_ANSWER_CHARS:
        return [answer]

    pieces: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in re.split(r"\n\s*\n", answer):
        para = para.strip()
        if not para:
            continue
        if current_len + len(para) + 2 > SPLIT_CHARS and current:
            pieces.append("\n\n".join(current))
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para) + 2
    if current:
        pieces.append("\n\n".join(current))
    return pieces


def load_raw_chunks(extracted_dir: Path) -> list[dict]:
    raw: list[dict] = []
    for path in sorted(extracted_dir.glob("*.jsonl")):
        for ln in path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            raw.append(json.loads(ln))
    return raw


def normalize(raw_chunks: Iterable[dict], version: str = DEFAULT_VERSION) -> list[dict]:
    by_qhash: dict[str, dict] = {}
    skipped_short = 0
    skipped_empty = 0

    for c in raw_chunks:
        q = _normalize_question(c.get("question", ""))
        a = _normalize_answer(c.get("answer", ""))

        if not q and not a:
            skipped_empty += 1
            continue
        if not q:
            q = (c.get("section") or "General information").strip(" .?:") + "?"
        if len(a) < 10:
            skipped_short += 1
            continue

        section = (c.get("section") or "").strip()
        source = c.get("source") or ""
        page = c.get("page")
        topic = _detect_topic(q, a, section)
        kind = c.get("kind", "qa")

        qhash = _normalize_question_for_dedupe(q)
        if qhash in by_qhash:
            existing = by_qhash[qhash]
            existing_kind = existing.get("_kind", "qa")
            if kind == "qa" and existing_kind != "qa":
                pass
            elif kind != "qa" and existing_kind == "qa":
                continue
            elif len(a) <= len(existing["answer"]):
                continue

        by_qhash[qhash] = {
            "question": q,
            "answer": a,
            "source": source,
            "source_page": page,
            "section": section,
            "topic": topic,
            "_kind": kind,
        }

    canonical: list[dict] = []
    for entry in by_qhash.values():
        for piece_idx, answer_piece in enumerate(_split_long_answer(entry["answer"])):
            text = _make_text(entry["question"], answer_piece)
            id_seed = entry["question"] if piece_idx == 0 else f"{entry['question']}#part{piece_idx}"
            chunk = {
                "id": _chunk_id(entry["topic"], id_seed),
                "source": entry["source"],
                "source_page": entry["source_page"],
                "topic": entry["topic"],
                "section": entry["section"],
                "question": entry["question"],
                "answer": answer_piece,
                "text": text,
                "version": version,
                "checksum": _checksum(text),
            }
            canonical.append(chunk)

    canonical.sort(key=lambda c: (c["topic"], c["id"]))
    if skipped_empty or skipped_short:
        print(f"  skipped: {skipped_empty} empty, {skipped_short} too-short answers")
    return canonical


def write_chunks(chunks: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize FAQ chunks into canonical schema.")
    parser.add_argument("--extracted-dir", type=Path, default=EXTRACTED_DIR)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    args = parser.parse_args()

    raw = load_raw_chunks(args.extracted_dir)
    print(f"Loaded {len(raw)} raw pairs from {args.extracted_dir}")

    chunks = normalize(raw, version=args.version)
    write_chunks(chunks, args.out)

    print(f"\nWrote {len(chunks)} canonical chunks → {args.out}")
    by_topic = Counter(c["topic"] for c in chunks)
    print("\nChunks per topic:")
    for topic, count in by_topic.most_common():
        print(f"  {count:>4}  {topic}")
    by_source = Counter(c["source"] for c in chunks)
    print("\nChunks per source:")
    for source, count in by_source.most_common():
        print(f"  {count:>4}  {source}")
    answer_lens = [len(c["answer"]) for c in chunks]
    if answer_lens:
        print(f"\nAnswer length: min={min(answer_lens)} avg={sum(answer_lens)//len(answer_lens)} max={max(answer_lens)}")


if __name__ == "__main__":
    main()
