"""Extract Q/A pairs (or section chunks) from a PDF FAQ doc.

Auto-detects which of two strategies applies, based on what the document
actually contains:

  - "qa_inline":  text has 'Q:' markers, e.g. the Regulatory & Compliance FAQ.
                  Each 'Q:' line opens a Q; subsequent non-'Q:' lines accumulate
                  as the A, until the next 'Q:' or a new numbered section header.

  - "sections":   numbered section headings ('1. ...', '2. ...') with prose
                  bodies, e.g. the Company Background document. Each section
                  becomes one knowledge-card chunk (synthetic question, body
                  as answer).

Text extraction uses pdfplumber (much cleaner than pypdf for these files —
preserves line breaks, no double-spaced words).

Usage:
    python -m scripts.extract_pdf <path.pdf> <out.jsonl> [--strategy auto|qa_inline|sections]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber


SECTION_BASE_RE = re.compile(r"^\s*(\d+)\.\s+(.+?)\s*$")
Q_LINE_RE = re.compile(r"^\s*Q\s*\d*\s*[:.\)]\s*(.+)$", re.IGNORECASE)
A_LINE_RE = re.compile(r"^\s*A\s*\d*\s*[:.\)]\s*(.+)$", re.IGNORECASE)

NOISE_LINE_RE = re.compile(
    r"^\s*(📄|🔹|⚠|🏢|📅|🌍|✅|⭐|📦|🚚|💊|🌐|✈|🛠|💼|🏷|🧾|📑|📞|💬|👋|👀|🎯|🧠|🚀|☎|✉)\s*$"
)


@dataclass
class _Pair:
    question: str = ""
    answer_parts: list[str] = field(default_factory=list)
    section: str = ""
    page: int = 0

    def is_open(self) -> bool:
        return bool(self.question) or bool(self.answer_parts)

    def to_dict(self, source: str, kind: str = "qa") -> dict | None:
        q = (self.question or "").strip()
        a = "\n".join(p for p in self.answer_parts if p.strip()).strip()
        if not q and not a:
            return None
        d = {
            "question": q,
            "answer": a,
            "source": source,
            "page": self.page or None,
            "section": self.section,
            "sub_section": "",
        }
        if kind != "qa":
            d["kind"] = kind
        return d


def _read_lines_with_pages(path: Path) -> list[tuple[int, str]]:
    """Return list of (page_number, line_text) tuples."""
    out: list[tuple[int, str]] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            for raw in text.splitlines():
                line = raw.strip()
                if not line:
                    continue
                if NOISE_LINE_RE.match(line):
                    continue
                out.append((i + 1, line))
    return out


def _is_section_heading(line: str) -> tuple[bool, str]:
    """Detect a numbered section heading.

    Matches both:
      - ALL-CAPS:    '1. LEGAL STATUS & BUSINESS'   (uppercase-letter ratio >= 70%)
      - Title-case:  '1. Company Overview'          (>=50% title-case words, <= 60 chars)

    Rejects lines that look like sentences (terminal '.', embedded ':', or
    too long) so we don't mistake "1. We ship to over 50 countries." for a heading.
    """
    m = SECTION_BASE_RE.match(line)
    if not m:
        return False, ""
    num, text = m.group(1), m.group(2).strip()
    if not text or len(text) > 80:
        return False, ""
    if text.endswith(".") or ":" in text:
        return False, ""

    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False, ""

    uppercase_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    if uppercase_ratio >= 0.7:
        return True, f"{num}. {text}"

    if len(text) > 60:
        return False, ""
    words = [w for w in text.split() if w]
    title_ratio = sum(1 for w in words if w[:1].isupper()) / len(words)
    if title_ratio >= 0.5:
        return True, f"{num}. {text}"

    return False, ""


def _detect_strategy(lines: list[tuple[int, str]]) -> str:
    q_hits = sum(1 for _, ln in lines if Q_LINE_RE.match(ln))
    section_hits = sum(1 for _, ln in lines if _is_section_heading(ln)[0])
    if q_hits >= 5:
        return "qa_inline"
    if section_hits >= 3:
        return "sections"
    if q_hits > 0:
        return "qa_inline"
    return "sections"


def _extract_qa_inline(lines: list[tuple[int, str]], source: str) -> list[dict]:
    pairs: list[dict] = []
    current = _Pair()
    section = ""

    def flush() -> None:
        nonlocal current
        if current.is_open():
            d = current.to_dict(source)
            if d:
                pairs.append(d)
        current = _Pair(section=section)

    for page, line in lines:
        is_section, section_label = _is_section_heading(line)
        if is_section:
            flush()
            section = section_label
            current = _Pair(section=section)
            continue

        m = Q_LINE_RE.match(line)
        if m:
            flush()
            current = _Pair(section=section, page=page)
            q_text = m.group(1).strip()
            current.question = q_text
            continue

        m_a = A_LINE_RE.match(line)
        if m_a:
            current.answer_parts.append(m_a.group(1).strip())
            continue

        if current.is_open():
            if not current.page:
                current.page = page
            current.answer_parts.append(line)

    flush()
    return pairs


def _extract_sections(lines: list[tuple[int, str]], source: str) -> list[dict]:
    pairs: list[dict] = []
    current = _Pair()
    in_intro = True

    def flush(kind: str = "section") -> None:
        nonlocal current
        if current.is_open():
            d = current.to_dict(source, kind=kind)
            if d:
                pairs.append(d)
        current = _Pair()

    for page, line in lines:
        is_section, section_label = _is_section_heading(line)
        if is_section:
            flush()
            in_intro = False
            current = _Pair(page=page, section=section_label)
            current.question = section_label
            continue

        if in_intro:
            continue

        if not current.page:
            current.page = page
        current.answer_parts.append(line)

    flush()
    return pairs


def extract(pdf_path: Path, source_name: str | None = None, strategy: str = "auto") -> tuple[list[dict], str]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    source_name = source_name or pdf_path.name

    lines = _read_lines_with_pages(pdf_path)
    if not lines:
        print(f"WARNING: no text extracted from {pdf_path}. PDF may be scanned (OCR required).", file=sys.stderr)
        return [], "empty"

    chosen = strategy if strategy != "auto" else _detect_strategy(lines)
    if chosen == "qa_inline":
        return _extract_qa_inline(lines, source_name), chosen
    if chosen == "sections":
        return _extract_sections(lines, source_name), chosen
    raise ValueError(f"Unknown strategy: {chosen}")


def write_jsonl(pairs: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Q/A chunks from a PDF FAQ document.")
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--strategy", choices=["auto", "qa_inline", "sections"], default="auto")
    parser.add_argument("--source-name", default=None)
    args = parser.parse_args()

    pairs, chosen = extract(args.pdf_path, source_name=args.source_name, strategy=args.strategy)
    write_jsonl(pairs, args.output_jsonl)
    print(f"Strategy: {chosen}")
    print(f"Extracted {len(pairs)} chunks → {args.output_jsonl}")
    sections = {p.get("section", "") for p in pairs if p.get("section")}
    print(f"  Distinct sections: {len(sections)}")


if __name__ == "__main__":
    main()
