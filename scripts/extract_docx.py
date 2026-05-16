"""Extract Q/A pairs (and structured knowledge blocks) from a .docx FAQ doc.

Designed for the Newlife Medicare FAQ docx, which uses custom paragraph styles:
  - SectionStyle, SubSectionStyle  → topic / sub-topic markers
  - QStyle                         → question
  - AStyle                         → answer (often followed by List Bullet paras)
  - TitleStyle, SubTitleStyle      → document preamble (skipped)

Strategy ("docx_styles"):
  Walk document body in order. Track current section/sub-section. Open a new
  Q/A pair on every QStyle. Append AStyle/List Bullet content to the open
  pair until the next QStyle, SectionStyle, or SubSectionStyle. SubSectionStyle
  blocks without an explicit Q become "knowledge cards" (synthetic question).
  Tables are emitted as separate knowledge cards.

Usage:
    python -m scripts.extract_docx \
        "data/faq_raw/Newlife_Medicare_Comprehensive_AI_FAQ.docx" \
        data/faq_extracted/newlife_faq.jsonl

The output is JSONL with one raw pair per line:
    {"question", "answer", "source", "page", "section", "sub_section"}

Output is intentionally "raw" — whitespace cleanup, deduping, topic
assignment, and id/checksum generation all happen in normalize_chunks.py.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from docx import Document
from docx.document import Document as _Document
from docx.oxml.ns import qn
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph


SKIP_STYLES = {"TitleStyle", "SubTitleStyle", "Normal"}


def iter_block_items(parent: _Document | _Cell) -> Iterator[Paragraph | Table]:
    """Yield paragraphs and tables in document order."""
    parent_elm = parent.element.body if isinstance(parent, _Document) else parent._tc
    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


@dataclass
class _Pair:
    question: str = ""
    answer_parts: list[str] = field(default_factory=list)
    section: str = ""
    sub_section: str = ""

    def is_open(self) -> bool:
        return bool(self.question) or bool(self.answer_parts)

    def to_dict(self, source: str) -> dict | None:
        q = _strip_qa_prefix(self.question)
        a = "\n".join(p for p in self.answer_parts if p.strip())
        a = _strip_qa_prefix(a)
        if not q and not a:
            return None
        if not q:
            # Section had body content but no explicit Q (e.g. appendix bullets).
            # Use the section/sub-section as a synthetic question so the chunk
            # is still retrievable.
            q = self.sub_section or self.section or "General information"
        return {
            "question": q,
            "answer": a,
            "source": source,
            "page": None,
            "section": self.section,
            "sub_section": self.sub_section,
        }


_QA_PREFIX = re.compile(r"^\s*(Q|A|Question|Answer)\s*[.:\)]\s*", re.IGNORECASE)


def _strip_qa_prefix(text: str) -> str:
    """Remove a leading 'Q.' / 'A.' / 'Question:' / 'Answer:' from a line.

    The DOCX uses these prefixes inside QStyle/AStyle paragraphs. We strip them
    so the canonical schema's `question`/`answer` fields are clean.
    """
    text = text.strip()
    while True:
        m = _QA_PREFIX.match(text)
        if not m:
            break
        text = text[m.end():].strip()
    return text


def _serialize_table(tbl: Table, section: str, sub_section: str, idx: int) -> dict | None:
    """Convert a Word table into a knowledge-card chunk.

    Two cases handled:
      - 2-column key/value table (e.g. company profile) → join "key: value" lines
      - 3+ column reference table (e.g. countries) → join cells per row
    """
    rows_text: list[str] = []
    for row in tbl.rows:
        cells = [cell.text.strip() for cell in row.cells]
        cells = [c for c in cells if c]
        if not cells:
            continue
        if len(cells) == 2:
            rows_text.append(f"{cells[0]}: {cells[1]}")
        else:
            rows_text.append(" | ".join(cells))
    if not rows_text:
        return None

    body = "\n".join(rows_text)
    synth_q = f"Reference table from section: {section or 'general'}"
    if sub_section:
        synth_q = f"{sub_section} (reference table)"
    return {
        "question": synth_q,
        "answer": body,
        "source": "",
        "page": None,
        "section": section,
        "sub_section": sub_section,
        "kind": "table",
        "table_index": idx,
    }


def extract(docx_path: Path, source_name: str | None = None) -> list[dict]:
    if not docx_path.exists():
        raise FileNotFoundError(f"DOCX not found: {docx_path}")

    source_name = source_name or docx_path.name
    doc = Document(str(docx_path))

    pairs: list[dict] = []
    current = _Pair()
    section = ""
    sub_section = ""
    table_idx = 0
    seen_section = False

    def flush() -> None:
        nonlocal current
        if current.is_open():
            d = current.to_dict(source_name)
            if d:
                pairs.append(d)
        current = _Pair(section=section, sub_section=sub_section)

    for block in iter_block_items(doc):
        if isinstance(block, Table):
            # Tables may appear before the first SectionStyle (e.g. company
            # profile table). Always emit them — the metadata is useful even
            # without a section label.
            flush()
            card = _serialize_table(block, section, sub_section, table_idx)
            table_idx += 1
            if card:
                card["source"] = source_name
                pairs.append(card)
            continue

        # Paragraph
        text = (block.text or "").strip()
        style = block.style.name if block.style else ""

        if not text:
            continue
        if style in SKIP_STYLES:
            continue
        if not seen_section and style != "SectionStyle":
            continue

        if style == "SectionStyle":
            flush()
            section = text
            sub_section = ""
            seen_section = True
            current = _Pair(section=section, sub_section=sub_section)
            continue

        if style == "SubSectionStyle":
            flush()
            sub_section = text
            current = _Pair(section=section, sub_section=sub_section)
            current.question = sub_section
            continue

        if style == "QStyle":
            flush()
            current = _Pair(section=section, sub_section=sub_section)
            current.question = text
            continue

        if style == "AStyle":
            current.answer_parts.append(text)
            continue

        if style == "List Bullet" or style.startswith("List "):
            current.answer_parts.append(f"- {text}")
            continue

        current.answer_parts.append(text)

    flush()
    return pairs


def write_jsonl(pairs: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Q/A pairs from a Word FAQ document.")
    parser.add_argument("docx_path", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--source-name", default=None, help="Override source name (defaults to filename).")
    args = parser.parse_args()

    pairs = extract(args.docx_path, source_name=args.source_name)
    write_jsonl(pairs, args.output_jsonl)

    qa_pairs = sum(1 for p in pairs if p.get("kind") != "table")
    tables = sum(1 for p in pairs if p.get("kind") == "table")
    print(f"Extracted {len(pairs)} chunks → {args.output_jsonl}")
    print(f"  Q/A pairs: {qa_pairs}")
    print(f"  Table chunks: {tables}")
    sections = {p.get("section", "") for p in pairs if p.get("section")}
    print(f"  Distinct sections: {len(sections)}")


if __name__ == "__main__":
    main()
