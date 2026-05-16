"""Diagnostic: print the structural shape of a PDF file.

Usage:
    python -m scripts.inspect_pdf "data/faq_raw/some.pdf"

Prints:
  - page count
  - whether text extraction returns real text (vs scanned image PDF)
  - regex-marker hit counts
  - whether pages contain tables (via pdfplumber)
  - first three lines of page 1
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

try:
    import pdfplumber
    HAS_PLUMBER = True
except ImportError:
    HAS_PLUMBER = False

from scripts._markers import is_question_line, is_answer_line


def _classify_extraction_quality(full_text: str, page_count: int) -> str:
    """Heuristic: a sane text PDF averages 80+ chars per page."""
    if not full_text.strip():
        return "EMPTY (likely scanned image PDF — OCR needed)"
    avg = len(full_text) / max(page_count, 1)
    if avg < 50:
        return f"SUSPICIOUSLY SHORT (~{avg:.0f} chars/page) — may be scanned"
    if avg < 200:
        return f"SHORT (~{avg:.0f} chars/page) — likely usable but verify"
    return f"GOOD (~{avg:.0f} chars/page)"


def inspect(path: Path) -> None:
    print(f"\n=== INSPECT PDF: {path.name} ===")
    if not path.exists():
        print(f"  ERROR: file not found: {path}")
        sys.exit(1)

    reader = PdfReader(str(path))
    pages = reader.pages
    page_count = len(pages)
    print(f"Pages: {page_count}")

    full_text_parts: list[str] = []
    marker_hits: Counter[str] = Counter()
    qmark_lines = 0
    samples: dict[str, list[str]] = {"Q_marker": [], "A_marker": [], "qmark_line": []}
    page1_lines: list[str] = []

    for i, page in enumerate(pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:  # noqa: BLE001
            text = ""
            print(f"  page {i + 1}: extraction error: {e}")
        full_text_parts.append(text)

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if i == 0 and len(page1_lines) < 10:
                page1_lines.append(line)
            if is_question_line(line):
                marker_hits["Q_marker"] += 1
                if len(samples["Q_marker"]) < 3:
                    samples["Q_marker"].append(line[:120])
            if is_answer_line(line):
                marker_hits["A_marker"] += 1
                if len(samples["A_marker"]) < 3:
                    samples["A_marker"].append(line[:120])
            if line.endswith("?"):
                qmark_lines += 1
                if len(samples["qmark_line"]) < 3:
                    samples["qmark_line"].append(line[:120])

    full_text = "\n".join(full_text_parts)
    quality = _classify_extraction_quality(full_text, page_count)
    print(f"Text extraction quality: {quality}")

    print("\nMarker regex hits:")
    print(f"  Q-style markers: {marker_hits.get('Q_marker', 0)}")
    print(f"  A-style markers: {marker_hits.get('A_marker', 0)}")
    print(f"  Lines ending with '?': {qmark_lines}")

    if HAS_PLUMBER:
        try:
            with pdfplumber.open(str(path)) as pdf:
                pages_with_tables = 0
                table_shapes: list[tuple[int, int, int]] = []
                for i, p in enumerate(pdf.pages):
                    tables = p.extract_tables() or []
                    if tables:
                        pages_with_tables += 1
                        for t in tables:
                            rows = len(t)
                            cols = len(t[0]) if rows else 0
                            table_shapes.append((i + 1, rows, cols))
                print(f"\nTables (via pdfplumber): {len(table_shapes)} across {pages_with_tables} pages")
                for pg, r, c in table_shapes[:5]:
                    print(f"  page {pg}: {r} rows x {c} cols")
        except Exception as e:  # noqa: BLE001
            print(f"  pdfplumber error: {e}")
    else:
        print("\n(pdfplumber unavailable — table detection skipped)")

    print("\nSample lines per detected pattern:")
    for key, items in samples.items():
        if items:
            print(f"  [{key}]")
            for s in items:
                print(f"    - {s}")

    if page1_lines:
        print("\nFirst lines of page 1:")
        for line in page1_lines:
            print(f"  | {line[:100]}")

    print("\n--- STRATEGY RECOMMENDATION ---")
    if "EMPTY" in quality:
        print("  PDF appears to be scanned. OCR (pytesseract + pdf2image + Poppler) required.")
    elif marker_hits.get("Q_marker", 0) >= 3:
        print(f"  Recommended: markers strategy ({marker_hits['Q_marker']} Q-markers found)")
    elif qmark_lines >= 3:
        print(f"  Recommended: 'qmark_then_paragraph' fallback ({qmark_lines} candidate questions)")
    else:
        print("  No strong Q/A pattern. Treat as prose; consider splitting by headings or sections.")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.inspect_pdf <path-to-pdf> [more paths...]")
        sys.exit(2)
    for arg in sys.argv[1:]:
        inspect(Path(arg))


if __name__ == "__main__":
    main()
