"""Import restricted terms workbook into restricted_terms table.

Uses the same schedule workbook format as `import_product_price_list.py`:
headers on row 1 like "Schedule X drugs:", "Schedule H drugs:", "Schedule H1 drugs:".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from app.business.restricted_products import clear_restricted_terms_cache, normalize_term
from app.db.database import SessionLocal
from app.db.models import RestrictedTerm
from scripts.import_product_price_list import SCHEDULE_CATEGORIES, load_schedule_terms_by_category

load_dotenv()


def import_restricted_terms(path: Path, *, dry_run: bool = False) -> tuple[int, int]:
    terms_by_category = load_schedule_terms_by_category(path)
    incoming: dict[str, tuple[str, str | None]] = {}
    for category in SCHEDULE_CATEGORIES:
        for term in terms_by_category.get(category, set()):
            normalized = normalize_term(term)
            if len(normalized) < 4:
                continue
            incoming[normalized] = (term.strip(), category)

    if dry_run:
        print(
            f"--dry-run: parsed {len(incoming)} restricted terms "
            f"(X={len(terms_by_category['X'])} H={len(terms_by_category['H'])} H1={len(terms_by_category['H1'])})"
        )
        return 0, 0

    inserted = 0
    updated = 0
    db = SessionLocal()
    try:
        existing = {
            row.normalized_term: row
            for row in db.query(RestrictedTerm).all()
        }
        for normalized, (term, category) in incoming.items():
            row = existing.get(normalized)
            if row is None:
                db.add(
                    RestrictedTerm(
                        term=term,
                        normalized_term=normalized,
                        schedule_category=category,
                        source="schedule_xlsx",
                    )
                )
                inserted += 1
            else:
                changed = False
                if row.term != term:
                    row.term = term
                    changed = True
                if row.schedule_category != category:
                    row.schedule_category = category
                    changed = True
                if row.source != "schedule_xlsx":
                    row.source = "schedule_xlsx"
                    changed = True
                if changed:
                    updated += 1

        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    clear_restricted_terms_cache()
    return inserted, updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Import restricted terms into restricted_terms table.")
    parser.add_argument("schedule_xlsx", type=Path, help="Path to schedule workbook (.xlsx)")
    parser.add_argument("--dry-run", action="store_true", help="Parse only; no DB writes")
    args = parser.parse_args()

    try:
        inserted, updated = import_restricted_terms(args.schedule_xlsx, dry_run=args.dry_run)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if not args.dry_run:
        print(f"Done. inserted={inserted} updated={updated}")


if __name__ == "__main__":
    main()
