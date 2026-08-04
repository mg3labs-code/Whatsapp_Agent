from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.business.restricted_products import (
    clear_restricted_terms_cache,
    match_restricted_term,
    normalize_term,
)
from app.db.models import Base, RestrictedTerm
from scripts.import_restricted_terms import import_restricted_terms


def _db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def test_normalize_term_collapses_spaces():
    assert normalize_term("  Tramadol   Hydrochloride  ") == "tramadol hydrochloride"


def test_match_restricted_term_prefers_longer_term():
    db = _db()
    try:
        db.add_all(
            [
                RestrictedTerm(
                    term="Tramadol",
                    normalized_term="tramadol",
                    schedule_category="H",
                    source="test",
                ),
                RestrictedTerm(
                    term="Tramadol Hydrochloride",
                    normalized_term="tramadol hydrochloride",
                    schedule_category="H1",
                    source="test",
                ),
            ]
        )
        db.commit()
        clear_restricted_terms_cache()

        hit = match_restricted_term("Price for Tramadol Hydrochloride 50mg", db)
        assert hit is not None
        assert hit.term == "Tramadol Hydrochloride"
        assert hit.schedule_category == "H1"
    finally:
        db.close()


def test_import_restricted_terms_dry_run_works_with_sample():
    # Covers script entry logic without DB write; sample is optional in repo.
    from pathlib import Path

    sample = Path(__file__).resolve().parent.parent / "data" / "import_samples" / "schedule_hx.xlsx"
    if not sample.is_file():
        return
    inserted, updated = import_restricted_terms(sample, dry_run=True)
    assert inserted == 0
    assert updated == 0
