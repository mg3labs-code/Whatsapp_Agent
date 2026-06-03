"""Country parsing guards for shortened qualification."""

from app.agents.qualification import _extract_country, _is_plausible_country


def test_is_plausible_country_rejects_pricing_sentence():
    assert _is_plausible_country("Hi, I need price for Amoxicillin 500mg") is False


def test_is_plausible_country_accepts_kenya():
    assert _is_plausible_country("Kenya") is True


def test_extract_country_from_usa():
    assert _extract_country("from USA") == "USA"
