"""Country tier lists — client priority / secondary / excluded."""

from app.business.countries import (
    PRIORITY_MARKETS,
    SECONDARY_MARKETS,
    SHIPMENT_EXCLUDED,
    classify_country,
    country_priority_points,
    is_priority_market,
    is_shipment_excluded_country,
)


def test_priority_secondary_excluded_sets_disjoint():
    assert not (PRIORITY_MARKETS & SECONDARY_MARKETS)
    assert not (PRIORITY_MARKETS & SHIPMENT_EXCLUDED)
    assert not (SECONDARY_MARKETS & SHIPMENT_EXCLUDED)


def test_list_counts_match_client_master():
    from app.business.countries import SECONDARY_MARKET_NAMES

    assert len(PRIORITY_MARKETS) == 33
    assert len(SECONDARY_MARKET_NAMES) == len(SECONDARY_MARKETS) == 151
    assert len(SHIPMENT_EXCLUDED) == 7


def test_priority_markets_and_aliases():
    assert classify_country("United States") == "priority"
    assert classify_country("USA") == "priority"
    assert classify_country("UAE") == "priority"
    assert classify_country("United Kingdom") == "priority"
    assert classify_country("Russia") == "priority"
    assert country_priority_points("Australia") == 15


def test_secondary_markets():
    assert classify_country("Germany") == "secondary"
    assert classify_country("Nigeria") == "secondary"
    assert classify_country("India") == "other"
    assert country_priority_points("Germany") == 10


def test_shipment_excluded_reject_list():
    for country in (
        "Iran",
        "Iraq",
        "Israel",
        "Afghanistan",
        "Pakistan",
        "Nepal",
        "Bhutan",
    ):
        assert classify_country(country) == "excluded"
        assert is_shipment_excluded_country(country)
        assert country_priority_points(country) == 0


def test_pakistan_not_secondary():
    assert classify_country("Pakistan") == "excluded"
    assert "pakistan" not in SECONDARY_MARKETS


def test_kenya_priority_not_secondary():
    assert classify_country("Kenya") == "priority"
    assert "kenya" not in SECONDARY_MARKETS
