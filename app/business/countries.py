"""Country tiers for lead scoring and shipment compliance (client lists)."""

from __future__ import annotations

from typing import Literal

CountryTier = Literal["priority", "secondary", "excluded", "other", "missing"]

# --- Priority markets (15 pts): fast response, bulk push ---
PRIORITY_MARKET_NAMES: tuple[str, ...] = (
    "Australia",
    "Canada",
    "China",
    "France",
    "Greece",
    "Hong Kong",
    "Japan",
    "Jordan",
    "Kazakhstan",
    "Kenya",
    "Kuwait",
    "Latvia",
    "Lithuania",
    "Malaysia",
    "Mexico",
    "New Zealand",
    "Oman",
    "Philippines",
    "Qatar",
    "Romania",
    "Russia",
    "Saudi Arabia",
    "Singapore",
    "South Africa",
    "South Korea",
    "Switzerland",
    "Taiwan",
    "Thailand",
    "Tanzania",
    "United Arab Emirates",
    "United Kingdom",
    "United States",
    "Vietnam",
)

# --- Secondary markets (10 pts) ---
SECONDARY_MARKET_NAMES: tuple[str, ...] = (
    "Albania",
    "Algeria",
    "Andorra",
    "Angola",
    "Antigua and Barbuda",
    "Argentina",
    "Armenia",
    "Austria",
    "Azerbaijan",
    "Bahamas",
    "Bahrain",
    "Bangladesh",
    "Barbados",
    "Belarus",
    "Belgium",
    "Belize",
    "Benin",
    "Bolivia",
    "Bosnia and Herzegovina",
    "Botswana",
    "Brazil",
    "Brunei",
    "Bulgaria",
    "Burkina Faso",
    "Burundi",
    "Cambodia",
    "Cameroon",
    "Cape Verde",
    "Central African Republic",
    "Chad",
    "Chile",
    "Colombia",
    "Comoros",
    "Congo (Republic)",
    "Costa Rica",
    "Croatia",
    "Cuba",
    "Cyprus",
    "Czech Republic",
    "Denmark",
    "Djibouti",
    "Dominica",
    "Dominican Republic",
    "Ecuador",
    "Egypt",
    "El Salvador",
    "Equatorial Guinea",
    "Eritrea",
    "Estonia",
    "Eswatini",
    "Ethiopia",
    "Fiji",
    "Finland",
    "Gabon",
    "Gambia",
    "Georgia",
    "Germany",
    "Ghana",
    "Grenada",
    "Guatemala",
    "Guinea",
    "Guinea-Bissau",
    "Guyana",
    "Haiti",
    "Honduras",
    "Hungary",
    "Iceland",
    "Indonesia",
    "Ireland",
    "Italy",
    "Jamaica",
    "Kosovo",
    "Kyrgyzstan",
    "Laos",
    "Lebanon",
    "Lesotho",
    "Liberia",
    "Libya",
    "Liechtenstein",
    "Luxembourg",
    "Madagascar",
    "Malawi",
    "Maldives",
    "Mali",
    "Malta",
    "Marshall Islands",
    "Mauritania",
    "Mauritius",
    "Micronesia",
    "Moldova",
    "Monaco",
    "Mongolia",
    "Montenegro",
    "Morocco",
    "Mozambique",
    "Myanmar",
    "Namibia",
    "Nauru",
    "Netherlands",
    "Nicaragua",
    "Niger",
    "Nigeria",
    "North Macedonia",
    "Norway",
    "Palau",
    "Panama",
    "Papua New Guinea",
    "Paraguay",
    "Peru",
    "Poland",
    "Portugal",
    "Rwanda",
    "Saint Kitts and Nevis",
    "Saint Lucia",
    "Saint Vincent and the Grenadines",
    "Samoa",
    "San Marino",
    "Senegal",
    "Serbia",
    "Seychelles",
    "Sierra Leone",
    "Slovakia",
    "Slovenia",
    "Solomon Islands",
    "Somalia",
    "South Sudan",
    "Spain",
    "Sri Lanka",
    "Sudan",
    "Suriname",
    "Sweden",
    "Syria",
    "Tajikistan",
    "Timor-Leste",
    "Togo",
    "Tonga",
    "Trinidad and Tobago",
    "Tunisia",
    "Turkey",
    "Turkmenistan",
    "Tuvalu",
    "Uganda",
    "Ukraine",
    "Uruguay",
    "Uzbekistan",
    "Vanuatu",
    "Vatican City",
    "Venezuela",
    "Yemen",
    "Zambia",
    "Zimbabwe",
)

# --- Shipment excluded: polite reject (not scored as secondary) ---
SHIPMENT_EXCLUDED_NAMES: tuple[str, ...] = (
    "Iran",
    "Iraq",
    "Israel",
    "Afghanistan",
    "Pakistan",
    "Nepal",
    "Bhutan",
)

# Short codes and synonyms → canonical normalized name (must exist in a tier set)
COUNTRY_ALIASES: dict[str, str] = {
    "usa": "united states",
    "us": "united states",
    "u.s.": "united states",
    "u.s.a.": "united states",
    "america": "united states",
    "uk": "united kingdom",
    "great britain": "united kingdom",
    "britain": "united kingdom",
    "england": "united kingdom",
    "uae": "united arab emirates",
    "emirates": "united arab emirates",
    "ksa": "saudi arabia",
    "saudi": "saudi arabia",
    "hk": "hong kong",
    "hong kong sar": "hong kong",
    "korea": "south korea",
    "republic of korea": "south korea",
    "rok": "south korea",
    "czechia": "czech republic",
    "holland": "netherlands",
    "burma": "myanmar",
    "drc": "congo (republic)",
    "congo-brazzaville": "congo (republic)",
    "congo republic": "congo (republic)",
}


def _normalize(label: str) -> str:
    return " ".join((label or "").strip().lower().split())


def _build_lookup(names: tuple[str, ...]) -> frozenset[str]:
    return frozenset(_normalize(n) for n in names)


PRIORITY_MARKETS = _build_lookup(PRIORITY_MARKET_NAMES)
SECONDARY_MARKETS = _build_lookup(SECONDARY_MARKET_NAMES)
SHIPMENT_EXCLUDED = _build_lookup(SHIPMENT_EXCLUDED_NAMES)

# Longest names first for safe substring matching
_ALL_PRIORITY_SORTED = sorted(PRIORITY_MARKET_NAMES, key=len, reverse=True)
_ALL_SECONDARY_SORTED = sorted(SECONDARY_MARKET_NAMES, key=len, reverse=True)
_ALL_EXCLUDED_SORTED = sorted(SHIPMENT_EXCLUDED_NAMES, key=len, reverse=True)

SHIPMENT_EXCLUDED_REFUSAL = (
    "I'm sorry, we're unable to process orders or quotes for that destination due to "
    "export compliance requirements. Please contact our compliance team directly."
)


def resolve_country_label(country: str) -> str:
    """Normalize and apply aliases."""
    normalized = _normalize(country)
    if not normalized:
        return ""
    return COUNTRY_ALIASES.get(normalized, normalized)


def _matches_list(normalized: str, names_sorted: list[str], canonical: frozenset[str]) -> bool:
    if normalized in canonical:
        return True
    for name in names_sorted:
        key = _normalize(name)
        if normalized == key:
            return True
        if len(key) >= 4 and (key in normalized or normalized in key):
            return True
    return False


def classify_country(country: str | None) -> CountryTier:
    if not country or not str(country).strip():
        return "missing"

    resolved = resolve_country_label(country)
    if not resolved:
        return "missing"

    if _matches_list(resolved, _ALL_EXCLUDED_SORTED, SHIPMENT_EXCLUDED):
        return "excluded"
    if _matches_list(resolved, _ALL_PRIORITY_SORTED, PRIORITY_MARKETS):
        return "priority"
    if _matches_list(resolved, _ALL_SECONDARY_SORTED, SECONDARY_MARKETS):
        return "secondary"
    return "other"


def canonicalize_country(country: str) -> str | None:
    """Return uppercase canonical country name when *country* is a known market, else None."""
    resolved = resolve_country_label(country)
    if not resolved:
        return None
    if classify_country(resolved) in ("missing", "other"):
        return None
    return resolved.upper()


def is_shipment_excluded_country(country: str | None) -> bool:
    return classify_country(country) == "excluded"


def is_priority_market(country: str | None) -> bool:
    return classify_country(country) == "priority"


def is_secondary_market(country: str | None) -> bool:
    return classify_country(country) == "secondary"


def country_priority_points(country: str | None) -> int:
    """SOP §3.3: priority 15, secondary 10, missing 0, other 5."""
    tier = classify_country(country)
    if tier == "priority":
        return 15
    if tier == "secondary":
        return 10
    if tier == "missing":
        return 0
    if tier == "other":
        return 5
    return 0  # excluded handled separately
