"""Phase 1 UX — country picker, bulk order prompt, quantity picker, checkout form."""

from __future__ import annotations

import re

from app.integrations.whatsapp import send_interactive_list

SESSION_COUNTRY_PICKER_SENT = "country_picker_sent"
SESSION_SKIP_WELCOME_COMPOSE = "skip_welcome_compose"
SESSION_AWAITING_CUSTOM_COUNTRY = "awaiting_custom_country"

COUNTRY_PICKER_ROWS: list[dict[str, str]] = [
    {"id": "country_us", "title": "🇺🇸 USA", "description": "United States"},
    {"id": "country_ca", "title": "🇨🇦 Canada", "description": "Canada"},
    {"id": "country_uk", "title": "🇬🇧 UK", "description": "United Kingdom"},
    {"id": "country_au", "title": "🇦🇺 Australia", "description": "Australia"},
    {"id": "country_other", "title": "🌍 Other", "description": "Type your country"},
]

COUNTRY_ID_TO_NAME: dict[str, str] = {
    "country_us": "United States",
    "country_ca": "Canada",
    "country_uk": "United Kingdom",
    "country_au": "Australia",
}

COUNTRY_BUTTON_IDS = frozenset(COUNTRY_ID_TO_NAME.keys()) | {"country_other"}

QTY_BUTTON_MAP: dict[str, int] = {
    "qty_50": 50,
    "qty_100": 100,
    "qty_250": 250,
    "qty_500": 500,
    "qty_1000": 1000,
}

QTY_BUTTON_IDS = frozenset(QTY_BUTTON_MAP.keys()) | {"qty_custom"}

QUANTITY_PICKER_ROWS: list[dict[str, str]] = [
    {"id": "qty_50", "title": "50 units", "description": ""},
    {"id": "qty_100", "title": "100 units", "description": ""},
    {"id": "qty_250", "title": "250 units", "description": ""},
    {"id": "qty_500", "title": "500 units", "description": ""},
    {"id": "qty_1000", "title": "1000 units", "description": ""},
    {"id": "qty_custom", "title": "Custom amount", "description": "Type your quantity"},
]

BULK_LIST_PROMPT = (
    "📋 *Paste your product list* (one per line):\n\n"
    "Example:\n"
    "Metformin 500mg - 100\n"
    "Amoxicillin 500mg - 200\n\n"
    "Or send one product name to add individually."
)

_COUNTRY_PROMPT = "🌎 *Select your country* from the list below."
_COUNTRY_REMINDER = "Please select your country from the list above 👆"
_CUSTOM_COUNTRY_PROMPT = "Please type your country name:"

_BULK_LINE_RE = re.compile(
    r"^(?P<name>.+?)\s*(?:[-–—]\s*|\s+[x×]\s*|\s+)(?P<qty>\d+)\s*$",
    re.IGNORECASE,
)


def country_prompt(*, reminded: bool = False) -> str:
    return _COUNTRY_REMINDER if reminded else _COUNTRY_PROMPT


def custom_country_prompt() -> str:
    return _CUSTOM_COUNTRY_PROMPT


def resolve_country_button(text: str) -> tuple[str | None, str | None]:
    """Return (canonical_country, follow_up_prompt) for list/button ids."""
    key = (text or "").strip().lower()
    if key in COUNTRY_ID_TO_NAME:
        return COUNTRY_ID_TO_NAME[key], None
    if key == "country_other":
        return None, custom_country_prompt()
    return None, None


def checkout_prompt(country: str) -> str:
    ship = country or "your country"
    return (
        "Almost done! 🎉\n\n"
        f"Ship to: *{ship}*\n\n"
        "Reply in *one message* with your details:\n"
        "*Name, City, Phone*\n\n"
        "Example: Jane Doe, Sydney, +61412345678"
    )


def parse_checkout_oneline(text: str, default_country: str | None) -> dict[str, str] | None:
    """Parse 'Name, City, Phone' (or 'Name, Company, City')."""
    parts = [p.strip() for p in (text or "").split(",") if p.strip()]
    if len(parts) < 2:
        return None

    country = (default_country or "").strip()
    if len(parts) == 2:
        contact, city = parts[0], parts[1]
    else:
        contact = parts[0]
        city = parts[1]
        phone = ", ".join(parts[2:])
        if phone:
            contact = f"{contact} ({phone})"

    if len(contact) < 2 or len(city) < 2:
        return None

    result: dict[str, str] = {"contact": contact, "city": city}
    if country:
        result["country"] = country
    return result


def looks_like_bulk_order(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    if "\n" in stripped:
        return True
    parts = [p.strip() for p in re.split(r"[,;]", stripped) if p.strip()]
    if len(parts) >= 2:
        return True
    return bool(_BULK_LINE_RE.match(stripped))


def parse_bulk_order_lines(text: str) -> list[tuple[str, int | None]]:
    """Return (product_query, quantity or None) for each line/segment."""
    items: list[tuple[str, int | None]] = []
    for raw_line in (text or "").split("\n"):
        segments = [s.strip() for s in re.split(r"[,;]", raw_line) if s.strip()]
        for segment in segments:
            match = _BULK_LINE_RE.match(segment)
            if match:
                items.append((match.group("name").strip(), int(match.group("qty"))))
            else:
                items.append((segment, None))
    return items


async def send_country_picker(phone: str, session: dict) -> dict:
    """Send country list once per session."""
    session = dict(session or {})
    if session.get(SESSION_COUNTRY_PICKER_SENT) or not phone:
        return session

    await send_interactive_list(
        phone,
        header_text="New Life Medicare",
        body_text="Welcome! Select your country to get started.",
        footer_text="Pharmaceutical exports worldwide",
        button_text="Select Country",
        rows=COUNTRY_PICKER_ROWS,
        section_title="Countries",
    )
    session[SESSION_COUNTRY_PICKER_SENT] = True
    session[SESSION_SKIP_WELCOME_COMPOSE] = True
    return session


async def send_quantity_picker(phone: str, product_name: str) -> bool:
    """Send quantity list for a matched product."""
    if not phone:
        return False
    safe_name = (product_name or "Product")[:40]
    return await send_interactive_list(
        phone,
        header_text="Select quantity",
        body_text=f"Choose quantity for *{safe_name}*",
        footer_text="Tap an option below",
        button_text="Quantities",
        rows=QUANTITY_PICKER_ROWS,
        section_title="Units",
    )
