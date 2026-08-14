"""Phase 1 UX — country picker, bulk order prompt, checkout form."""

from __future__ import annotations

import re

from app.integrations.whatsapp import send_interactive_list
from app.messages.welcome import prepend_ai_disclosure

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

# Canonical order-start + multi-product format (strips). Use everywhere buyers start ordering.
BULK_LIST_PROMPT = (
    "📋 *Add products to your order:*\n\n"
    "• Send a *product name* — we'll ask how many *strips*\n"
    "• Or send *Product name - strips* (one line or many)\n\n"
    "*Example:*\n"
    "JGLUT 2000MG 30ML - 350\n"
    "Metformin 500mg - 100\n"
    "Amoxicillin 500mg - 200"
)

# Alias for a single consistent order-entry path.
ORDER_START_PROMPT = BULK_LIST_PROMPT


def product_qty_prompt(product_name: str) -> str:
    """Ask for strip quantity when a product was matched without a qty."""
    safe = (product_name or "Product").strip()
    return (
        f"Found: *{safe}*\n\n"
        f"How many *strips* do you need?\n"
        f"Reply with a *number only* (example: *350*)."
    )


# Buyers sometimes echo the old "full line" instruction instead of typing a qty.
_QTY_INSTRUCTION_ECHOES = frozenset(
    {
        "full line",
        "fullline",
        "full-line",
        "fulliline",
        "quantity only",
        "qty only",
        "reply with quantity",
        "reply with quantity only",
    }
)


def is_qty_instruction_echo(message: str) -> bool:
    """True when the buyer echoed qty-prompt instructions instead of a number."""
    key = (message or "").strip().lower()
    if not key:
        return False
    if key in _QTY_INSTRUCTION_ECHOES:
        return True
    # Soft match: message is only the words "full" + "line" (any separators).
    compact = re.sub(r"[\s\-_/]+", " ", key).strip()
    return compact in _QTY_INSTRUCTION_ECHOES


_COUNTRY_PROMPT = "🌎 *Select your country* from the list below."
_COUNTRY_REMINDER = "Please select your country from the list above 👆"
_CUSTOM_COUNTRY_PROMPT = "Please type your country name:"

# Explicit order qty: "Product - 100" / "Product x 100" / "Product × 100"
_BULK_SEP_RE = re.compile(
    r"^(?P<name>.+?)\s*(?:[-–—]\s*|\s+[x×]\s+)(?P<qty>\d[\d,]*)\s*$",
    re.IGNORECASE,
)
_TRAILING_BARE_QTY_RE = re.compile(r"^(?P<name>.+?)\s+(?P<qty>\d[\d,]*)\s*$")
_ORDER_UNITS_QTY_RE = re.compile(
    r"^(?P<qty>\d[\d,]*)\s*(?:units?|pcs?|pieces?|strips?|tablets?|tabs?)\s*$",
    re.IGNORECASE,
)
_BARE_QTY_RE = re.compile(r"^\d[\d,]*$")
# Dose / pack tokens — digits here must never be treated as order quantity
_DOSE_OR_PACK_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:mg|mcg|ug|µg|g|ml|iu|i\.u\.|%|mcg/ml|mg/ml)"
    r"|\d+\s*[x×*]\s*\d+",
    re.IGNORECASE,
)


def _name_contains_dose_or_pack(name: str) -> bool:
    return bool(_DOSE_OR_PACK_RE.search(name or ""))


def _parse_qty_int(raw: str) -> int | None:
    try:
        value = int((raw or "").replace(",", ""))
    except ValueError:
        return None
    return value if value >= 1 else None


def is_bare_order_qty(text: str) -> int | None:
    """Return qty when the whole message is clearly an order quantity (not a dose)."""
    stripped = (text or "").strip()
    if not stripped:
        return None
    if _BARE_QTY_RE.fullmatch(stripped.replace(",", "")) or _BARE_QTY_RE.fullmatch(
        stripped
    ):
        return _parse_qty_int(stripped)
    units = _ORDER_UNITS_QTY_RE.fullmatch(stripped)
    if units:
        return _parse_qty_int(units.group("qty"))
    return None


def _parse_product_qty_segment(segment: str) -> tuple[str, int | None]:
    """Parse product query + optional *explicit* order qty (dose-safe).

    Quantity is accepted only when clearly an order qty:
    - separator: 'Product - 100', 'Product x 100'
    - trailing bare number only if the name already contains dose/pack units
      (e.g. 'JGLUT 2000MG 30ML 350') — never 'Arkacan 100mcg' or dose digits alone
    """
    text = (segment or "").strip()
    if not text:
        return "", None

    sep = _BULK_SEP_RE.match(text)
    if sep:
        qty = _parse_qty_int(sep.group("qty"))
        name = sep.group("name").strip()
        if qty is not None and name:
            return name, qty

    trailing = _TRAILING_BARE_QTY_RE.match(text)
    if trailing:
        name = trailing.group("name").strip()
        qty = _parse_qty_int(trailing.group("qty"))
        # Require dose/pack in the name so 'Arkacan 100' / '100mcg' are not qty.
        if qty is not None and len(name) >= 3 and _name_contains_dose_or_pack(name):
            return name, qty

    return text, None


def parse_order_line(text: str) -> tuple[str, int | None]:
    """Public helper: (product_query, explicit_order_qty_or_none)."""
    return _parse_product_qty_segment(text)


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
        "Reply in *one message* with:\n"
        "*Name, City*\n\n"
        "Example: Jane Doe, Sydney\n\n"
        "We'll use your WhatsApp number for contact."
    )


def parse_checkout_oneline(
    text: str,
    default_country: str | None,
    *,
    whatsapp_phone: str | None = None,
) -> dict[str, str] | None:
    """Parse 'Name, City'. Optional trailing phone is ignored; WhatsApp phone is used."""
    parts = [p.strip() for p in (text or "").split(",") if p.strip()]
    if len(parts) < 2:
        return None

    country = (default_country or "").strip()
    contact = parts[0]
    city = parts[1]
    # Ignore extra comma fields (legacy Name, City, Phone) — WA number is source of truth.
    wa = (whatsapp_phone or "").strip()
    if wa:
        contact = f"{contact} ({wa})"

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
    _, qty = _parse_product_qty_segment(stripped)
    return qty is not None


def parse_bulk_order_lines(text: str) -> list[tuple[str, int | None]]:
    """Return (product_query, quantity or None) for each line/segment."""
    items: list[tuple[str, int | None]] = []
    for raw_line in (text or "").split("\n"):
        segments = [s.strip() for s in re.split(r"[,;]", raw_line) if s.strip()]
        for segment in segments:
            items.append(_parse_product_qty_segment(segment))
    return items


async def send_country_picker(phone: str, session: dict) -> dict:
    """Send country list once per session."""
    session = dict(session or {})
    if session.get(SESSION_COUNTRY_PICKER_SENT) or not phone:
        return session

    body_text, session = prepend_ai_disclosure(
        "Welcome! Select your country to get started.",
        session,
    )
    await send_interactive_list(
        phone,
        header_text="New Life Medicare",
        body_text=body_text,
        footer_text="Pharmaceutical exports worldwide",
        button_text="Select Country",
        rows=COUNTRY_PICKER_ROWS,
        section_title="Countries",
    )
    session[SESSION_COUNTRY_PICKER_SENT] = True
    session[SESSION_SKIP_WELCOME_COMPOSE] = True
    return session
