"""WhatsApp conversation presentation helpers (formatting + navigation).

These helpers change how replies look and feel in chat without altering agent
business logic, routing rules, or qualification/order state machines.
"""

from __future__ import annotations

from app.integrations.whatsapp import send_interactive_list

MAIN_MENU_ID = "main_menu"
MY_ORDERS_ID = "my_orders"
SPEAK_MENU_ID = "speak"

MENU_OPTION_IDS = frozenset({"order", "pricing", "faq", MY_ORDERS_ID})

# Backward-compatible alias used by router/qualification imports.
MENU_BUTTON_IDS = MENU_OPTION_IDS

MENU_ACK_IDS = MENU_OPTION_IDS | {SPEAK_MENU_ID}

SESSION_PENDING_MENU_ACK = "pending_menu_ack"
SESSION_SUPPRESS_NAV_FOOTER = "suppress_nav_footer"

NAV_FOOTER_BODY = "Anything else?"

MAIN_MENU_BODY = "How can I help you today? Select an option below 👇"

MAIN_MENU_LIST_ROWS: list[dict[str, str]] = [
    {
        "id": "order",
        "title": "Place an Order",
        "description": "Start a new purchase order",
    },
    {
        "id": MY_ORDERS_ID,
        "title": "My Orders",
        "description": "Past orders, payment & shipping status",
    },
    {
        "id": "pricing",
        "title": "Get Pricing",
        "description": "Product quotes and MOQ",
    },
    {
        "id": "faq",
        "title": "FAQs",
        "description": "Shipping, docs and policies",
    },
    {
        "id": SPEAK_MENU_ID,
        "title": "Speak to Team",
        "description": "Connect with our export team",
    },
]

MAIN_MENU_BUTTON = [{"id": MAIN_MENU_ID, "title": "Main Menu"}]

MENU_SELECTION_ACK: dict[str, str] = {
    "order": (
        "You selected *Place an Order* 📦\n\n"
        "Let's get started! Which product(s) would you like to order?"
    ),
    MY_ORDERS_ID: (
        "You selected *My Orders* 📋\n\n"
        "Here is your order history for this number:"
    ),
    "pricing": (
        "You selected *Get Pricing* 💰\n\n"
        "Please share the product name and quantity you need a quote for."
    ),
    "faq": (
        "You selected *FAQs* ❓\n\n"
        "What would you like to know about shipping, documents, or policies?"
    ),
    SPEAK_MENU_ID: (
        "You selected *Speak to Team* 👤\n\n"
        "Connecting you with our export team."
    ),
}


async def send_main_menu_list(phone: str, *, body: str | None = None) -> bool:
    """Send the full main menu (list supports more than 3 reply buttons)."""
    if not phone:
        return False
    return await send_interactive_list(
        phone,
        header_text="New Life Medicare",
        body_text=body or MAIN_MENU_BODY,
        footer_text="Pharmaceutical exports worldwide 🌍",
        button_text="View Options",
        rows=MAIN_MENU_LIST_ROWS,
        section_title="Menu",
    )


def is_menu_option(message: str) -> bool:
    return (message or "").strip().lower() in MENU_OPTION_IDS


def is_main_menu_request(message: str) -> bool:
    return (message or "").strip().lower() == MAIN_MENU_ID


def mark_menu_selection(session: dict, message: str) -> dict:
    """Remember a main-menu option tap so send_reply can prepend a selection ack."""
    session = dict(session or {})
    key = (message or "").strip().lower()
    if key in MENU_ACK_IDS:
        session[SESSION_PENDING_MENU_ACK] = key
    return session


def apply_menu_selection_ack(reply: str, session: dict) -> tuple[str, dict]:
    """Prepend a screenshot-style selection acknowledgment once per menu tap."""
    session = dict(session or {})
    key = session.pop(SESSION_PENDING_MENU_ACK, None)
    body = (reply or "").strip()
    if not key:
        return reply, session

    ack = MENU_SELECTION_ACK.get(key, "")
    if not ack:
        return reply, session

    if not body:
        return ack, session

    # Avoid repeating the same question when the agent already asks for products.
    if key == "order" and any(
        token in body.lower() for token in ("product", "sku", "which product", "add")
    ):
        return f"You selected *Place an Order* 📦\n\n{body}", session

    if key == "pricing" and any(
        token in body.lower() for token in ("product", "quote", "price", "which")
    ):
        return f"You selected *Get Pricing* 💰\n\n{body}", session

    if key == MY_ORDERS_ID and body.startswith("📊"):
        return f"You selected *My Orders* 📋\n\n{body}", session

    if body.startswith(ack.split("\n\n")[0]):
        return body, session

    return f"{ack}\n\n{body}", session


def should_send_navigation_footer(session: dict) -> bool:
    """Show 'Anything else?' only after qualification/order flows — not mid-qual."""
    session = session or {}
    if session.get("human_active"):
        return False
    if session.get(SESSION_SUPPRESS_NAV_FOOTER):
        return False
    if session.get("qual_state"):
        return False
    if session.get("order_state"):
        return False
    if not session.get("lead_qualified"):
        return False
    return True
