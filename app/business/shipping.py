"""Shipping weight, rate lookup, and WhatsApp message formatting."""

from __future__ import annotations

import logging
import re
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Common country aliases not in the Excel — maps buyer input to Excel name
_COUNTRY_ALIASES: dict[str, str] = {
    "UAE": "UNITED ARAB EMIRATES",
    "UK": "UNITED KINGDOM",
    "USA": "UNITED STATES OF AMERICA",
    "US": "UNITED STATES OF AMERICA",
    "AMERICA": "UNITED STATES OF AMERICA",
    "KSA": "SAUDI ARABIA",
    "SOUTH KOREA": "REPUBLIC OF KOREA",
    "KOREA": "REPUBLIC OF KOREA",
    "RUSSIA": "RUSSIAN FEDERATION",
    "VIETNAM": "VIET NAM",
    "TANZANIA": "UNITED REPUBLIC OF TANZANIA",
    "IVORY COAST": "COTE D IVORE",
    "COTE D'IVOIRE": "COTE D IVORE",
    "CZECHIA": "CZECHIA",
    "CZECH REPUBLIC": "CZECHIA",
    "CONGO": "DEMOCRATIC REPUBLIC OF THE CONGO",
    "TRINIDAD": "TRINIDAD AND TOBAGO",
    "TRINIDAD & TOBAGO": "TRINIDAD AND TOBAGO",
    "LAOS": "LAO",
    "MYANMAR": "MYANMAR",
    "BURMA": "MYANMAR",
}


def _resolve_country(buyer_input: str) -> str:
    """Normalise buyer country input to match the Excel country name."""
    upper = str(buyer_input or "").strip().upper()

    # 1. Direct alias lookup
    if upper in _COUNTRY_ALIASES:
        return _COUNTRY_ALIASES[upper]

    # 2. WASA country canonicalisation (aliases in app.business.countries)
    try:
        from app.business.countries import resolve_country_label

        canonical = resolve_country_label(buyer_input)
        if canonical:
            return canonical.upper()
    except (ImportError, Exception):
        pass

    # 3. Return normalised input — let SQL UPPER() handle it
    return upper


def _fuzzy_match_shipping_country(db: Session, resolved: str) -> str | None:
    """Fallback: closest country_name in shipping_rates (≥85% similarity)."""
    from difflib import get_close_matches

    try:
        rows = db.execute(
            text("SELECT DISTINCT UPPER(country_name) AS name FROM shipping_rates")
        ).fetchall()
        names = [str(r[0]) for r in rows if r[0]]
        if not names:
            return None
        matches = get_close_matches(resolved, names, n=1, cutoff=0.85)
        return matches[0] if matches else None
    except Exception:
        logger.exception("fuzzy shipping country match failed resolved=%r", resolved)
        return None


def _fetch_shipping_rate_rows(
    db: Session, resolved: str, weight_g: int
) -> list[tuple[Any, Any]]:
    return db.execute(
        text(
            """
            SELECT shipping_type, rate_usd
            FROM shipping_rates
            WHERE UPPER(country_name) = :resolved
              AND weight_from_g <= :weight_g
              AND weight_to_g >= :weight_g
              AND shipping_type IN ('EMS', 'LP')
            """
        ),
        {"resolved": resolved, "weight_g": weight_g},
    ).fetchall()

_CATEGORY_DEFAULTS_G: dict[str, float] = {
    "tablet": 15,
    "strip": 15,
    "injection": 25,
    "vial": 25,
    "ointment": 30,
    "cream": 30,
    "gel": 30,
    "capsule": 12,
    "syrup": 80,
    "solution": 80,
    "eye_drop": 20,
    "ear_drop": 20,
    "default": 20,
}

# Box selection by total unit count (sum of cart qty)
_BOX_BY_ITEM_COUNT: tuple[tuple[int, int, str], ...] = (
    (1, 10, "3"),
    (11, 30, "7"),
    (31, 60, "9"),
    (61, 100, "11"),
    (101, 10_000_000, "13"),
)

# Fallback box weights (grams) if box_specs row missing
_BOX_FALLBACK_G: dict[str, float] = {
    "3": 64.0,
    "7": 97.0,
    "9": 150.0,
    "11": 243.0,
    "13": 195.0,
}

_EMS_LABEL = "Express – EMS"
_LP_LABEL = "Normal – LP"
_EMS_DAYS = "7-14 days"
_LP_DAYS = "15-30 days"

_SKU_RE = re.compile(r"^PROD-(\d+)$", re.IGNORECASE)


def _detect_form_category(product_name: str) -> str:
    """Map product name keywords to a category key for default weight."""
    name = (product_name or "").upper()
    if any(k in name for k in ("INJ", "INJECTION", "VIAL", "AMP")):
        return "injection"
    if any(k in name for k in ("OINT", "OINTMENT")):
        return "ointment"
    if "CREAM" in name:
        return "cream"
    if "GEL" in name:
        return "gel"
    if any(k in name for k in ("CAP", "CAPS", "SOFTGEL")):
        return "capsule"
    if any(k in name for k in ("SYR", "SYRUP", "SUSP")):
        return "syrup"
    if "SOLUTION" in name:
        return "solution"
    if any(k in name for k in ("EYE DROP", "EYE-DROP", "EYEDROP")):
        return "eye_drop"
    if any(k in name for k in ("EAR DROP", "EAR-DROP", "EARDROP")):
        return "ear_drop"
    if any(k in name for k in ("TAB", "STRIP", "TABLET")):
        return "tablet"
    return "default"


def _default_weight_g(product_name: str) -> float:
    category = _detect_form_category(product_name)
    return _CATEGORY_DEFAULTS_G.get(category, _CATEGORY_DEFAULTS_G["default"])


def _select_box_no(total_units: int) -> str:
    count = max(total_units, 1)
    for low, high, box_no in _BOX_BY_ITEM_COUNT:
        if low <= count <= high:
            return box_no
    return "13"


def _lookup_box_weight_g(db: Session, box_no: str) -> float:
    try:
        row = db.execute(
            text("SELECT weight_g FROM box_specs WHERE box_no = :box_no LIMIT 1"),
            {"box_no": box_no},
        ).fetchone()
        if row and row[0] is not None:
            return float(row[0])
    except Exception:
        logger.exception("box_specs lookup failed for box_no=%s", box_no)
    return _BOX_FALLBACK_G.get(box_no, _BOX_FALLBACK_G["13"])


def _lookup_product_weight_g(
    db: Session,
    *,
    sku: str | None,
    product_name: str | None,
) -> tuple[float | None, bool]:
    """
    Return (weight_g, used_default).
    weight_g is None if product not found in DB.
    used_default True when DB weight is 0/NULL and category default applied.
    """
    try:
        row = None
        sku_val = (sku or "").strip()
        sku_match = _SKU_RE.match(sku_val)
        if sku_match:
            row = db.execute(
                text(
                    "SELECT weight_g FROM products WHERE id = :id LIMIT 1"
                ),
                {"id": int(sku_match.group(1))},
            ).fetchone()

        if row is None and product_name:
            row = db.execute(
                text(
                    """
                    SELECT weight_g FROM products
                    WHERE UPPER(TRIM(product_name)) = UPPER(TRIM(:name))
                    ORDER BY id
                    LIMIT 1
                    """
                ),
                {"name": product_name.strip()},
            ).fetchone()

        if row is None:
            return None, False

        raw = row[0]
        if raw is not None and float(raw) > 0:
            return float(raw), False

        label = product_name or sku_val or "UNKNOWN"
        return _default_weight_g(label), True
    except Exception:
        logger.exception(
            "product weight lookup failed sku=%r product_name=%r",
            sku,
            product_name,
        )
        if product_name or sku:
            return _default_weight_g(product_name or sku or ""), True
        return None, False


def calculate_cart_weight(cart_items: list[dict], db: Session) -> dict[str, Any] | None:
    """
    Compute total product weight plus packaging box weight (all grams).

    cart_items: [{"sku": "PROD-0001", "product_name": "AMLIP 10MG", "qty": 50}, ...]
    """
    try:
        if not cart_items:
            return {
                "total_product_g": 0,
                "box_no": "3",
                "box_weight_g": _lookup_box_weight_g(db, "3"),
                "total_shipment_g": _lookup_box_weight_g(db, "3"),
                "items_missing_weight": [],
            }

        total_product_g = 0.0
        total_units = 0
        items_missing_weight: list[str] = []

        for item in cart_items:
            qty = item.get("qty", item.get("quantity", 0))
            try:
                qty_int = int(qty)
            except (TypeError, ValueError):
                qty_int = 0
            if qty_int <= 0:
                continue

            sku = item.get("sku")
            product_name = (item.get("product_name") or "").strip()
            label = product_name or str(sku or "UNKNOWN")

            weight_g, used_default = _lookup_product_weight_g(
                db,
                sku=str(sku) if sku is not None else None,
                product_name=product_name or None,
            )
            if weight_g is None:
                weight_g = _default_weight_g(label)
                used_default = True

            if used_default:
                items_missing_weight.append(label)

            total_product_g += weight_g * qty_int
            total_units += qty_int

        box_no = _select_box_no(total_units)
        box_weight_g = _lookup_box_weight_g(db, box_no)
        total_shipment_g = round(total_product_g + box_weight_g)

        return {
            "total_product_g": round(total_product_g),
            "box_no": box_no,
            "box_weight_g": round(box_weight_g),
            "total_shipment_g": int(total_shipment_g),
            "items_missing_weight": items_missing_weight,
        }
    except Exception:
        logger.exception("calculate_cart_weight failed")
        return None


def get_shipping_options(country: str, total_g: int, db: Session) -> dict[str, Any]:
    """Look up EMS/LP rates for country and total shipment weight (grams)."""
    fallback: dict[str, Any] = {
        "available": False,
        "country": (country or "").strip().upper(),
        "weight_g": int(total_g),
        "EMS": None,
        "LP": None,
    }
    try:
        country_clean = (country or "").strip()
        if not country_clean:
            return fallback

        weight_g = max(int(total_g), 0)
        resolved = _resolve_country(country_clean)
        logger.debug(
            "Shipping country lookup: input=%r resolved=%r",
            country_clean,
            resolved,
        )

        rows = _fetch_shipping_rate_rows(db, resolved, weight_g)

        ems_rate: float | None = None
        lp_rate: float | None = None
        for shipping_type, rate_usd in rows:
            if rate_usd is None:
                continue
            rate = float(rate_usd)
            if shipping_type == "EMS":
                ems_rate = rate
            elif shipping_type == "LP":
                lp_rate = rate

        if ems_rate is None and lp_rate is None:
            fuzzy = _fuzzy_match_shipping_country(db, resolved)
            if fuzzy and fuzzy != resolved:
                logger.debug(
                    "Shipping country fuzzy match: resolved=%r fuzzy=%r",
                    resolved,
                    fuzzy,
                )
                resolved = fuzzy
                rows = _fetch_shipping_rate_rows(db, resolved, weight_g)
                for shipping_type, rate_usd in rows:
                    if rate_usd is None:
                        continue
                    rate = float(rate_usd)
                    if shipping_type == "EMS":
                        ems_rate = rate
                    elif shipping_type == "LP":
                        lp_rate = rate

        if ems_rate is None and lp_rate is None:
            return fallback

        result: dict[str, Any] = {
            "available": True,
            "country": resolved,
            "weight_g": weight_g,
            "EMS": None,
            "LP": None,
        }
        if ems_rate is not None:
            result["EMS"] = {
                "rate_usd": ems_rate,
                "label": _EMS_LABEL,
                "days": _EMS_DAYS,
            }
        if lp_rate is not None:
            result["LP"] = {
                "rate_usd": lp_rate,
                "label": _LP_LABEL,
                "days": _LP_DAYS,
            }
        return result
    except Exception:
        logger.exception(
            "get_shipping_options failed country=%r total_g=%s",
            country,
            total_g,
        )
        return fallback


def format_shipping_choice_message(options: dict, order_ref: str) -> str | None:
    """WhatsApp message prompting buyer to choose EMS or LP shipping."""
    try:
        if not options or not options.get("available"):
            country = (options or {}).get("country") or "your country"
            return (
                f"📦 Our team will confirm shipping cost for {country} shortly."
            )

        weight_g = options.get("weight_g", 0)
        ems = options.get("EMS")
        lp = options.get("LP")
        ref = order_ref or "your order"

        if ems and lp:
            return (
                f"📦 *Shipping for {ref}*\n"
                f"Weight: {weight_g}g\n\n"
                f"🚀 *{_EMS_LABEL}* ({_EMS_DAYS}) — ${ems['rate_usd']:.2f}\n"
                f"🐢 *{_LP_LABEL}* ({_LP_DAYS}) — ${lp['rate_usd']:.2f}\n\n"
                f"Tap a button below or reply *express* or *normal*"
            )

        if ems:
            return (
                f"📦 Only Express available at this weight.\n"
                f"🚀 *{_EMS_LABEL}* ({_EMS_DAYS}) — ${ems['rate_usd']:.2f}\n"
                f"Reply *express* to confirm."
            )

        if lp:
            return (
                f"📦 Only Normal shipping available at this weight.\n"
                f"🐢 *{_LP_LABEL}* ({_LP_DAYS}) — ${lp['rate_usd']:.2f}\n"
                f"Reply *normal* to confirm."
            )

        country = options.get("country") or "your country"
        return f"📦 Our team will confirm shipping cost for {country} shortly."
    except Exception:
        logger.exception("format_shipping_choice_message failed")
        return None


def format_cart_with_shipping(
    cart_items: list[dict],
    subtotal: float,
    shipping_option: dict | None,
    order_ref: str,
) -> str | None:
    """Format order review with line items, subtotal, shipping, and total."""
    try:
        lines = ["📋 *Order Review*", ""]

        for item in cart_items:
            name = (item.get("product_name") or item.get("sku") or "Item").strip()
            qty = item.get("qty", item.get("quantity", 0))
            try:
                qty_int = int(qty)
            except (TypeError, ValueError):
                qty_int = 0

            unit_price = item.get("unit_price")
            if unit_price is not None:
                line_total = float(unit_price) * qty_int
            else:
                line_total = float(item.get("line_total", 0))

            lines.append(f"• {name} × {qty_int} = ${line_total:.2f}")

        lines.append("─────────────────")

        sub = float(subtotal)
        lines.append(f"Subtotal  : ${sub:.2f}")

        shipping_cost = 0.0
        if shipping_option and shipping_option.get("rate_usd") is not None:
            shipping_cost = float(shipping_option["rate_usd"])
            label = shipping_option.get("label", "Shipping")
            days = shipping_option.get("days", "")
            day_suffix = f", {days}" if days else ""
            lines.append(
                f"Shipping  : ${shipping_cost:.2f} ({label}{day_suffix})"
            )
        else:
            lines.append("Shipping  : TBD")

        total = sub + shipping_cost
        lines.append("═════════════════")
        lines.append(f"Total     : *${total:.2f} USD*")

        if order_ref:
            lines.append("")
            lines.append(f"Ref: {order_ref}")

        return "\n".join(lines)
    except Exception:
        logger.exception("format_cart_with_shipping failed")
        return None
