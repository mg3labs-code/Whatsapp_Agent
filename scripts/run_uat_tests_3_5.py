"""Run India Post UAT tests 3–5 (AWB assign → order status → webhook).

Usage (PowerShell):
  $env:DATABASE_URL="postgresql://..."          # Railway public Postgres URL
  $env:BASE_URL="https://pay-newlife-medex.mg3verse.com"
  $env:INDIAPOST_USERNAME="9999974410"
  $env:INDIAPOST_PASSWORD="Dop@1234"
  $env:INDIAPOST_ENV="sandbox"
  python scripts/run_uat_tests_3_5.py ORD-20260608-6727 EB126023474IN
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from app.agents.order import _append_indiapost_tracking, _status_message
from app.db.database import SessionLocal
from app.db.models import Order
from app.integrations.indiapost import lookup_tracking_message


def test_3_assign_awb(order_ref: str, awb: str) -> Order:
    print("\n=== Test 3: Assign AWB to order (ops team) ===")
    if not os.getenv("DATABASE_URL"):
        raise RuntimeError("DATABASE_URL required for Test 3")

    base_ref = order_ref.strip().upper().split("-L")[0]
    db = SessionLocal()
    try:
        order = (
            db.query(Order)
            .filter(Order.order_ref.like(f"{base_ref}%"))
            .order_by(Order.created_at.desc())
            .first()
        )
        if not order:
            raise RuntimeError(f"No order found for {base_ref}")

        order.tracking_number = awb.upper()
        order.status = "shipped"
        order.payment_status = "shipped"
        db.commit()
        db.refresh(order)
        print(f"OK: {order.order_ref} -> AWB {awb.upper()} (phone {order.phone})")
        return order
    finally:
        db.close()


async def test_4_order_status(order: Order, awb: str) -> None:
    print("\n=== Test 4: Order Status + live India Post tracking ===")
    ref = (order.order_ref or "ORD-UNKNOWN").split("-L")[0]
    display_status = (order.payment_status or order.status or "processing").strip().lower()
    base_message = (
        f"📦 *Order {ref}*\n"
        f"Status: {display_status.replace('_', ' ')}\n"
        f"{_status_message(display_status)}"
    )
    reply = await _append_indiapost_tracking(
        base_message,
        tracking_number=awb.upper(),
        order_ref=ref,
    )
    print("WhatsApp reply preview:")
    print("-" * 40)
    print(reply)
    print("-" * 40)
    if awb.upper() not in reply:
        raise RuntimeError("AWB missing from order status reply")
    if "temporarily unavailable" in reply.lower():
        raise RuntimeError("India Post tracking unavailable — check INDIAPOST_* env on Railway")


async def test_5_webhook(base_url: str, awb: str) -> None:
    print("\n=== Test 5: Simulated India Post webhook (dispatched) ===")
    url = f"{base_url.rstrip('/')}/webhook/indiapost"
    payload = {
        "article_number": awb.upper(),
        "event_code": "IB",
        "event_description": "Item dispatched",
        "event_office_name": "Vijayanagar S.O",
        "destination_city": "Hyderabad",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload)
    print(f"POST {url} -> HTTP {response.status_code}")
    if response.status_code != 200:
        raise RuntimeError(f"Webhook failed: {response.text[:300]}")
    print("OK: webhook accepted (check WhatsApp for proactive update if AWB linked to order)")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("order_ref", nargs="?", default="ORD-20260608-6727")
    parser.add_argument("awb", nargs="?", default="EB126023474IN")
    args = parser.parse_args()

    base_url = (os.getenv("BASE_URL") or "https://pay-newlife-medex.mg3verse.com").strip()

    order = test_3_assign_awb(args.order_ref, args.awb)
    await test_4_order_status(order, args.awb)
    await test_5_webhook(base_url, args.awb)

    print("\n=== UAT tests 3–5 complete ===")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"\nFAILED: {exc}")
        sys.exit(1)
