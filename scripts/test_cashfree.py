"""Manual Cashfree sandbox test — creates a payment link and prints the URL.

Usage:
  set CASHFREE_APP_ID=...
  set CASHFREE_SECRET_KEY=...
  set CASHFREE_ENV=sandbox
  set BASE_URL=https://your-ngrok-or-railway-url
  python scripts/test_cashfree.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from app.integrations.cashfree import create_payment_link


async def main() -> None:
    for key in ("CASHFREE_APP_ID", "CASHFREE_SECRET_KEY"):
        if not os.getenv(key):
            print(f"Missing env var: {key}")
            sys.exit(1)

    order_ref = f"ORD-TEST-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    amount = float(os.getenv("CASHFREE_TEST_AMOUNT", "115"))
    phone = os.getenv("CASHFREE_TEST_PHONE", "9999999999")

    print(f"Creating payment link for {order_ref} amount={amount} phone={phone}")
    result = await create_payment_link(order_ref, amount, phone, "Test Buyer")
    link_url = result.get("link_url")

    if not link_url:
        print("Failed to create payment link. Check Railway/ngrok logs and Cashfree credentials.")
        sys.exit(1)

    print("\nPayment link created:")
    print(link_url)
    print("\nNext steps:")
    print("1. Open the link in a browser")
    print("2. Pay with sandbox test card: 4111 1111 1111 1111 | 12/30 | CVV 123")
    print("3. Watch backend logs for: Cashfree event received / Payment confirmed")
    print("4. Buyer WhatsApp should get payment success message if webhook + phone match")


if __name__ == "__main__":
    asyncio.run(main())
