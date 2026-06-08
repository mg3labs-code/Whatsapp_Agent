"""Assign an India Post AWB to an order (simulates ops team after dispatch).

Usage (PowerShell):
  $env:DATABASE_URL="postgresql://..."
  python scripts/set_order_awb.py ORD-20260608-6727 EB126023474IN

Optional — mark as shipped:
  python scripts/set_order_awb.py ORD-20260608-6727 EB126023474IN --shipped
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from app.db.database import SessionLocal
from app.db.models import Order


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach AWB to an order")
    parser.add_argument("order_ref", help="Order ref, e.g. ORD-20260608-6727")
    parser.add_argument("awb", help="India Post AWB, e.g. EB126023474IN")
    parser.add_argument(
        "--shipped",
        action="store_true",
        help="Also set order status to shipped",
    )
    args = parser.parse_args()

    if not os.getenv("DATABASE_URL"):
        print("Missing DATABASE_URL")
        sys.exit(1)

    base_ref = args.order_ref.strip().upper().split("-L")[0]
    awb = args.awb.strip().upper()

    db = SessionLocal()
    try:
        order = (
            db.query(Order)
            .filter(Order.order_ref.like(f"{base_ref}%"))
            .order_by(Order.created_at.desc())
            .first()
        )
        if not order:
            print(f"No order found matching {base_ref}")
            sys.exit(1)

        order.tracking_number = awb
        if args.shipped:
            order.status = "shipped"
            order.payment_status = "shipped"
        db.commit()
        print(f"Updated {order.order_ref}")
        print(f"  phone: {order.phone}")
        print(f"  AWB:   {awb}")
        print(f"  status: {order.payment_status or order.status}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
