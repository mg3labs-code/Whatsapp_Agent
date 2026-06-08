"""POST a sample India Post tracking event to your webhook (UAT / sandbox).

Use when India Post sandbox does not push live events but you need to demo
proactive WhatsApp shipment updates.

Usage (PowerShell):
  $env:BASE_URL="https://pay-newlife-medex.mg3verse.com"
  python scripts/simulate_indiapost_webhook.py EB126023474IN --event "Item dispatched" --code IB

  python scripts/simulate_indiapost_webhook.py EB126023474IN --event "Item Delivered" --code ID
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate India Post webhook event")
    parser.add_argument("awb", help="Article number / AWB")
    parser.add_argument("--event", default="Item booked", help="Event description")
    parser.add_argument("--code", default="IB", help="Event code (IB, ID, etc.)")
    parser.add_argument(
        "--office",
        default="Vijayanagar S.O",
        help="Event office name",
    )
    args = parser.parse_args()

    base = (os.getenv("BASE_URL") or "").strip().rstrip("/")
    if not base:
        print("Missing BASE_URL")
        sys.exit(1)

    payload = {
        "article_number": args.awb.strip().upper(),
        "event_code": args.code,
        "event_description": args.event,
        "event_office_name": args.office,
        "destination_city": "Hyderabad",
    }
    url = f"{base}/webhook/indiapost"
    print(f"POST {url}")
    print(f"Payload: {payload}")

    response = httpx.post(url, json=payload, timeout=30.0)
    print(f"HTTP {response.status_code}")
    if response.text:
        print(response.text[:500])


if __name__ == "__main__":
    main()
