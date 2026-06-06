"""Smoke-test India Post sandbox login + tracking.

Usage:
  set INDIAPOST_USERNAME=...
  set INDIAPOST_PASSWORD=...
  set INDIAPOST_ENV=sandbox
  python scripts/test_indiapost.py EB126023474IN
"""

from __future__ import annotations

import asyncio
import os
import sys

from app.integrations.indiapost import get_access_token, lookup_tracking_message, track_bulk


async def main() -> None:
    for key in ("INDIAPOST_USERNAME", "INDIAPOST_PASSWORD"):
        if not os.getenv(key):
            print(f"Missing env var: {key}")
            sys.exit(1)

    token = await get_access_token()
    if not token:
        print("Login failed")
        sys.exit(1)
    print("Login OK")

    awb = (sys.argv[1] if len(sys.argv) > 1 else "EB126023474IN").upper()
    bulk = await track_bulk([awb])
    print("Bulk response success:", bulk.get("success"))

    msg = await lookup_tracking_message(awb)
    print(msg or "No tracking message returned")


if __name__ == "__main__":
    asyncio.run(main())
