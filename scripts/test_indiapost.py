"""Smoke-test India Post sandbox login + tracking (httpx only, no app deps).

Usage (PowerShell):
  $env:INDIAPOST_USERNAME="9999974410"
  $env:INDIAPOST_PASSWORD="Dop@1234"
  $env:INDIAPOST_ENV="sandbox"
  python scripts/test_indiapost.py EB126023474IN

Usage (cmd.exe):
  set INDIAPOST_USERNAME=9999974410
  set INDIAPOST_PASSWORD=Dop@1234
  set INDIAPOST_ENV=sandbox
  python scripts/test_indiapost.py EB126023474IN
"""

from __future__ import annotations

import asyncio
import os
import sys

import httpx

REQUEST_TIMEOUT_SECONDS = 15.0


def _api_base() -> str:
    explicit = os.getenv("INDIAPOST_API_BASE", "").strip().rstrip("/")
    if explicit:
        return explicit
    env = os.getenv("INDIAPOST_ENV", "sandbox").strip().lower()
    if env == "production":
        return "https://cept.gov.in/beextcustomer"
    return "https://test.cept.gov.in/beextcustomer"


async def _login(client: httpx.AsyncClient) -> str:
    username = os.getenv("INDIAPOST_USERNAME", "").strip()
    password = os.getenv("INDIAPOST_PASSWORD", "").strip()
    url = f"{_api_base()}/v1/access/login"
    response = await client.post(
        url,
        json={"username": username, "password": password},
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    print(f"Login HTTP: {response.status_code}")
    if response.status_code >= 400:
        print(response.text[:400])
        return ""
    payload = response.json()
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    return str(data.get("access_token") or "")


def _format_row(awb: str, row: dict) -> str:
    booking = row.get("booking_details") if isinstance(row.get("booking_details"), dict) else {}
    events = row.get("tracking_details") if isinstance(row.get("tracking_details"), list) else []
    article = str(booking.get("article_number") or row.get("trackingNumber") or awb).upper()
    del_status = row.get("del_status")
    if isinstance(del_status, dict):
        status = str(del_status.get("del_status") or "in_transit")
    elif events:
        status = str(events[-1].get("event") or "in_transit")
    else:
        status = str(row.get("currentStatus") or row.get("current_status") or "in_transit")
    lines = [f"AWB: {article}", f"Status: {status}"]
    if events:
        latest = events[-1]
        lines.append(
            f"Latest: {latest.get('event') or '?'} @ {latest.get('office') or '?'}"
        )
    return "\n".join(lines)


async def main() -> None:
    for key in ("INDIAPOST_USERNAME", "INDIAPOST_PASSWORD"):
        if not os.getenv(key):
            print(f"Missing env var: {key}")
            print("PowerShell: $env:INDIAPOST_USERNAME=\"your_id\"")
            sys.exit(1)

    awb = (sys.argv[1] if len(sys.argv) > 1 else "EB126023474IN").upper()
    base = _api_base()
    print(f"API base: {base}")
    print(f"AWB: {awb}")

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        token = await _login(client)
        if not token:
            print("Login failed")
            sys.exit(1)
        print("Login OK")

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

        bulk = await client.post(
            f"{base}/v1/tracking/bulk",
            headers=headers,
            json={"bulk": [awb]},
        )
        print(f"Bulk HTTP: {bulk.status_code}")
        if bulk.status_code < 400:
            payload = bulk.json()
            print("Bulk success:", payload.get("success"))
            rows = payload.get("data") if isinstance(payload.get("data"), list) else []
            if rows and isinstance(rows[0], dict):
                print("\n--- Tracking (bulk) ---")
                print(_format_row(awb, rows[0]))
                return
            print("Bulk body:", str(payload)[:600])
        else:
            print(bulk.text[:400])

        single = await client.get(f"{base}/v1/tracking/{awb}", headers=headers)
        print(f"Single HTTP: {single.status_code}")
        if single.status_code < 400:
            payload = single.json()
            data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
            if isinstance(data, dict):
                print("\n--- Tracking (single) ---")
                print(_format_row(awb, data))
                return
            print("Single body:", str(payload)[:600])
        else:
            print(single.text[:400])

    print("\nNo tracking data returned for this AWB in sandbox.")


if __name__ == "__main__":
    asyncio.run(main())
