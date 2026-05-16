"""Business operating hours — New Life Medicare (IST).

Primary human support: Monday–Saturday, 10:00 AM – 8:00 PM IST.
Sunday: limited operations (AI 24/7; human support reduced).
Public holidays: AI active; dispatch/processing may be delayed.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta

import pytz

DEFAULT_TIMEZONE = "Asia/Kolkata"
DEFAULT_START_HOUR = 10
DEFAULT_END_HOUR = 20

# Fixed-date Indian public holidays (month, day). Variable dates via BUSINESS_HOLIDAY_DATES.
FIXED_PUBLIC_HOLIDAYS: tuple[tuple[int, int], ...] = (
    (1, 26),   # Republic Day
    (8, 15),   # Independence Day
    (12, 25),  # Christmas
)

HOLIDAY_NAMES: dict[tuple[int, int], str] = {
    (1, 26): "Republic Day",
    (8, 15): "Independence Day",
    (12, 25): "Christmas",
}


def _read_config() -> tuple[pytz.BaseTzInfo, int, int]:
    tz_name = os.getenv("BUSINESS_TIMEZONE", DEFAULT_TIMEZONE)
    start_hour = int(os.getenv("BUSINESS_HOURS_START", str(DEFAULT_START_HOUR)))
    end_hour = int(os.getenv("BUSINESS_HOURS_END", str(DEFAULT_END_HOUR)))
    return pytz.timezone(tz_name), start_hour, end_hour


def _now_in_tz(tz: pytz.BaseTzInfo) -> datetime:
    """Indirection so tests can monkeypatch the wall clock."""
    return datetime.now(tz)


def _parse_extra_holiday_dates() -> set[date]:
    """ISO dates from BUSINESS_HOLIDAY_DATES (e.g. 2026-03-14 for Holi)."""
    raw = os.getenv("BUSINESS_HOLIDAY_DATES", "")
    dates: set[date] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            dates.add(date.fromisoformat(part))
        except ValueError:
            continue
    return dates


def is_fixed_public_holiday(d: date) -> bool:
    return (d.month, d.day) in FIXED_PUBLIC_HOLIDAYS


def is_public_holiday(when: datetime | None = None) -> bool:
    tz, _, _ = _read_config()
    now = when or _now_in_tz(tz)
    local_date = now.date()
    if is_fixed_public_holiday(local_date):
        return True
    return local_date in _parse_extra_holiday_dates()


def get_public_holiday_name(when: datetime | None = None) -> str | None:
    tz, _, _ = _read_config()
    now = when or _now_in_tz(tz)
    local_date = now.date()
    name = HOLIDAY_NAMES.get((local_date.month, local_date.day))
    if name:
        return name
    if local_date in _parse_extra_holiday_dates():
        return "public holiday"
    return None


def is_sunday(when: datetime | None = None) -> bool:
    tz, _, _ = _read_config()
    now = when or _now_in_tz(tz)
    return now.weekday() == 6


def is_full_operations_weekday(weekday: int) -> bool:
    """Monday=0 … Saturday=5."""
    return weekday <= 5


def is_business_hours(when: datetime | None = None) -> bool:
    """True during Mon–Sat primary hours (10 AM–8 PM IST), excluding public holidays."""
    tz, start_hour, end_hour = _read_config()
    now = when or _now_in_tz(tz)
    if not is_full_operations_weekday(now.weekday()):
        return False
    if is_public_holiday(now):
        return False
    return start_hour <= now.hour < end_hour


def is_limited_operations(when: datetime | None = None) -> bool:
    """Sunday or public holiday — AI on; human support limited."""
    tz, _, _ = _read_config()
    now = when or _now_in_tz(tz)
    return is_sunday(now) or is_public_holiday(now)


def get_operations_mode(when: datetime | None = None) -> str:
    """One of: full | limited | closed."""
    if is_business_hours(when):
        return "full"
    if is_limited_operations(when):
        return "limited"
    return "closed"


def _format_hour_12h(hour: int) -> str:
    suffix = "AM" if hour < 12 else "PM"
    display = hour % 12 or 12
    return f"{display}:00 {suffix}"


def get_next_business_open_str(when: datetime | None = None) -> str:
    """When full human support resumes (Mon–Sat from BUSINESS_HOURS_START)."""
    tz, start_hour, end_hour = _read_config()
    now = when or _now_in_tz(tz)
    tz_abbrev = now.strftime("%Z") or "IST"
    formatted_start = _format_hour_12h(start_hour)

    def _next_open_day(from_dt: datetime) -> datetime:
        candidate = from_dt
        for _ in range(14):
            if is_full_operations_weekday(candidate.weekday()):
                check = candidate.replace(hour=start_hour, minute=0, second=0, microsecond=0)
                if not is_public_holiday(check):
                    return check
            candidate = (candidate + timedelta(days=1)).replace(
                hour=start_hour, minute=0, second=0, microsecond=0
            )
        return from_dt + timedelta(days=1)

    weekday = now.weekday()
    hour = now.hour

    if is_business_hours(now):
        return f"today at {formatted_start} {tz_abbrev}"

    if weekday <= 5 and hour < start_hour and not is_public_holiday(now):
        return f"today at {formatted_start} {tz_abbrev}"

    if weekday <= 4 and hour >= end_hour and not is_public_holiday(now):
        tomorrow = now + timedelta(days=1)
        if tomorrow.weekday() == 6 or is_public_holiday(tomorrow):
            nxt = _next_open_day(tomorrow)
            day_name = nxt.strftime("%A")
            return f"{day_name} {formatted_start} {tz_abbrev}"
        return f"tomorrow {formatted_start} {tz_abbrev}"

    if weekday == 5 and hour >= end_hour:
        return f"Monday {formatted_start} {tz_abbrev}"

    if is_sunday(now) or is_public_holiday(now):
        nxt = _next_open_day(now + timedelta(days=1))
        if nxt.date() == now.date():
            return f"today at {formatted_start} {tz_abbrev}"
        if (nxt.date() - now.date()).days == 1:
            return f"tomorrow {formatted_start} {tz_abbrev}"
        return f"{nxt.strftime('%A')} {formatted_start} {tz_abbrev}"

    nxt = _next_open_day(now)
    if nxt.date() == now.date():
        return f"today at {formatted_start} {tz_abbrev}"
    return f"{nxt.strftime('%A')} {formatted_start} {tz_abbrev}"


def get_off_hours_notice(when: datetime | None = None) -> str:
    """WhatsApp-friendly notice for outside primary human support hours."""
    tz, start_hour, end_hour = _read_config()
    now = when or _now_in_tz(tz)
    tz_abbrev = now.strftime("%Z") or "IST"
    start_s = _format_hour_12h(start_hour)
    end_s = _format_hour_12h(end_hour)
    resume = get_next_business_open_str(now)

    if is_public_holiday(now):
        holiday = get_public_holiday_name(now) or "a public holiday"
        return (
            f"Thank you for reaching out to New Life Medicare!\n\n"
            f"Today is {holiday} in India. Our AI assistant is available 24/7. "
            f"Human support and dispatch may be limited; order processing can be delayed. "
            f"Full operations resume {resume}.\n\n"
            f"For urgent inquiries: exports@newlifemedicare.com"
        )

    if is_sunday(now):
        return (
            f"Thank you for reaching out to New Life Medicare!\n\n"
            f"It's Sunday — limited operations today. Our AI assistant is active 24/7. "
            f"Priority human support resumes {resume} "
            f"(Mon–Sat, {start_s} – {end_s} {tz_abbrev}).\n\n"
            f"For urgent inquiries: exports@newlifemedicare.com"
        )

    return (
        f"Thank you for reaching out to New Life Medicare!\n\n"
        f"Our team is currently offline. Business hours: Mon–Sat, {start_s} – {end_s} {tz_abbrev}. "
        f"Our AI assistant is available 24/7. Priority human support resumes {resume}.\n\n"
        f"For urgent inquiries: exports@newlifemedicare.com"
    )
