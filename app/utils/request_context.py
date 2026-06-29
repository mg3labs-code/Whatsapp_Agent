import contextvars
import logging

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


def set_request_id(phone_hash: str, message_id: str) -> str:
    rid = f"{phone_hash}:{(message_id or '')[:8]}"
    request_id_var.set(rid)
    return rid


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True
