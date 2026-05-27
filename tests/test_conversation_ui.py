"""Conversation UI presentation helpers."""

from app.messages.conversation_ui import (
    apply_menu_selection_ack,
    is_main_menu_request,
    mark_menu_selection,
)


def test_is_main_menu_request():
    assert is_main_menu_request("main_menu") is True
    assert is_main_menu_request("order") is False


def test_mark_and_apply_menu_selection_ack():
    session = mark_menu_selection({}, "order")
    reply, session = apply_menu_selection_ack(
        "Which product would you like to add? (name or SKU)",
        session,
    )
    assert "Place an Order" in reply
    assert "Which product" in reply
    assert "pending_menu_ack" not in session
