"""Conversation UI presentation helpers."""

from app.messages.conversation_ui import (
    apply_menu_selection_ack,
    is_main_menu_request,
    mark_menu_selection,
)


def test_is_main_menu_request():
    assert is_main_menu_request("main_menu") is True
    assert is_main_menu_request("menu") is True
    assert is_main_menu_request("Main Menu") is True
    assert is_main_menu_request("  show   menu  ") is True
    assert is_main_menu_request("options") is True
    assert is_main_menu_request("order") is False
    assert is_main_menu_request("menu please") is False


def test_mark_and_apply_menu_selection_ack():
    session = mark_menu_selection({}, "order")
    reply, session = apply_menu_selection_ack(
        "📋 *Send each product like this (quantity = number of strips):*\n\nExample",
        session,
    )
    assert "Place an Order" in reply
    assert "strips" in reply.lower()
    assert "pending_menu_ack" not in session


def test_speak_menu_ack_does_not_double_connecting_copy():
    session = mark_menu_selection({}, "speak")
    escalate_body = (
        "I'm connecting you with our export team right now!\n\n"
        "Our team will follow up with you soon."
    )
    reply, session = apply_menu_selection_ack(escalate_body, session)
    assert reply.startswith("You selected *Speak to Team*")
    assert reply.count("connecting you") == 1
    assert "sales team" not in reply.lower()
    assert "pending_menu_ack" not in session
