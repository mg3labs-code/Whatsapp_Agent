from app.messages.session_flow import (
    clear_human_handoff,
    is_discount_request,
    is_order_reset_request,
    is_speak_to_team_request,
    resolve_business_type_button,
    should_resume_from_human_handoff,
)


def test_should_resume_from_human_handoff_greetings():
    assert should_resume_from_human_handoff("hello") is True
    assert should_resume_from_human_handoff("hi") is True
    assert should_resume_from_human_handoff("main_menu") is True
    assert should_resume_from_human_handoff("faqs") is True


def test_should_not_resume_on_speak():
    assert should_resume_from_human_handoff("speak") is False


def test_clear_human_handoff():
    session = clear_human_handoff({"human_active": True, "escalation_reason": "hot_lead"})
    assert "human_active" not in session
    assert "escalation_reason" not in session


def test_is_order_reset_request_phrases():
    assert is_order_reset_request("I need new order") is True
    assert is_order_reset_request("not needed cancel") is True
    assert is_order_reset_request("350") is False


def test_discount_and_speak_intents():
    assert is_discount_request("any discount available?") is True
    assert is_speak_to_team_request("I want to speak to team") is True


def test_resolve_business_type_button():
    assert resolve_business_type_button("biz_pharmacy") == "pharmacy clinic"


def test_main_menu_includes_my_orders():
    from app.messages.conversation_ui import MAIN_MENU_LIST_ROWS, MY_ORDERS_ID

    ids = {row["id"] for row in MAIN_MENU_LIST_ROWS}
    assert MY_ORDERS_ID in ids
    assert "order" in ids
    assert "pricing" in ids
    assert "faq" in ids
    assert "speak" in ids
