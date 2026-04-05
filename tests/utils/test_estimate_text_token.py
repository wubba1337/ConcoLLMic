from app.utils.utils import estimate_text_token


def test_estimate_text_token_basic():
    assert estimate_text_token("hello world") > 0


def test_estimate_text_token_none_text():
    assert estimate_text_token(None) == 0
