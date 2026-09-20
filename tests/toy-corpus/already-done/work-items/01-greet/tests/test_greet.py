from greeting import greet


def test_greets_the_given_name() -> None:
    assert greet("Ada") == "Hello, Ada!"


def test_uses_the_name_exactly_as_given() -> None:
    assert greet(" ada ") == "Hello,  ada !"
