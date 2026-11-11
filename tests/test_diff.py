from src.diff import changes, redline


def test_redline_marks_insert_and_delete():
    assert redline("rates stay high today", "rates stay today soon") == "rates stay [-high-] today {+soon+}"


def test_redline_replace():
    assert redline("inflation is high", "inflation is low") == "inflation is [-high-] {+low+}"


def test_identical_texts_no_changes():
    assert changes("same words here", "same words here") == []
    assert redline("same words here", "same words here") == "same words here"


def test_first_statement_all_insert():
    assert redline("", "brand new text") == "{+brand new text+}"
    assert changes("", "brand new text") == [{"op": "insert", "old": "", "new": "brand new text"}]
