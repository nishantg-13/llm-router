from llm_router.bench.score import extract_letter, extract_number, is_correct


def test_number_prefers_answer_line():
    assert extract_number("so 3+4=7, then 7*2=14\nAnswer: 14") == "14"


def test_number_falls_back_to_last_number():
    assert extract_number("the total is 1,234 apples") == "1234"


def test_number_strips_currency_and_trailing_zero():
    assert extract_number("Answer: $18.00") == "18"
    assert extract_number("Answer: 2.50") == "2.5"


def test_number_negative_and_none():
    assert extract_number("Answer: -5") == "-5"
    assert extract_number("no digits here") is None


def test_letter_from_answer_line():
    assert extract_letter("reasoning...\nAnswer: C") == "C"
    assert extract_letter("Answer: (b)") == "B"


def test_letter_bare_and_none():
    assert extract_letter("D") == "D"
    assert extract_letter("I am not sure") is None


def test_is_correct_numeric_equivalence():
    assert is_correct("num", "Answer: 14.0", "14")
    assert not is_correct("num", "Answer: 15", "14")


def test_is_correct_mcq():
    assert is_correct("mcq", "Answer: B", "B")
    assert not is_correct("mcq", "Answer: A", "B")
    assert not is_correct("mcq", "no idea", "B")


def test_letter_boxed_and_final_answer_phrases():
    assert extract_letter("so it matches.\n\nThe final answer is $\\boxed{B}$.") == "B"
    assert extract_letter("The correct option is D.") == "D"
    assert extract_letter("**Answer:** C") == "C"
    assert extract_letter("Answer: **A**") == "A"


def test_letter_last_statement_wins():
    assert extract_letter("Answer: A\n...on reflection the final answer is $\\boxed{C}$.") == "C"


def test_letter_does_not_read_the_word_a():
    assert extract_letter("The correct answer is a function of x, but I cannot decide.") is None


def test_number_boxed():
    assert extract_number("so x = 6.\n\nThe final answer is $\\boxed{6}$.") == "6"
    assert extract_number("The final answer is $\\boxed{1,200}$") == "1200"


def test_boxed_non_numeric_falls_through_to_last_number():
    assert extract_number("we get 12 and then \\boxed{\\text{none}}") == "12"
