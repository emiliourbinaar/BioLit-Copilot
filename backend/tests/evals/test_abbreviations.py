from biolit_evals.abbreviations import find_abbreviations


def test_parenthetical_that_is_not_a_definition_is_rejected():
    # Without this the measurement would count arbitrary parentheticals as expansions and
    # overstate how much of the gap a deterministic mechanism reaches.
    assert find_abbreviations("The patient (aged 55) recovered.") == {}


def test_finds_a_parenthetical_definition():
    # The real BC5CDR shape: the abstract defines its abbreviation on first use.
    text = "Patients developed torsades de pointes (TdP) after the second dose."
    assert find_abbreviations(text)["TdP"] == "torsades de pointes"
