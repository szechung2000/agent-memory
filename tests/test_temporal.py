"""Tests for temporal penalty heuristics."""

from agent_memory.core.temporal import query_anchor_year, temporal_penalty

Q_PRE = "Who owned Bear Stearns before the financial crisis?"
Q_POST = "Who owns Bear Stearns after the financial crisis?"
C_ACQ = "In March 2008, JPMorgan Chase acquired Bear Stearns in a fire sale"
C_IND = "Bear Stearns was an independent investment bank in New York"


def test_anchor_from_explicit_year():
    assert query_anchor_year("Where was Jamie Dimon CEO in 2002?") == 2002


def test_anchor_from_event_phrase():
    assert query_anchor_year(Q_PRE) == 2008


def test_no_anchor():
    assert query_anchor_year("What is self-attention?") is None


def test_penalizes_later_state_change_for_before_query():
    assert temporal_penalty(Q_PRE, C_ACQ) == 0.5


def test_no_penalty_for_after_query():
    assert temporal_penalty(Q_POST, C_ACQ) == 0.0


def test_no_penalty_for_non_event_fact():
    assert temporal_penalty(Q_PRE, C_IND) == 0.0


def test_penalizes_future_event_for_year_query():
    q = "Where was Jamie Dimon CEO in 2002?"
    c = (
        "In 2004 JPMorgan Chase merged with Bank One, and Jamie Dimon "
        "became CEO of the combined company in 2005"
    )
    assert temporal_penalty(q, c) == 0.5
