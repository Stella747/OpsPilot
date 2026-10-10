
from main import has_sufficient_historical_evidence


def test_relevant_match_is_accepted():
    assert has_sufficient_historical_evidence(0.65, 0.40) is True


def test_weak_match_is_rejected():
    assert has_sufficient_historical_evidence(0.25, 0.40) is False


def test_match_exactly_at_threshold_is_accepted():
    assert has_sufficient_historical_evidence(0.40, 0.40) is True


def test_match_just_below_threshold_is_rejected():
    assert has_sufficient_historical_evidence(0.3999, 0.40) is False
