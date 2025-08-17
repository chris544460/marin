import math

import pytest

from marin.utils.unimax_weights import unimax_weights


def assert_invariants(weights, alloc, budget):
    assert math.isclose(sum(weights.values()), 1.0, rel_tol=1e-9)
    assert math.isclose(sum(alloc.values()), budget, rel_tol=1e-9)


def test_equal_sizes_split_evenly():
    sizes = {"en": 100, "fr": 100}
    weights, alloc = unimax_weights(sizes, steps=1, tokens_per_step=100, max_epochs=1.0)
    assert weights == pytest.approx({"en": 0.5, "fr": 0.5})
    assert alloc == pytest.approx({"en": 50.0, "fr": 50.0})
    assert_invariants(weights, alloc, 100.0)


def test_small_corpus_saturates_and_remainder_evenly_split():
    sizes = {"big1": 1000, "big2": 1000, "small": 100}
    weights, alloc = unimax_weights(sizes, steps=9, tokens_per_step=100, max_epochs=1.0)
    expected_alloc = {"big1": 400.0, "big2": 400.0, "small": 100.0}
    expected_weights = {k: v / 900.0 for k, v in expected_alloc.items()}
    assert alloc == pytest.approx(expected_alloc)
    assert weights == pytest.approx(expected_weights)
    assert_invariants(weights, alloc, 900.0)


def test_budget_exceeds_epoch_cap_auto_bumps():
    sizes = {"a": 100, "b": 100}
    weights, alloc = unimax_weights(sizes, steps=4, tokens_per_step=100, max_epochs=1.0)
    assert alloc == pytest.approx({"a": 200.0, "b": 200.0})
    assert weights == pytest.approx({"a": 0.5, "b": 0.5})
    assert_invariants(weights, alloc, 400.0)


def test_zero_size_corpus_gets_zero():
    sizes = {"a": 100, "zero": 0}
    weights, alloc = unimax_weights(sizes, steps=1, tokens_per_step=100, max_epochs=1.0)
    assert alloc["zero"] == 0.0
    assert weights["zero"] == 0.0
    assert_invariants(weights, alloc, 100.0)
