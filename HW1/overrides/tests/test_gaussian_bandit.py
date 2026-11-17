"""Fast numerical and deterministic checks, no Streamlit runtime required."""
import numpy as np
import pytest

from gaussian_bandit import normal_posterior_update, run_gaussian_bandit


def test_first_posterior_update_known_sigma():
    # Prior N(0,25), sigma=1, observation=3: posterior mean=75/26.
    mean, variance = normal_posterior_update(0.0, 25.0, 3.0, 1.0)
    assert variance == pytest.approx(25.0 / 26.0)
    assert mean == pytest.approx(75.0 / 26.0)


def test_selected_arm_only_updated_on_one_round():
    r = run_gaussian_bandit(rounds=1, seed=42)
    assert r.selection_counts.sum() == 1
    untouched = np.flatnonzero(r.selection_counts == 0)
    assert np.all(r.posterior_means[untouched] == 0.0)
    assert np.all(r.posterior_variances[untouched] == 25.0)


def test_reproducible_with_identical_seed():
    a = run_gaussian_bandit((1, 2, 3), sigma=3, rounds=1000, seed=42)
    b = run_gaussian_bandit((1, 2, 3), sigma=3, rounds=1000, seed=42)
    np.testing.assert_array_equal(a.selected_alternatives, b.selected_alternatives)
    np.testing.assert_array_equal(a.rewards, b.rewards)
    np.testing.assert_array_equal(a.posterior_means, b.posterior_means)


@pytest.mark.parametrize("sigma,rounds", [(1.0, 100), (1.0, 1000), (3.0, 1000)])
def test_selections_sum_to_rounds(sigma, rounds):
    r = run_gaussian_bandit((1, 2, 3), sigma=sigma, rounds=rounds, seed=42)
    assert sum(r.selection_counts) == rounds
    assert len(r.rewards) == rounds
    assert r.posterior_mean_history.shape == (rounds, 3)
    assert r.recommendation == int(np.argmax(r.posterior_means)) + 1


def test_negative_rewards_are_not_clipped():
    r = run_gaussian_bandit((-20, -20, -20), sigma=1.0, rounds=30, seed=2)
    assert np.any(r.rewards < 0.0)
    assert r.total_reward < 0.0


@pytest.mark.parametrize("sigma", [0.0, -1.0, float("nan"), float("inf")])
def test_sigma_must_be_positive_finite(sigma):
    with pytest.raises(ValueError):
        run_gaussian_bandit(sigma=sigma)


@pytest.mark.parametrize("rounds", [0, -5, 2.5, True])
def test_round_count_must_be_positive_integer(rounds):
    with pytest.raises(ValueError):
        run_gaussian_bandit(rounds=rounds)


def test_three_alternatives_required():
    with pytest.raises(ValueError):
        run_gaussian_bandit(true_means=(1, 2))
