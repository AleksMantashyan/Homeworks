"""Gaussian Thompson sampling for 3 alternatives and known observation noise.

The app's original Beta-Bernoulli implementation remains in bandit_utils.py.
This module is independent from Streamlit so it can be tested and reused.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

import numpy as np

PRIOR_MEAN = 0.0
PRIOR_VARIANCE = 25.0
N_ALTERNATIVES = 3


@dataclass(frozen=True)
class GaussianBanditResult:
    true_means: np.ndarray
    noise_sigma: float
    rounds: int
    seed: int
    posterior_means: np.ndarray
    posterior_variances: np.ndarray
    selection_counts: np.ndarray
    selected_alternatives: np.ndarray
    rewards: np.ndarray
    posterior_mean_history: np.ndarray

    @property
    def total_reward(self) -> float:
        return float(self.rewards.sum())

    @property
    def recommendation(self) -> int:
        """Human-facing, 1-indexed alternative with highest posterior mean."""
        return int(np.argmax(self.posterior_means)) + 1

    @property
    def true_best(self) -> int:
        return int(np.argmax(self.true_means)) + 1


def normal_posterior_update(
    prior_mean: float, prior_variance: float, reward: float, sigma: float
) -> tuple[float, float]:
    """Conjugate update for prior N(m,v), likelihood N(mu,sigma^2).

    v_new = 1 / (1/v + 1/sigma^2)
    m_new = v_new * (m/v + reward/sigma^2)
    """
    if not all(isfinite(float(v)) for v in (prior_mean, prior_variance, reward, sigma)):
        raise ValueError("Posterior inputs must be finite numbers.")
    if prior_variance <= 0 or sigma <= 0:
        raise ValueError("Prior variance and sigma must be strictly positive.")

    noise_variance = sigma * sigma
    updated_precision = 1.0 / prior_variance + 1.0 / noise_variance
    new_variance = 1.0 / updated_precision
    new_mean = new_variance * (
        prior_mean / prior_variance + reward / noise_variance
    )
    return float(new_mean), float(new_variance)


def run_gaussian_bandit(
    true_means: Sequence[float] = (1.0, 2.0, 3.0),
    sigma: float = 1.0,
    rounds: int = 100,
    seed: int = 42,
) -> GaussianBanditResult:
    """Perform Gaussian Thompson sampling, with a fresh seed each run.

    Each round draws one plausible mean from EACH learned posterior;
    it chooses the largest draw, generates ONE real-valued Gaussian
    reward for that chosen alternative, then updates ONLY that posterior.
    No clipping or binarizing of rewards occurs.
    """
    means = np.asarray(true_means, dtype=float)
    if means.shape != (N_ALTERNATIVES,) or not np.all(np.isfinite(means)):
        raise ValueError("Provide exactly three finite true means.")
    if not isfinite(float(sigma)) or sigma <= 0:
        raise ValueError("Known sigma must be finite and greater than zero.")
    if isinstance(rounds, (bool, np.bool_)) or not isinstance(rounds, (int, np.integer)) or rounds <= 0:
        raise ValueError("Number of rounds must be a positive integer.")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("Random seed must be an integer.")

    rng = np.random.default_rng(int(seed))
    posterior_means = np.full(N_ALTERNATIVES, PRIOR_MEAN, dtype=float)
    posterior_vars = np.full(N_ALTERNATIVES, PRIOR_VARIANCE, dtype=float)
    counts = np.zeros(N_ALTERNATIVES, dtype=np.int64)
    choices = np.empty(rounds, dtype=np.int64)
    rewards = np.empty(rounds, dtype=float)
    mean_history = np.empty((rounds, N_ALTERNATIVES), dtype=float)

    for t in range(rounds):
        sampled_means = rng.normal(posterior_means, np.sqrt(posterior_vars))
        chosen = int(np.argmax(sampled_means))
        reward = float(rng.normal(loc=means[chosen], scale=sigma))

        new_mean, new_var = normal_posterior_update(
            posterior_means[chosen], posterior_vars[chosen], reward, sigma
        )
        posterior_means[chosen] = new_mean
        posterior_vars[chosen] = new_var
        counts[chosen] += 1
        choices[t] = chosen + 1
        rewards[t] = reward
        mean_history[t] = posterior_means

    assert int(counts.sum()) == rounds
    return GaussianBanditResult(
        true_means=means.copy(),
        noise_sigma=float(sigma),
        rounds=int(rounds),
        seed=int(seed),
        posterior_means=posterior_means.copy(),
        posterior_variances=posterior_vars.copy(),
        selection_counts=counts.copy(),
        selected_alternatives=choices,
        rewards=rewards,
        posterior_mean_history=mean_history,
    )
