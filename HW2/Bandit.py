"""HW2: Epsilon-Greedy and Gaussian Thompson Sampling for four advertisements.

The assignment provides expected rewards [1, 2, 3, 4] but no reward distribution.
We explicitly model each arm as Normal(mean=its expected reward, precision=tau).
Here, *precision* means inverse observation variance, not standard deviation.

Run ``python Bandit.py`` for the required reproducible 20,000-trial experiment.
"""

from __future__ import annotations

import argparse
import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")  # Save figures without needing a desktop display.
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


BANDIT_REWARD = (1.0, 2.0, 3.0, 4.0)
NUMBER_OF_TRIALS = 20_000
DEFAULT_SEED = 2026


class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass


# --------------------------------------#


@dataclass(frozen=True)
class ExperimentResult:
    """History of a single bandit algorithm's experiment.

    Attributes:
        algorithm: Algorithm name used in the CSV and figures.
        arms: Zero-based chosen arm indices, one per trial.
        rewards: Observed (possibly noisy) rewards.
        regrets: Expected, or pseudo-, regret per trial: max(mean) - chosen mean.
        estimates: Estimated mean of each arm after each trial.
        true_means: True expected rewards of the arms in the simulation.
    """

    algorithm: str
    arms: np.ndarray
    rewards: np.ndarray
    regrets: np.ndarray
    estimates: np.ndarray
    true_means: np.ndarray

    @property
    def cumulative_rewards(self) -> np.ndarray:
        """Return total observed reward up to each trial."""
        return np.cumsum(self.rewards)

    @property
    def cumulative_regrets(self) -> np.ndarray:
        """Return nonnegative cumulative pseudo-regret up to each trial."""
        return np.cumsum(self.regrets)

    @property
    def optimal_arm_fraction(self) -> float:
        """Return the fraction of trials selecting a highest-mean arm."""
        return float(np.mean(self.true_means[self.arms] == np.max(self.true_means)))


def _initialize_bandit(
    obj: Bandit, p: tuple[float, ...] | list[float] | np.ndarray,
    *, observation_precision: float, seed: int,
) -> None:
    """Set up the shared Gaussian environment and independent random generators."""
    means = np.asarray(p, dtype=float)
    if means.ndim != 1 or means.size == 0 or not np.all(np.isfinite(means)):
        raise ValueError("p must be a nonempty, finite, one-dimensional set of means")
    if not np.isfinite(observation_precision) or observation_precision <= 0:
        raise ValueError("observation_precision must be strictly positive and finite")

    obj.p = means.copy()  # Preserve the original assignment's constructor naming.
    obj.means = means.copy()
    obj.n_arms = len(means)
    obj.observation_precision = float(observation_precision)
    obj.counts = np.zeros(obj.n_arms, dtype=int)
    obj.reward_sums = np.zeros(obj.n_arms, dtype=float)
    obj.estimates = np.zeros(obj.n_arms, dtype=float)
    obj.action_rng = np.random.default_rng(seed)
    obj.reward_rng = np.random.default_rng(seed + 10_000)
    obj._reward_table: np.ndarray | None = None
    obj.result: ExperimentResult | None = None


def _observed_pull(obj: Bandit, arm: int) -> tuple[int, float]:
    """Observe reward from arm, using a shared outcome table if supplied."""
    if obj._reward_table is None:
        reward = float(
            obj.reward_rng.normal(
                loc=obj.means[arm],
                scale=1.0 / np.sqrt(obj.observation_precision),
            )
        )
    else:
        trial_index = int(np.sum(obj.counts))
        reward = float(obj._reward_table[trial_index, arm])
    return arm, reward


def _record_update(obj: Bandit, arm: int, reward: float) -> None:
    """Update common observations; subclasses update their own beliefs afterward."""
    if not isinstance(arm, (int, np.integer)) or not 0 <= arm < obj.n_arms:
        raise ValueError(f"arm must be an integer between 0 and {obj.n_arms - 1}")
    if not np.isfinite(reward):
        raise ValueError("reward must be finite")
    obj.counts[arm] += 1
    obj.reward_sums[arm] += float(reward)


def _run_experiment(
    obj: Bandit,
    n_trials: int = NUMBER_OF_TRIALS,
    reward_table: np.ndarray | None = None,
) -> ExperimentResult:
    """Execute a full sequential experiment, then retain trial-by-trial history.

    A table of shape (n_trials, n_arms) supplies potential Gaussian outcomes:
    both strategies can use the *same* table for a lower-noise comparison.
    An algorithm sees only the reward of the arm it actually selects.
    """
    if not isinstance(n_trials, (int, np.integer)) or n_trials <= 0:
        raise ValueError("n_trials must be a positive integer")
    if obj.result is not None or np.any(obj.counts):
        raise RuntimeError("Use a fresh algorithm instance for each experiment")
    if reward_table is not None:
        table = np.asarray(reward_table, dtype=float)
        if table.shape != (n_trials, obj.n_arms) or not np.all(np.isfinite(table)):
            raise ValueError("reward_table must be finite, with shape (n_trials, n_arms)")
    else:
        table = None

    arms = np.empty(n_trials, dtype=int)
    rewards = np.empty(n_trials, dtype=float)
    regrets = np.empty(n_trials, dtype=float)
    estimates = np.empty((n_trials, obj.n_arms), dtype=float)
    best_mean = float(np.max(obj.means))
    obj._reward_table = table
    try:
        for t in range(n_trials):
            arm, reward = obj.pull()
            obj.update(arm, reward)
            arms[t] = arm
            rewards[t] = reward
            # Pseudo-regret compares *true expected rewards*, not noisy draws.
            regrets[t] = best_mean - obj.means[arm]
            estimates[t, :] = obj.estimates
    finally:
        obj._reward_table = None

    obj.result = ExperimentResult(
        algorithm=obj.__class__.__name__,
        arms=arms,
        rewards=rewards,
        regrets=regrets,
        estimates=estimates,
        true_means=obj.means.copy(),
    )
    return obj.result


def _report_experiment(
    obj: Bandit, csv_path: str | Path, *, append: bool = False,
) -> dict[str, float]:
    """Write exactly {Bandit, Reward, Algorithm} and log summary statistics.

    Args:
        csv_path: Destination CSV. One row per selected advertisement/trial.
        append: True for the second algorithm so both share one CSV file.

    Returns:
        Dictionary of cumulative and average statistics for further checking.
    """
    if obj.result is None:
        raise RuntimeError("Run experiment() before report()")
    output_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if append and not output_path.is_file():
        raise FileNotFoundError("Run the first algorithm report before appending")

    with output_path.open("a" if append else "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if not append:
            writer.writerow(("Bandit", "Reward", "Algorithm"))
        for arm, reward in zip(obj.result.arms, obj.result.rewards):
            writer.writerow((int(arm) + 1, repr(float(reward)), obj.result.algorithm))

    total_reward = float(np.sum(obj.result.rewards))
    total_regret = float(np.sum(obj.result.regrets))
    trials = len(obj.result.rewards)
    summary = {
        "cumulative_reward": total_reward,
        "average_reward": total_reward / trials,
        "cumulative_regret": total_regret,
        "average_regret": total_regret / trials,
        "optimal_arm_fraction": obj.result.optimal_arm_fraction,
    }
    logger.info(f"{obj.result.algorithm}: cumulative reward = {total_reward:,.4f}")
    logger.info(f"{obj.result.algorithm}: average reward = {summary['average_reward']:.4f}")
    logger.info(f"{obj.result.algorithm}: cumulative regret = {total_regret:,.4f}")
    logger.info(f"{obj.result.algorithm}: average regret = {summary['average_regret']:.4f}")
    logger.info(
        f"{obj.result.algorithm}: optimal-ad selection = "
        f"{100.0 * summary['optimal_arm_fraction']:.2f}%"
    )
    return summary


class EpsilonGreedy(Bandit):
    """Sample-average Epsilon-Greedy with exploration probability 1/t.

    Every arm is pulled once at the start, avoiding estimates based on zero
    observations. Thereafter, at trial t (1-indexed), epsilon_t = 1/t.
    """

    def __init__(
        self, p=BANDIT_REWARD, *, epsilon: float = 1.0,
        observation_precision: float = 1.0, seed: int = DEFAULT_SEED + 1,
    ) -> None:
        _initialize_bandit(self, p, observation_precision=observation_precision, seed=seed)
        if not np.isfinite(epsilon) or epsilon < 0:
            raise ValueError("epsilon must be nonnegative and finite")
        self.epsilon = float(epsilon)

    def __repr__(self) -> str:
        return f"EpsilonGreedy(epsilon_t={self.epsilon}/t, means={self.means.tolist()})"

    def epsilon_at(self, trial: int) -> float:
        """Exploration probability at one-indexed trial t."""
        if trial < 1:
            raise ValueError("trial must be >= 1")
        return min(1.0, self.epsilon / trial)

    def pull(self) -> tuple[int, float]:
        """Select an arm using epsilon-greedy and observe its Gaussian reward."""
        untried = np.flatnonzero(self.counts == 0)
        if untried.size:
            arm = int(untried[0])  # Four-arm warm-up: 1, 2, 3, 4.
        else:
            t = int(np.sum(self.counts)) + 1
            if self.action_rng.random() < self.epsilon_at(t):
                arm = int(self.action_rng.integers(self.n_arms))
            else:
                best = np.flatnonzero(self.estimates == np.max(self.estimates))
                arm = int(self.action_rng.choice(best))  # Unbiased tie-breaking.
        return _observed_pull(self, arm)

    def update(self, arm: int, reward: float) -> None:
        """Update the selected arm's empirical mean by an online average."""
        _record_update(self, arm, reward)
        self.estimates[arm] = self.reward_sums[arm] / self.counts[arm]

    def experiment(
        self, n_trials: int = NUMBER_OF_TRIALS,
        reward_table: np.ndarray | None = None,
    ) -> ExperimentResult:
        """Run sequential epsilon-greedy decisions for n_trials."""
        return _run_experiment(self, n_trials, reward_table)

    def report(self, csv_path: str | Path, *, append: bool = False) -> dict[str, float]:
        """Save chosen arm/reward/algorithm to CSV and log performance."""
        return _report_experiment(self, csv_path, append=append)


# --------------------------------------#


class ThompsonSampling(Bandit):
    """Gaussian conjugate Thompson Sampling with *known observation precision*.

    Prior for each unknown reward mean: N(prior_mean, 1/prior_precision).
    Observation model: reward | mu ~ N(mu, 1/observation_precision).
    After n observations with sum S for arm i:

        posterior_precision = prior_precision + n * observation_precision
        posterior_mean = (prior_precision * prior_mean
                          + observation_precision * S) / posterior_precision

    An independent sample of each arm's posterior *mean* determines the action.
    """

    def __init__(
        self, p=BANDIT_REWARD, *, observation_precision: float = 1.0,
        prior_mean: float = 0.0, prior_precision: float = 0.01,
        seed: int = DEFAULT_SEED + 2,
    ) -> None:
        _initialize_bandit(self, p, observation_precision=observation_precision, seed=seed)
        if not np.isfinite(prior_mean):
            raise ValueError("prior_mean must be finite")
        if not np.isfinite(prior_precision) or prior_precision <= 0:
            raise ValueError("prior_precision must be strictly positive and finite")
        self.prior_mean = float(prior_mean)
        self.prior_precision = float(prior_precision)
        self.posterior_precisions = np.full(self.n_arms, self.prior_precision)
        self.posterior_means = np.full(self.n_arms, self.prior_mean)
        self.estimates[:] = self.prior_mean

    def __repr__(self) -> str:
        return (
            f"ThompsonSampling(known_precision={self.observation_precision}, "
            f"prior_mean={self.prior_mean}, prior_precision={self.prior_precision})"
        )

    def pull(self) -> tuple[int, float]:
        """Sample each arm's mean from its posterior; pull the largest sample."""
        samples = self.action_rng.normal(
            loc=self.posterior_means,
            scale=np.sqrt(1.0 / self.posterior_precisions),
        )
        arm = int(np.argmax(samples))
        return _observed_pull(self, arm)

    def update(self, arm: int, reward: float) -> None:
        """Apply exact Normal-Normal Bayesian updating to the selected arm."""
        _record_update(self, arm, reward)
        posterior_precision = (
            self.prior_precision + self.counts[arm] * self.observation_precision
        )
        posterior_mean = (
            self.prior_precision * self.prior_mean
            + self.observation_precision * self.reward_sums[arm]
        ) / posterior_precision
        self.posterior_precisions[arm] = posterior_precision
        self.posterior_means[arm] = posterior_mean
        self.estimates[arm] = posterior_mean

    def experiment(
        self, n_trials: int = NUMBER_OF_TRIALS,
        reward_table: np.ndarray | None = None,
    ) -> ExperimentResult:
        """Run n_trials with posterior sampling and updating after each reward."""
        return _run_experiment(self, n_trials, reward_table)

    def report(self, csv_path: str | Path, *, append: bool = False) -> dict[str, float]:
        """Save chosen arm/reward/algorithm to CSV and log performance."""
        return _report_experiment(self, csv_path, append=append)


# --------------------------------------#


class Visualization:
    """Build the homework's individual learning and comparison plots."""

    def __init__(
        self, epsilon_result: ExperimentResult,
        thompson_result: ExperimentResult,
        output_dir: str | Path,
    ) -> None:
        self.results = (epsilon_result, thompson_result)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot1(self) -> list[Path]:
        """Plot each algorithm's arm-mean learning on linear and log trial axes.

        One figure per algorithm; each has four estimated-mean curves and
        the associated dashed true means for easy convergence assessment.
        """
        files: list[Path] = []
        for result in self.results:
            times = np.arange(1, len(result.rewards) + 1)
            fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
            for axis in axes:
                for arm in range(len(result.true_means)):
                    (line,) = axis.plot(
                        times, result.estimates[:, arm],
                        linewidth=1.3, label=f"Ad {arm + 1} estimate",
                    )
                    axis.axhline(
                        result.true_means[arm], color=line.get_color(),
                        linestyle="--", alpha=0.6, linewidth=1,
                    )
                axis.set_xlabel("Trial number")
                axis.set_ylabel("Estimated expected reward")
                axis.grid(True, alpha=0.25)
            axes[0].set_title("Linear trial scale")
            axes[1].set_title("Logarithmic trial scale")
            axes[1].set_xscale("log")
            axes[0].legend(loc="best", fontsize=9)
            fig.suptitle(f"{result.algorithm}: learning each advertisement's mean")
            fig.text(
                0.5, 0.01, "Dashed horizontal lines = true expected rewards (1, 2, 3, 4)",
                ha="center", fontsize=9,
            )
            fig.tight_layout(rect=(0, 0.035, 1, 0.95))
            filename = "epsilon_greedy_learning.png" if result.algorithm == "EpsilonGreedy" else "thompson_sampling_learning.png"
            path = self.output_dir / filename
            fig.savefig(path, dpi=170, bbox_inches="tight")
            plt.close(fig)
            files.append(path)
            logger.info(f"Saved learning visualization: {path}")
        return files

    def plot2(self) -> Path:
        """Compare cumulative observed reward and cumulative expected regret."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        for result in self.results:
            times = np.arange(1, len(result.rewards) + 1)
            axes[0].plot(
                times, result.cumulative_rewards,
                label=result.algorithm, linewidth=1.8,
            )
            axes[1].plot(
                times, result.cumulative_regrets,
                label=result.algorithm, linewidth=1.8,
            )
        axes[0].set(title="Cumulative observed reward", xlabel="Trial number", ylabel="Total reward")
        axes[1].set(title="Cumulative pseudo-regret (lower is better)", xlabel="Trial number", ylabel="Total expected regret")
        for axis in axes:
            axis.grid(True, alpha=0.25)
            axis.legend()
        fig.suptitle("Epsilon-Greedy vs. Gaussian Thompson Sampling")
        fig.tight_layout()
        path = self.output_dir / "algorithm_comparison.png"
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved algorithm comparison: {path}")
        return path


def comparison(
    *, n_trials: int = NUMBER_OF_TRIALS, p=BANDIT_REWARD,
    epsilon: float = 1.0, observation_precision: float = 1.0,
    prior_mean: float = 0.0, prior_precision: float = 0.01,
    seed: int = DEFAULT_SEED, output_dir: str | Path = "results",
    make_plots: bool = True,
) -> tuple[ExperimentResult, ExperimentResult]:
    """Compare strategies on paired potential outcomes; save required outputs.

    Both algorithms face an identical, pre-generated Normal reward table.
    They never see rewards for arms they did not select. Their random choices
    use separate seeds, preventing accidental sharing of decision randomness.
    """
    means = np.asarray(p, dtype=float)
    if means.ndim != 1 or means.size == 0 or not np.all(np.isfinite(means)):
        raise ValueError("p must be nonempty and contain finite expected rewards")
    if not np.isfinite(observation_precision) or observation_precision <= 0:
        raise ValueError("observation_precision must be strictly positive and finite")
    if not isinstance(n_trials, (int, np.integer)) or n_trials <= 0:
        raise ValueError("n_trials must be a positive integer")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reward_rng = np.random.default_rng(seed)
    reward_table = reward_rng.normal(
        loc=means,
        scale=1.0 / np.sqrt(observation_precision),
        size=(n_trials, len(means)),
    )
    greedy = EpsilonGreedy(
        p=means, epsilon=epsilon,
        observation_precision=observation_precision, seed=seed + 1,
    )
    thompson = ThompsonSampling(
        p=means, observation_precision=observation_precision,
        prior_mean=prior_mean, prior_precision=prior_precision, seed=seed + 2,
    )
    logger.info(f"Experiment: {n_trials:,} trials per algorithm; true means = {means.tolist()}")
    logger.info(
        f"Normal observation precision = {observation_precision}; "
        f"Epsilon-Greedy epsilon_t = {epsilon}/t; "
        f"Thompson prior = N({prior_mean}, 1/{prior_precision})"
    )
    epsilon_result = greedy.experiment(n_trials, reward_table)
    thompson_result = thompson.experiment(n_trials, reward_table)

    csv_path = out / "rewards.csv"
    greedy.report(csv_path)
    thompson.report(csv_path, append=True)
    logger.info(f"Saved combined {2 * n_trials:,}-row CSV: {csv_path}")
    if make_plots:
        visuals = Visualization(epsilon_result, thompson_result, out)
        visuals.plot1()
        visuals.plot2()

    difference = (
        epsilon_result.cumulative_regrets[-1]
        - thompson_result.cumulative_regrets[-1]
    )
    logger.info(
        f"Difference in cumulative pseudo-regret (E-Greedy minus Thompson): "
        f"{difference:,.4f}; positive favors Thompson Sampling"
    )
    return epsilon_result, thompson_result


def robustness_analysis(
    *, repeats: int, n_trials: int, p=BANDIT_REWARD,
    epsilon: float = 1.0, observation_precision: float = 1.0,
    prior_mean: float = 0.0, prior_precision: float = 0.01,
    seed: int = DEFAULT_SEED, output_dir: str | Path = "results",
) -> Path:
    """Optional bonus: assess both methods over multiple independent seeds.

    A single stochastic run can be misleading. Repeated paired simulations
    provide means, sample standard deviations and standard errors for final
    cumulative pseudo-regret. This is intentionally off by default.
    """
    if not isinstance(repeats, int) or repeats < 2:
        raise ValueError("robustness repeats must be an integer >= 2")
    means = np.asarray(p, dtype=float)
    if means.ndim != 1 or means.size == 0 or not np.all(np.isfinite(means)):
        raise ValueError("p must be nonempty and contain finite expected rewards")
    if not np.isfinite(observation_precision) or observation_precision <= 0:
        raise ValueError("observation_precision must be strictly positive and finite")
    if not isinstance(n_trials, (int, np.integer)) or n_trials <= 0:
        raise ValueError("n_trials must be a positive integer")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_regrets = np.empty((repeats, 2))
    all_rewards = np.empty((repeats, 2))
    for run in range(repeats):
        # Distinct streams across runs, shared potential rewards within a run.
        run_seed = seed + 100_000 + 3 * run
        outcomes = np.random.default_rng(run_seed).normal(
            loc=means, scale=1.0 / np.sqrt(observation_precision),
            size=(n_trials, len(means)),
        )
        strategies = (
            EpsilonGreedy(
                means, epsilon=epsilon, observation_precision=observation_precision,
                seed=run_seed + 1,
            ),
            ThompsonSampling(
                means, observation_precision=observation_precision,
                prior_mean=prior_mean, prior_precision=prior_precision,
                seed=run_seed + 2,
            ),
        )
        for i, strategy in enumerate(strategies):
            result = strategy.experiment(n_trials, outcomes)
            all_regrets[run, i] = result.cumulative_regrets[-1]
            all_rewards[run, i] = result.cumulative_rewards[-1]

    # Save all runs, not only a mean, to expose rare failures and variability.
    detail_path = out / "bonus_robustness_runs.csv"
    with detail_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "Run", "Seed", "EpsilonGreedyCumulativeReward",
            "ThompsonSamplingCumulativeReward", "EpsilonGreedyCumulativeRegret",
            "ThompsonSamplingCumulativeRegret",
        ))
        for run in range(repeats):
            writer.writerow((
                run + 1, seed + 100_000 + 3 * run,
                *map(float, all_rewards[run]), *map(float, all_regrets[run]),
            ))

    path = out / "bonus_robustness.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "Algorithm", "Runs", "MeanCumulativeReward", "SDCumulativeReward",
            "MeanCumulativeRegret", "SDCumulativeRegret", "SEMeanRegret",
        ))
        for i, name in enumerate(("EpsilonGreedy", "ThompsonSampling")):
            mean_regret = float(np.mean(all_regrets[:, i]))
            sd_regret = float(np.std(all_regrets[:, i], ddof=1))
            writer.writerow((
                name, repeats, float(np.mean(all_rewards[:, i])),
                float(np.std(all_rewards[:, i], ddof=1)),
                mean_regret, sd_regret, sd_regret / np.sqrt(repeats),
            ))
            logger.info(
                f"Bonus {name}: {repeats} runs; mean cumulative regret "
                f"= {mean_regret:.2f}; SE = {sd_regret / np.sqrt(repeats):.2f}"
            )
    logger.info(f"Saved optional robustness summary: {path}")
    logger.info(f"Saved optional per-run values: {detail_path}")

    fig, axis = plt.subplots(figsize=(10, 5))
    positions = np.arange(1, repeats + 1)
    axis.scatter(positions, all_regrets[:, 0], label="Epsilon-Greedy", s=42, marker="o")
    axis.scatter(positions, all_regrets[:, 1], label="Thompson Sampling", s=42, marker="x")
    axis.set_yscale("log")  # Allows small regrets and rare large regrets to be seen.
    axis.set_xlabel("Independent simulation run")
    axis.set_ylabel("Final cumulative pseudo-regret (log scale)")
    axis.set_title(f"Optional bonus: robustness across {repeats} random seeds")
    axis.legend()
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    plot_path = out / "bonus_regret_distribution.png"
    fig.savefig(plot_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved optional robustness visualization: {plot_path}")
    return path


def main() -> None:
    """Parse optional settings and run the standard assignment experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=NUMBER_OF_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epsilon", type=float, default=1.0, help="Coefficient for epsilon_t = epsilon/t")
    parser.add_argument("--precision", type=float, default=1.0, help="Known Normal observation precision (=1/variance)")
    parser.add_argument("--prior-mean", type=float, default=0.0)
    parser.add_argument("--prior-precision", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--robustness-runs", type=int, default=0, help="Optional independent repeats; >=2")
    args = parser.parse_args()

    logger.remove()  # Replace starter's debug/info/warning/error/critical demo.
    logger.add(sys.stdout, level="INFO", colorize=False, format="{time:HH:mm:ss} | {level} | {message}")
    comparison(
        n_trials=args.trials, epsilon=args.epsilon,
        observation_precision=args.precision,
        prior_mean=args.prior_mean, prior_precision=args.prior_precision,
        seed=args.seed, output_dir=args.output_dir,
    )
    if args.robustness_runs:
        robustness_analysis(
            repeats=args.robustness_runs, n_trials=args.trials,
            epsilon=args.epsilon, observation_precision=args.precision,
            prior_mean=args.prior_mean, prior_precision=args.prior_precision,
            seed=args.seed, output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
