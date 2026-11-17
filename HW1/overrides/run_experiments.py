"""Reproduce the three assigned experiments and export their numeric results."""
from __future__ import annotations

import csv
from pathlib import Path

from gaussian_bandit import run_gaussian_bandit

EXPERIMENTS = [
    ("Short run", (1.0, 2.0, 3.0), 1.0, 100),
    ("Longer run", (1.0, 2.0, 3.0), 1.0, 1000),
    ("Higher noise", (1.0, 2.0, 3.0), 3.0, 1000),
]
SEED = 42


def main() -> None:
    rows = []
    for name, true_means, sigma, rounds in EXPERIMENTS:
        result = run_gaussian_bandit(true_means, sigma=sigma, rounds=rounds, seed=SEED)
        rerun = run_gaussian_bandit(true_means, sigma=sigma, rounds=rounds, seed=SEED)
        assert result.total_reward == rerun.total_reward
        assert (result.posterior_means == rerun.posterior_means).all()
        assert int(result.selection_counts.sum()) == rounds
        row = {
            "experiment": name,
            "seed": SEED,
            "true_means": "[1, 2, 3]",
            "sigma": sigma,
            "rounds": rounds,
            "estimated_mean_1": round(float(result.posterior_means[0]), 6),
            "estimated_mean_2": round(float(result.posterior_means[1]), 6),
            "estimated_mean_3": round(float(result.posterior_means[2]), 6),
            "selections_1": int(result.selection_counts[0]),
            "selections_2": int(result.selection_counts[1]),
            "selections_3": int(result.selection_counts[2]),
            "recommendation": f"Alternative {result.recommendation}",
            "total_observed_reward": round(result.total_reward, 6),
        }
        rows.append(row)
        print(f"{name}: {row}")

    filename = Path(__file__).parent / "experiment_results.csv"
    with filename.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {filename}")


if __name__ == "__main__":
    main()
