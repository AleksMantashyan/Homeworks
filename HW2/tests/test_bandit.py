"""Regression and mathematical checks for Homework 2.

Run from HW2: python -m unittest discover -s tests -v
Uses only Python standard library plus the homework runtime dependencies.
"""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Bandit import (  # noqa: E402
    Bandit, EpsilonGreedy, ThompsonSampling, Visualization,
    comparison, robustness_analysis,
)


class TestBanditContract(unittest.TestCase):
    def test_abstract_contract_preserved(self):
        with self.assertRaises(TypeError):
            Bandit([1, 2, 3, 4])
        expected = {"__init__", "__repr__", "pull", "update", "experiment", "report"}
        self.assertEqual(expected, Bandit.__abstractmethods__)
        self.assertTrue(issubclass(EpsilonGreedy, Bandit))
        self.assertTrue(issubclass(ThompsonSampling, Bandit))

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            EpsilonGreedy([], seed=2)
        with self.assertRaises(ValueError):
            EpsilonGreedy([1, 2], epsilon=-0.1)
        with self.assertRaises(ValueError):
            EpsilonGreedy([1, 2], observation_precision=0)
        with self.assertRaises(ValueError):
            ThompsonSampling([1, 2], prior_precision=0)
        with self.assertRaises(ValueError):
            ThompsonSampling([1, 2], prior_mean=float("nan"))
        with self.assertRaises(ValueError):
            EpsilonGreedy([1, 2]).experiment(0)
        with self.assertRaises(ValueError):
            EpsilonGreedy([1, 2]).experiment(8, np.zeros((7, 2)))
        with self.assertRaises(ValueError):
            EpsilonGreedy([1, 2]).update(2, 1.5)


class TestMathematics(unittest.TestCase):
    def test_epsilon_is_one_over_t(self):
        policy = EpsilonGreedy([1, 2, 3, 4], epsilon=1.0)
        self.assertEqual(policy.epsilon_at(1), 1.0)
        self.assertAlmostEqual(policy.epsilon_at(5), 0.2)
        self.assertAlmostEqual(policy.epsilon_at(20000), 1 / 20000)
        with self.assertRaises(ValueError):
            policy.epsilon_at(0)

    def test_sample_mean_update(self):
        policy = EpsilonGreedy([1, 2], seed=3)
        policy.update(1, 4.0)
        policy.update(1, 2.0)
        self.assertEqual(policy.counts.tolist(), [0, 2])
        self.assertAlmostEqual(policy.estimates[1], 3.0)
        self.assertEqual(policy.estimates[0], 0.0)

    def test_gaussian_posterior_known_precision(self):
        # Prior N(0, 1/0.5) and known likelihood precision = 2.
        # After r=3: posterior precision = 0.5+2 = 2.5,
        # posterior mean = (0.5*0 + 2*3)/2.5 = 2.4.
        policy = ThompsonSampling(
            [1, 2], prior_mean=0, prior_precision=0.5,
            observation_precision=2, seed=11,
        )
        policy.update(1, 3.0)
        self.assertAlmostEqual(policy.posterior_precisions[1], 2.5)
        self.assertAlmostEqual(policy.posterior_means[1], 2.4)
        self.assertAlmostEqual(policy.posterior_precisions[0], 0.5)
        self.assertEqual(policy.posterior_means[0], 0)
        policy.update(1, 1.0)
        self.assertAlmostEqual(policy.posterior_precisions[1], 4.5)
        self.assertAlmostEqual(policy.posterior_means[1], 8 / 4.5)
        self.assertAlmostEqual(policy.estimates[1], 8 / 4.5)

    def test_constant_reward_table_makes_pseudo_regret_exact(self):
        means = np.array([1.0, 2.0, 3.0, 4.0])
        outcomes = np.tile(means, (65, 1))
        for cls in (EpsilonGreedy, ThompsonSampling):
            with self.subTest(algorithm=cls.__name__):
                policy = cls(means, seed=31)
                result = policy.experiment(65, reward_table=outcomes)
                np.testing.assert_allclose(result.rewards, means[result.arms])
                np.testing.assert_allclose(result.regrets, 4.0 - means[result.arms])
                self.assertTrue(np.all(np.diff(result.cumulative_regrets) >= 0))
                self.assertEqual(result.rewards.shape, (65,))
                self.assertEqual(result.estimates.shape, (65, 4))
                self.assertEqual(int(np.sum(policy.counts)), 65)
                self.assertAlmostEqual(
                    result.cumulative_regrets[-1], float(np.sum(result.regrets))
                )
                with self.assertRaises(RuntimeError):
                    policy.experiment(2)

    def test_initial_epsilon_greedy_warmup(self):
        policy = EpsilonGreedy([1, 2, 3, 4], seed=17)
        outcomes = np.ones((10, 4))
        result = policy.experiment(10, outcomes)
        self.assertEqual(result.arms[:4].tolist(), [0, 1, 2, 3])

    def test_seeded_reproducibility(self):
        for cls in (EpsilonGreedy, ThompsonSampling):
            a = cls([1, 2, 3, 4], seed=39).experiment(120)
            b = cls([1, 2, 3, 4], seed=39).experiment(120)
            np.testing.assert_array_equal(a.arms, b.arms)
            np.testing.assert_array_equal(a.rewards, b.rewards)
            np.testing.assert_array_equal(a.regrets, b.regrets)


class TestDeliverables(unittest.TestCase):
    def test_csv_plots_and_report(self):
        with tempfile.TemporaryDirectory() as folder:
            greedy, thompson = comparison(n_trials=35, seed=15, output_dir=folder)
            path = Path(folder)
            with (path / "rewards.csv").open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 70)
            self.assertEqual(list(rows[0]), ["Bandit", "Reward", "Algorithm"])
            self.assertEqual(
                {r["Algorithm"] for r in rows}, {"EpsilonGreedy", "ThompsonSampling"}
            )
            self.assertEqual(sum(r["Algorithm"] == "EpsilonGreedy" for r in rows), 35)
            self.assertEqual(sum(r["Algorithm"] == "ThompsonSampling" for r in rows), 35)
            for row in rows:
                self.assertIn(int(row["Bandit"]), (1, 2, 3, 4))
                self.assertTrue(np.isfinite(float(row["Reward"])))
            for img in (
                "epsilon_greedy_learning.png",
                "thompson_sampling_learning.png",
                "algorithm_comparison.png",
            ):
                self.assertGreater((path / img).stat().st_size, 5_000)
            self.assertEqual(greedy.algorithm, "EpsilonGreedy")
            self.assertEqual(thompson.algorithm, "ThompsonSampling")

    def test_no_plots_option(self):
        with tempfile.TemporaryDirectory() as folder:
            comparison(n_trials=8, output_dir=folder, make_plots=False)
            self.assertTrue((Path(folder) / "rewards.csv").is_file())
            self.assertFalse((Path(folder) / "algorithm_comparison.png").exists())

    def test_optional_bonus_repeats(self):
        with tempfile.TemporaryDirectory() as folder:
            path = robustness_analysis(repeats=3, n_trials=20, output_dir=folder)
            with path.open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual({r["Algorithm"] for r in rows}, {"EpsilonGreedy", "ThompsonSampling"})
            self.assertEqual({r["Runs"] for r in rows}, {"3"})


if __name__ == "__main__":
    unittest.main()
