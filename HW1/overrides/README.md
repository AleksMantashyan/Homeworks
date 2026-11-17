# Gaussian Thompson Sampling — DS223 assignment

**GitHub repository:** [Add the URL of your own GitHub repository after pushing](https://github.com/)

**Deployed Streamlit app:** [Add your Streamlit URL after deploying](https://share.streamlit.io/)

## Overview

This project extends the **Bayesian Carousel Bandits** Streamlit frontend with a fourth page called **Gaussian Bandit**. The original three pages remain available:

1. **Create Carousels** — creates and manages a carousel with configurable total and visible alternatives.
2. **Interaction** — displays Thompson-sampled alternatives and collects click/no-click feedback.
3. **Analytics** — shows impressions, clicks, estimated click-through rates, posterior distributions, and history.
4. **Gaussian Bandit (new)** — runs three-armed Thompson sampling with continuous Normal-distributed rewards and known noise.

### Source credit

The original frontend is based on **[hovhannisyan91/DS223_Bayesian_AB_Testing, `app/front/`](https://github.com/hovhannisyan91/DS223_Bayesian_AB_Testing/tree/main/app/front)**. The assignment's older URL `DS223_Group_Project` redirects to that repository. Original carousel pages and `bandit_utils.py` were retained; the Gaussian implementation, tests, and this documentation were added for this assignment.

## How the original algorithm works: Beta-Bernoulli Thompson sampling

The carousel models each alternative's unknown **click probability**. Each alternative starts with an independent **Beta(1, 1)** prior (parameters `alpha=1`, `beta=1`). On each carousel display, `sample_carousel` samples a click probability from each alternative's current Beta posterior and shows the highest-scoring `visible_count` alternatives.

When the user presses **Click** on a displayed alternative, that arm receives a binary success (`alpha += 1`, `clicks += 1`). Every other **displayed** alternative receives a failure (`beta += 1`). When the user chooses **No click / Next round**, all displayed alternatives receive a failure. Each displayed alternative also receives an impression. Alternatives not displayed in the round are unchanged. The event history records a **round reward of 1 if there was a click, and 0 otherwise**, while the per-alternative Bayesian updates use each displayed arm's clicked/not-clicked status. The algorithm resamples the next displayed set after feedback. The posterior mean of each click probability is `alpha / (alpha + beta)`; Analytics labels the highest posterior-mean alternative the champion.

The app calls `init_state` from `main.py`, and all three original pages use the same `st.session_state` dictionary. It contains `carousels`, the selected `active_carousel_id`, each carousel's `alternatives` DataFrame, the current display IDs, round number, and interaction history. Thus creating a carousel, clicking an alternative, and opening Analytics affect and read the same in-session values. **The frontend has no implemented database persistence**: the utility file still contains `TODO` comments for backend API reads and writes, so the current session is not a persistent user account or database record.

## New algorithm: Normal-Normal Gaussian Thompson sampling

Unlike clicks, our **reward is continuous** and may be negative. We simulate exactly three alternatives with true means `mu_1, mu_2, mu_3`. A reward from selected alternative `i` is drawn from:

```text
Reward | chosen alternative i ~ Normal(mu_i, sigma^2)
```

The observation noise **sigma > 0 is known and shared** across all three arms. Each unknown arm mean starts with the **same independent Normal(0, 25) prior** (mean 0, variance 25, standard deviation 5).

For every round:

1. Sample a plausible mean `theta_i` from each arm's **current learned Normal posterior**, not its true mean.
2. Select the alternative with the largest sampled `theta_i`.
3. Simulate one continuous reward from `Normal(true_mean_of_selected_arm, sigma^2)`.
4. Update **only the selected alternative's** posterior with that reward and the known sigma; leave the other two unchanged.

The simulator uses true means to **generate** rewards and to **evaluate** learning. It never uses them to decide which alternative to select. The learner sees only its own posterior state and observed rewards. This is the exploration/exploitation tradeoff of Thompson sampling.

### Posterior update

If an arm currently has posterior `Normal(m, v)` and receives reward `r` with known noise standard deviation `sigma`, the new Normal posterior is:

```text
updated_precision = 1/v + 1/sigma^2
v_new = 1 / updated_precision
m_new = v_new * (m/v + r/sigma^2)
```

For example, from prior `Normal(0, 25)`, one observed reward of `3` and `sigma=1` produces posterior variance `25/26 ≈ 0.96154` and mean `75/26 ≈ 2.88462`. **Only the chosen arm is updated.**

The final recommendation is **the alternative with the highest final posterior mean** (not the highest true mean and not the last randomly sampled theta). `gaussian_bandit.py` contains the simulator; `pages/Gaussian_Bandit.py` contains the Streamlit UI. The new Gaussian page saves its last run in a separate `st.session_state.gaussian_bandit_result`, leaving the original carousel state untouched.

## Reproducible experiments

All experiments use prior `Normal(0, 25)`, true means **[1, 2, 3]**, and random seed **42**. Only sigma and round count vary. Estimated posterior means and selection counts are ordered by alternative (1, 2, 3).

| Experiment | Known σ | Rounds | Estimated posterior means (1, 2, 3) | Selection counts (1, 2, 3) | Final recommendation | Total observed reward |
|---|---:|---:|---|---|---|---:|
| Short run | 1 | 100 | [1.570, 1.729, 2.995] | [4, 2, 94] | Alternative 3 | 291.518 |
| Longer run | 1 | 1,000 | [1.306, 2.142, 2.979] | [5, 13, 982] | Alternative 3 | 2960.201 |
| Higher noise | 3 | 1,000 | [1.154, 1.484, 2.954] | [21, 23, 956] | Alternative 3 | 2884.604 |

### Interpretation

- **More observations:** Increasing rounds from 100 to 1,000 (sigma=1) makes the overall estimates more accurate: mean absolute estimation error across the three true means decreases from approximately **0.282** to **0.156**. The algorithm increasingly exploits the best arm: arm 3 is chosen **982/1000** times in the longer run (versus 94/100 in the short run). These counts show the best arm is heavily preferred; they also show that Thompson sampling does not sample the others equally often.
- **Higher noise:** Raising sigma from 1 to 3 at 1,000 rounds makes each observation less informative. The algorithm explores alternatives 1 and 2 more frequently (**21 and 23** selections instead of **5 and 13**). Overall mean absolute estimation error rises to approximately **0.239**; the recommendation remains arm 3.
- **Unexpected but valid result:** Arm 1's posterior estimate happens to be **closer** to the true value under higher noise (1.154 versus 1.306), even though overall estimation accuracy is worse. Individual simulation outcomes are random; greater noise does not guarantee every individual estimate will worsen. Also, arm 3's short-run estimate (2.995) happens to be closer to 3 than its longer-run estimate (2.979). More observations improve statistical learning in general, not monotonically for every estimate in every seeded trajectory.
- **Sanity checks:** Selection counts sum to the number of rounds in all three experiments (`4+2+94=100`, `5+13+982=1000`, `21+23+956=1000`). Running an identical configuration with seed 42 gives exactly the same selections, rewards, posteriors, and totals with the same NumPy version. The test suite also verifies that unselected arms are unchanged and negative rewards are kept rather than clipped to zero.

Run the experiments again with `uv run python run_experiments.py`. The script prints the computed results and recreates `experiment_results.csv`.

## Installation and execution

Requirements: **Python 3.12**, [`uv`](https://docs.astral.sh/uv/) and Git.

The repository root is the original `app/front/` source copied into an **independent** repository; do not run the app from the parent upstream repository.

```bash
# From the root of YOUR new GitHub repository
uv python pin 3.12
uv sync --locked
uv run streamlit run main.py
```

To run the numerical tests:

```bash
uv run --with pytest pytest -q tests/test_gaussian_bandit.py
```

The original frontend dependencies are kept in the original `pyproject.toml` and `uv.lock`, including NumPy, pandas, SciPy, and Streamlit. Do not commit `.venv/`, caches, credentials, or Streamlit secrets.

## How to use the product

**Original three pages:** Create a carousel; open Interaction and click one visible alternative or choose "No click"; open Analytics to inspect the updated Beta posteriors and cumulative click reward.

**New Gaussian page:** Choose three true means, known sigma, a positive integer number of rounds, and an integer random seed. Click **Run Gaussian Bandit**. The results table shows each arm's true mean, final posterior estimated mean, posterior uncertainty, and selection count. Summary cards show total observed reward, final recommendation, and the true best arm (for simulation evaluation). Two charts display posterior learning and cumulative reward. Expand the reward history for individual real-valued rewards, including negative observations.

## GitHub + Streamlit Community Cloud deployment

1. Create a new independent GitHub repository and push the **frontend root files** to its `main` branch, including `main.py`, the `pages/` directory, `bandit_utils.py`, `gaussian_bandit.py`, `pyproject.toml`, `uv.lock`, and this README.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/), choose to create an app from your GitHub repository.
3. Set **branch `main`**, **entrypoint `main.py`**, and **Python 3.12** in Advanced settings; deploy.
4. Open the deployed app and confirm the original pages still work. On Gaussian Bandit, try `[1,2,3]`, sigma `1`, rounds `100`, seed `42`, and verify the counts sum to 100 and the final recommended arm is 3.
5. Replace the two placeholder links at the top of this README with your actual **GitHub** and **deployed Streamlit** URLs before submission.

**Submission:** Your own GitHub repository URL and your own deployed Streamlit URL. Be ready for a **3–5 minute presentation** explaining the original Beta model, Gaussian posterior update with known sigma, the live page, and the three experiments.
