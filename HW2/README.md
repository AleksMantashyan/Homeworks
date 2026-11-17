# Homework 2 — Epsilon-Greedy vs. Thompson Sampling

**Course:** A/B Testing  
**Assignment:** Compare two multi-armed bandit algorithms for four advertisement options.  
**Required experiment:** 20,000 trials **per algorithm** with `Bandit_Reward = [1, 2, 3, 4]`.

This solution implements the supplied `Bandit` abstract interface without removing any of its original methods. It uses the requested `EpsilonGreedy`, `ThompsonSampling`, `Visualization`, and `comparison` interfaces. Experiments are reproducible, documented, and tested.

## 1. Run the assignment

Requires **Python 3.10 or later**. From the `HW2` folder:

```bash
python -m pip install -r requirements.txt
python Bandit.py
```

This runs both algorithms for 20,000 trials each, writes **40,000 rows** of reward observations, generates the plots, and logs cumulative/average reward and cumulative/average regret. You can reproduce the output with the default random seed (`2026`).

Run tests with:

```bash
python -m unittest discover -s tests -v
```

Optional settings:

```bash
python Bandit.py --trials 20000 --seed 2026 --epsilon 1.0 --precision 1.0
```

**For the bonus repeated-run experiment:**

```bash
python Bandit.py --robustness-runs 30
```

This additional command reruns the main experiment before running the 30 independent replications. Results are generated in `results/` relative to `Bandit.py`, even if the command is called from another working directory.

## 2. Explicit statistical assumption

The PDF specifies four `Bandit_Reward` values and asks for Thompson Sampling with *known precision*, but it **does not specify a probability distribution for the rewards**. Therefore, this solution models **Gaussian rewards**:

$$
  R_{t,i}\mid \mu_i \sim \mathcal N(\mu_i,\,\tau^{-1}),
  \qquad \boldsymbol\mu=(1,2,3,4),\qquad \tau=1.
$$

- `Bandit_Reward` values are **true expected rewards (means)**, not fixed rewards and not probabilities.
- The known observation precision is `tau = 1`, so **variance = 1 / tau = 1** and **standard deviation = 1**.
- Rewards vary randomly around the four means. They need not be integers and can sometimes be negative or above four.
- The optimal advertisement is **Bandit 4**, with expected reward 4.

This interpretation is appropriate for a Normal-Normal Thompson sampler with known precision. A Bernoulli click model would require reward probabilities between 0 and 1; the supplied values `[1,2,3,4]` cannot directly be Bernoulli probabilities.

## 3. Epsilon-Greedy

At every time step, Epsilon-Greedy either chooses a uniformly random advertisement (exploration) or one with the highest observed sample mean (exploitation). Each advertisement is tried once at the beginning, so no mean needs to be estimated from zero observations.

After this four-trial initialization, the required exploration schedule is:

$$
  \epsilon_t = \frac{1}{t}, \quad t=5,6,\ldots,20000.
$$

The class permits an optional exploration coefficient `epsilon`, giving `min(1, epsilon/t)`; the default is **exactly `1/t`**. Sample means are updated online using the running reward sum and count:

$$
  \widehat\mu_i = \frac{\sum_{s:\,A_s=i}R_{s,i}}{N_i}.
$$

Ties between estimated best advertisements are broken uniformly at random.

## 4. Gaussian Thompson Sampling with known precision

The unknown expected reward of advertisement `i` has a conjugate Gaussian prior:

$$
  \mu_i \sim \mathcal N(m_0,\lambda_0^{-1}),
  \qquad m_0=0,\qquad \lambda_0=0.01.
$$

The prior variance is `100`, so the prior is deliberately weak relative to the observation precision `tau = 1`. For advertisement `i`, after `n_i` observed rewards totaling `S_i`, the **exact posterior** is:

$$
  \lambda_i^{\mathrm{post}}=\lambda_0+n_i\tau,
$$

$$
  m_i^{\mathrm{post}}=
  \frac{\lambda_0m_0+\tau S_i}{\lambda_0+n_i\tau}.
$$

At each trial the sampler independently draws:

$$
  \theta_i \sim \mathcal N\left(m_i^{\mathrm{post}},\,
    (\lambda_i^{\mathrm{post}})^{-1}\right),
  \qquad A_t=\arg\max_i\theta_i.
$$

It observes the reward from `A_t` and updates **only that advertisement's posterior**. Importantly, the selection draw is from the **posterior over the unknown mean**, not the posterior predictive reward distribution; injecting observation noise into the selection samples would be a different algorithm.

## 5. Experiment design, regret, and reproducibility

The 20,000-trial comparison uses a shared, pre-generated table of potential Gaussian rewards: one value for each `(trial, advertisement)`. Each algorithm sees only the chosen advertisement's reward. Sharing the underlying potential outcomes makes the reward comparison less sensitive to unrelated simulation noise; the strategies use separate, independently seeded action random generators.

The comparison reports:

- **Cumulative observed reward:** $\sum_{t=1}^{T}R_{t,A_t}$.
- **Average observed reward:** cumulative observed reward divided by $T$.
- **Instantaneous pseudo-regret:** $\mu^*-\mu_{A_t}$, where $\mu^*=4$.
- **Cumulative pseudo-regret:** $\sum_{t=1}^{T}(4-\mu_{A_t})$.
- **Optimal-arm selection rate:** proportion of trials selecting advertisement 4.

The assignment does not define whether "regret" means realized regret or expected/pseudo-regret. We explicitly use **pseudo-regret**, the standard nonnegative expected-reward gap for a stochastic bandit. Using `4 - observed_reward` instead would yield negative stepwise values from Gaussian noise and would mix estimation noise into the policy-quality metric.

A single run does **not** prove an algorithm is universally better; outcomes depend on the seed, prior, noise, and exploration schedule.

## 6. Results from the default 20,000-trial run

Parameters: `seed=2026`, `epsilon=1.0`, `observation_precision=1.0`, `prior_mean=0.0`, `prior_precision=0.01`.

| Statistic | Epsilon-Greedy | Thompson Sampling |
|---|---:|---:|
| Number of trials | 20,000 | 20,000 |
| Cumulative observed reward | 80,052.3242 | 80,053.1756 |
| Average observed reward | 4.0026 | 4.0027 |
| Cumulative pseudo-regret | 26.0000 | 25.0000 |
| Average pseudo-regret | 0.0013 | 0.0013 |
| Optimal-arm selection rate | 99.94% | 99.91% |

Both algorithms identified advertisement 4 in this run. Thompson Sampling finished with **one unit less** pseudo-regret, which is too small a single-run difference to support a general performance claim. The cumulative observed reward may exceed 80,000 because individual Gaussian rewards can be greater than the optimal arm's expected reward of 4.

## 7. Saved outputs

| File | Contents |
|---|---|
| `results/rewards.csv` | Exactly **`Bandit,Reward,Algorithm`**, with 40,000 observation rows: 20,000 for each strategy. `Bandit` is numbered 1–4. |
| `results/epsilon_greedy_learning.png` | Epsilon-Greedy estimated means for each advertisement, with both linear and logarithmic trial axes and dashed true means. |
| `results/thompson_sampling_learning.png` | Thompson Sampling posterior-mean learning curves on linear and logarithmic trial axes. |
| `results/algorithm_comparison.png` | Cumulative observed rewards and cumulative pseudo-regrets on common trial axes. |
| `results/bonus_robustness.csv` | Optional 30-run means, sample standard deviations, and standard error of cumulative pseudo-regret. |
| `results/bonus_robustness_runs.csv` | Individual reward and regret outcomes for all 30 repeated simulations. |
| `results/bonus_regret_distribution.png` | Optional logarithmic scatter plot of cumulative pseudo-regret across independent runs. |

All plots are saved directly, without requiring a graphical desktop or interactive window.

## 8. BONUS — Suggested improved implementation plan (10 points)

A single seeded experiment can give a **misleading ranking** of stochastic bandit policies, especially when Epsilon-Greedy's exploration decreases as aggressively as `1/t`. A stronger experiment design would be:

1. **Separate environment, policy, and evaluation responsibilities.** Keep the true reward model and potential outcomes in a reusable environment; give each policy only observed actions and rewards; maintain plotting, export, and significance analysis in an independent evaluator. This prevents accidental access to hidden means inside a decision policy and enables reuse with different reward distributions.
2. **Run many independent paired experiments instead of trusting one seed.** In each repetition, reuse the same potential-outcome table for both algorithms but create a fresh table and fresh policy random seeds for each repetition. Record average and median cumulative regret, reward, standard deviations, variability across seeds, and paired algorithm differences.
3. **Report uncertainty and outliers.** Plot distributions and compare robust summaries (e.g., median and upper quantiles) as well as means. Calculate confidence intervals across independent simulation runs. This is especially important because one policy can have rare, costly failures that are invisible in a good-looking default run.
4. **Evaluate sensitivity to hyperparameters and assumptions.** Test alternative `epsilon/t` coefficients, prior means/precisions, observation noise precisions, and arm-reward gaps. An algorithm ranking for four Gaussian arms need not generalize to another environment. A fair comparison should state the tuning protocol separately from final evaluation.
5. **For real advertisements, validate the reward model and safeguards.** If the outcome is a binary click, use a Bernoulli/Beta model instead of treating `[1,2,3,4]` as probabilities; if it is monetary revenue, consider its distribution and delayed conversions. Monitor fairness, cost limits, drift, and guardrail metrics during online deployment.

**Bonus implementation included:** The optional `--robustness-runs 30` actually executes step 2, exports run-level and summary results for step 3, and saves a log-scale plot that makes rare failures visible.

### Observation from 30 additional independent runs

With the same 20,000-trial configuration and different random seeds:

| Statistic | Epsilon-Greedy | Thompson Sampling |
|---|---:|---:|
| Mean cumulative pseudo-regret | 1,354.00 | 29.77 |
| Median cumulative pseudo-regret | 19.00 | 29.00 |
| Lowest cumulative pseudo-regret | 7 | 14 |
| Highest cumulative pseudo-regret | 20,010 | 44 |
| Runs with pseudo-regret above 10,000 | 2 / 30 | 0 / 30 |

These results demonstrate why the single-run comparison should not be the sole basis for evaluating a policy. **Epsilon-Greedy typically performed well but experienced two near-complete failures to identify the best advertisement.** Because epsilon decreases as `1/t`, early misleading observations can leave insufficient later exploration. Thompson Sampling was more consistent over these 30 seeds, although its typical (median) pseudo-regret was a little higher. Thirty runs remain a limited sample; much larger replication counts would be needed for precise tail-risk estimates.

## 9. Submission structure

Use **one** GitHub repository named `Homeworks` and put this homework under its own folder:

```text
Homeworks/
├── HW1/                  # Move previously submitted HW1 files here
└── HW2/
    ├── Bandit.py
    ├── requirements.txt
    ├── README.md
    ├── .gitignore
    ├── tests/
    │   └── test_bandit.py
    └── results/
        ├── rewards.csv
        ├── epsilon_greedy_learning.png
        ├── thompson_sampling_learning.png
        ├── algorithm_comparison.png
        ├── bonus_robustness.csv
        ├── bonus_robustness_runs.csv
        └── bonus_regret_distribution.png
```

Keep any other assignment in its own `HW3/`, etc. **Commit and push** the changes, check that the files are visible on the GitHub website, and submit the **repository link only** to Moodle as instructed. `HW1` is shown as a placeholder in the diagram; it is not included in this HW2 package because the earlier HW1 files were not uploaded here.
