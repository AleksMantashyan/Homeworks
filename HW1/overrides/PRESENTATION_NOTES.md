# DS223 presentation notes (3–5 minutes)

## 0:00–0:45 — What changed?

"The starting app is a Bayesian carousel system with binary click/no-click feedback. It uses a Beta prior to estimate each alternative's click probability. Each round, it samples probabilities and displays the best-scoring alternatives. My extension adds a Gaussian Bandit page for situations where a reward can be a real number, such as revenue or a numerical score, instead of just 0 or 1. I kept the three original pages available."

## 0:45–1:45 — How does Gaussian Thompson sampling work?

"There are exactly three alternatives. We assume each unknown mean starts with the same Normal(0, 25) prior. The reward from arm i has a Normal distribution with unknown true mean mu_i and known standard deviation sigma. On every round, we draw a possible mean from each current posterior, choose the arm with the highest draw, observe one continuous reward, and update only that arm. The true means are only used by the simulator to generate rewards, not by the selection rule."

"The update uses precision. The new precision equals the old precision plus 1 divided by sigma squared. The new mean is a precision-weighted average of the old estimate and the new reward. Higher sigma means less information per observed reward."

## 1:45–2:45 — Live demonstration

Open the deployed app. Point out the original navigation pages, then select **Gaussian Bandit**. Enter:

- Three true means: `1, 2, 3`
- Known sigma: `1`
- Rounds: `100`
- Seed: `42`

Click **Run Gaussian Bandit**. Show the summary table, posterior estimated means, selection counts, final recommendation, and total observed reward. Explain the posterior learning chart. Briefly show the reward history: **negative outcomes are kept as real numbers, not converted to zero or one.**

## 2:45–3:45 — Experiments

"I tested three cases using seed 42. With 100 rounds and sigma 1, the algorithm chose arm 3 in 94 of 100 rounds. With 1,000 rounds and the same sigma, it chose arm 3 in 982 of 1,000 rounds. Its mean absolute estimation error over the three arms fell from about 0.282 to 0.156."

"Keeping 1,000 rounds but increasing noise to sigma 3 increased exploration of arms 1 and 2 (21 and 23 times), and mean absolute error rose to about 0.239. All three runs recommended arm 3, which is truly best because its mean is 3. An interesting result is that arm 1 happened to be estimated more accurately at higher noise; randomness doesn't mean every estimate is monotonically worse."

## 3:45–4:15 — Correctness and conclusion

"I tested that the counts add up to the requested round count, identical seeds reproduce results, only the selected arm updates, and invalid sigma or round counts are rejected. The main takeaway is that Thompson sampling applies to binary clicks with a Beta-Bernoulli model and also to continuous rewards with a Normal-Normal model, as long as the posterior update matches the reward distribution."

## Likely questions

**Why the Normal(0,25) prior?** The task requires it; prior standard deviation is 5, so it is relatively broad compared with true means 1–3.

**What is known sigma?** The standard deviation of the observation noise, not the uncertainty of the mean. Sigma is given to the algorithm, whereas each arm's true mean must be inferred.

**Why use posterior samples rather than posterior means for selection?** Sampling naturally trades off exploration and exploitation. Arms with uncertain means can be selected even when their current point estimate is lower.

**What gets updated?** Only the arm that was selected and actually generated a reward. Other arms' posterior parameters are unchanged for that round.

**Why can rewards be negative?** A Normal distribution has support over all real numbers, and the assignment explicitly forbids clipping/binarizing rewards.

**What is the difference between a posterior sampled theta and the final recommendation?** Theta is a random draw for deciding *one round*. The final recommendation maximizes the posterior **mean** after all rounds.

**Does the experiment give the same recommendation every time?** With the same seed and configuration yes, exactly; with different seeds there is random variation and a different recommendation is possible, particularly with fewer rounds or higher noise.
