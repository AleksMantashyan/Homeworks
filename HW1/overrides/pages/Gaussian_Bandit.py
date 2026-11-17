"""Interactive Gaussian Thompson Sampling experiment."""
import numpy as np
import pandas as pd
import streamlit as st

from gaussian_bandit import PRIOR_VARIANCE, run_gaussian_bandit

st.title("Gaussian Bandit")
st.caption(
    "Thompson sampling with three alternatives, continuous Gaussian rewards, "
    "known observation noise, and a Normal(0, 25) prior for each unknown mean."
)

with st.form("gaussian_bandit_form"):
    st.markdown("### Simulation parameters")
    a, b, c = st.columns(3)
    with a:
        mu1 = st.number_input("Alternative 1: true mean", value=1.0, step=0.5)
    with b:
        mu2 = st.number_input("Alternative 2: true mean", value=2.0, step=0.5)
    with c:
        mu3 = st.number_input("Alternative 3: true mean", value=3.0, step=0.5)

    noise_col, rounds_col, seed_col = st.columns(3)
    with noise_col:
        sigma = st.number_input("Known noise standard deviation (σ)", value=1.0, step=0.5)
    with rounds_col:
        rounds = st.number_input("Number of rounds", value=100, step=1, format="%d")
    with seed_col:
        seed = st.number_input("Random seed", value=42, step=1, format="%d")

    run_button = st.form_submit_button("Run Gaussian Bandit", type="primary")

if run_button:
    if not np.isfinite(sigma) or sigma <= 0:
        st.error("σ must be a positive, finite number.")
    elif not isinstance(rounds, (int, np.integer)) or rounds <= 0:
        st.error("Number of rounds must be a positive integer.")
    else:
        try:
            st.session_state["gaussian_bandit_result"] = run_gaussian_bandit(
                true_means=(mu1, mu2, mu3),
                sigma=float(sigma),
                rounds=int(rounds),
                seed=int(seed),
            )
        except ValueError as exc:
            st.error(str(exc))

result = st.session_state.get("gaussian_bandit_result")
if result is None:
    st.info("Choose parameters and press **Run Gaussian Bandit** to see results.")
else:
    st.divider()
    st.subheader("Simulation results")
    st.caption(
        f"Last completed run: σ={result.noise_sigma:g}, "
        f"rounds={result.rounds:,}, seed={result.seed}. "
        "The displayed results persist across page navigation."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total observed reward", f"{result.total_reward:,.3f}")
    col2.metric("Final recommendation", f"Alternative {result.recommendation}")
    col3.metric("True best alternative (simulation)", f"Alternative {result.true_best}")

    summary = pd.DataFrame(
        {
            "Alternative": ["Alternative 1", "Alternative 2", "Alternative 3"],
            "True mean": result.true_means,
            "Posterior estimated mean": result.posterior_means,
            "Posterior std. dev.": np.sqrt(result.posterior_variances),
            "Selection count": result.selection_counts,
        }
    )
    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            "True mean": st.column_config.NumberColumn(format="%.3f"),
            "Posterior estimated mean": st.column_config.NumberColumn(format="%.3f"),
            "Posterior std. dev.": st.column_config.NumberColumn(format="%.3f"),
            "Selection count": st.column_config.NumberColumn(format="%d"),
        },
    )
    st.write(f"**Selections:** {int(result.selection_counts.sum()):,} / {result.rounds:,} rounds")
    st.write(
        "**Recommendation rule:** alternative with the highest posterior mean "
        "after the final round (not the largest sampled value or true mean)."
    )

    st.subheader("Learning over time")
    history_df = pd.DataFrame(
        result.posterior_mean_history,
        columns=["Alternative 1", "Alternative 2", "Alternative 3"],
    )
    history_df.index = np.arange(1, result.rounds + 1)
    history_df.index.name = "Round"
    st.line_chart(history_df, x_label="Round", y_label="Posterior mean")

    cumulative_df = pd.DataFrame(
        {"Round": np.arange(1, result.rounds + 1),
         "Cumulative observed reward": np.cumsum(result.rewards)}
    ).set_index("Round")
    st.line_chart(cumulative_df, x_label="Round", y_label="Observed reward (cumulative)")

    with st.expander("Show raw reward history (including negative rewards)"):
        history = pd.DataFrame(
            {
                "Round": np.arange(1, result.rounds + 1),
                "Chosen alternative": result.selected_alternatives,
                "Continuous reward": result.rewards,
                "Cumulative reward": np.cumsum(result.rewards),
            }
        )
        st.dataframe(history, use_container_width=True, hide_index=True)
        st.download_button(
            "Download history as CSV",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name=f"gaussian_bandit_seed{result.seed}_{result.rounds}rounds.csv",
            mime="text/csv",
        )

    st.info(
        f"Each arm starts with prior mean 0 and prior variance {PRIOR_VARIANCE:g}. "
        "Every round, the algorithm samples from all three *current learned posteriors*, "
        "selects the largest sample, obtains one Gaussian reward, and updates only that arm."
    )
