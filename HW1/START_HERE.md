# Start here — Gaussian Bandit assignment kit

This kit contains the **finished new Gaussian algorithm and page**, the three **computed experiments**, a **README** and **presentation notes**. The original app must be copied into your own repository. To make sure the exact original `main.py`, pages, `bandit_utils.py`, `pyproject.toml`, and especially **`uv.lock`** are preserved, the assembly script downloads the original GitHub source when you run it on your computer.

The kit is a *project assembler*, not a complete deployed GitHub repository by itself. **Do not upload the `overrides/` folder as your final repo.** Instead, follow these steps.

## A. Create the complete project on your computer

1. Unzip this kit and open Terminal in the unzipped `DS223_Gaussian_Bandit_Kit` folder.
2. Run:

   ```bash
   python3 build_project.py
   ```

   The script downloads the professor's GitHub repository, copies `app/front/` into a **new independent** folder named `DS223_Gaussian_Bandit_Project`, and installs the extension files over the top. It doesn't modify or push to the professor's repository.

   If you already cloned the original repository and have trouble downloading, run:

   ```bash
   python3 build_project.py --source-path /path/to/DS223_Bayesian_AB_Testing/app/front
   ```

3. Move into the assembled frontend folder:

   ```bash
   cd DS223_Gaussian_Bandit_Project
   ```

4. [Install uv](https://docs.astral.sh/uv/getting-started/installation/) if necessary, then run:

   ```bash
   uv sync --locked
   uv run streamlit run main.py
   ```

5. In the browser, make sure the original **Create Carousels**, **Interaction**, and **Analytics** pages still work. Then run **Gaussian Bandit** with `mu=(1, 2, 3)`, `sigma=1`, `rounds=100`, `seed=42`.

## B. Confirm the three experiment results

Inside your assembled project:

```bash
uv run python run_experiments.py
uv run --with pytest pytest -q tests/test_gaussian_bandit.py
```

The README already contains the experiment results computed with seed 42, and `experiment_results.csv` is included.

## C. Create your GitHub repository and push

Make a **new empty GitHub repository** (for example, `DS223-Gaussian-Bandit`). Do **not** initialize it with a README, license or `.gitignore`, because those are already in your assembled folder. Copy its repository URL, then run inside `DS223_Gaussian_Bandit_Project`:

```bash
git init -b main
git add .
git commit -m "Implement Gaussian Thompson sampling"
git remote add origin https://github.com/YOUR_USERNAME/DS223-Gaussian-Bandit.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username and replace the repository name if different. If Git complains about your commit identity, set `git config --global user.name "..."` and `git config --global user.email "..."` with your own details.

## D. Deploy and submit

At https://share.streamlit.io/ create a new app from your GitHub repository. Set:

- Repository: **your** newly created repository
- Branch: `main`
- Entrypoint: `main.py`
- Python: **3.12** (Advanced settings)

Verify that Gaussian Bandit runs in the deployed app. Update the two placeholder URLs in `README.md` to your real GitHub and Streamlit links, then commit/push that README change.

**Submit:** the two links (GitHub and deployed Streamlit). Keep `PRESENTATION_NOTES.md` ready for the 3–5 minute presentation.

## Files included in this kit

- `build_project.py` — assembles your independent project from the exact upstream frontend.
- `overrides/main.py` — retains the original pages; registers Gaussian Bandit.
- `overrides/gaussian_bandit.py` — independently testable Normal-Normal Thompson sampling implementation.
- `overrides/pages/Gaussian_Bandit.py` — interactive page.
- `overrides/tests/test_gaussian_bandit.py` — numerical correctness tests.
- `overrides/run_experiments.py`, `overrides/experiment_results.csv` — reproducible experiment outputs.
- `overrides/README.md` — assignment writeup, formulas, evidence, observations, credit, setup and deploy instructions.
- `overrides/PRESENTATION_NOTES.md` — a 3–5 minute speaking outline.

**Verification already done here:** the numerical tests pass and the experiment results have been computed. **Still to verify on your machine:** installation with the original locked dependencies, the original pages and new page working interactively, the GitHub push, and deployed Streamlit operation. Those last parts require your local/GitHub environment.
