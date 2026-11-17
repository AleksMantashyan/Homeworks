#!/usr/bin/env python3
"""Create an independent Streamlit frontend repo from the original + changes.

Usage:
  python3 build_project.py
  python3 build_project.py --source-path /path/to/DS223_Bayesian_AB_Testing/app/front
  python3 build_project.py --output /path/to/my-new-project

The script copies the actual upstream source from app/front/, preserving its
pyproject.toml and uv.lock; it then adds the new Gaussian files and docs.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO = "https://github.com/hovhannisyan91/DS223_Bayesian_AB_Testing.git"
ARCHIVE = "https://codeload.github.com/hovhannisyan91/DS223_Bayesian_AB_Testing/zip/refs/heads/main"
EXCLUDED = {".venv", "venv", ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
ROOT = Path(__file__).resolve().parent


def download_source(temp_dir: Path) -> Path:
    """Download the original project source from upstream, fallback to git."""
    zip_path = temp_dir / "upstream.zip"
    try:
        print(f"Downloading original source from {ARCHIVE} ...")
        with urllib.request.urlopen(ARCHIVE, timeout=40) as response:
            zip_path.write_bytes(response.read())
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(temp_dir / "archive")
        candidate = next((temp_dir / "archive").glob("*/app/front"), None)
        if candidate is not None:
            return candidate
        raise RuntimeError("Could not locate app/front in the downloaded archive.")
    except Exception as error:
        print(f"Archive download failed ({error}). Trying git clone instead...")
        git_path = temp_dir / "upstream_git"
        subprocess.run(["git", "clone", "--depth", "1", REPO, str(git_path)], check=True)
        return git_path / "app" / "front"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path.cwd() / "DS223_Gaussian_Bandit_Project")
    parser.add_argument(
        "--source-path", type=Path, default=None,
        help="Path to an existing local clone's app/front (skips downloading).",
    )
    args = parser.parse_args()
    out = args.output.expanduser().resolve()
    if out.exists():
        parser.error(f"Destination already exists and will not be overwritten: {out}")

    with tempfile.TemporaryDirectory(prefix="ds223_original_") as temp:
        source = args.source_path.expanduser().resolve() if args.source_path else download_source(Path(temp))
        required_source = ["main.py", "bandit_utils.py", "pyproject.toml", "uv.lock", "pages/Create_Carousels.py", "pages/Interaction.py", "pages/Analytics.py"]
        missing = [name for name in required_source if not (source / name).is_file()]
        if missing:
            parser.error(f"Original source is missing these required files: {', '.join(missing)} (searched {source})")
        shutil.copytree(source, out, ignore=shutil.ignore_patterns(*EXCLUDED))

    overlays = ROOT / "overrides"
    for item in overlays.rglob("*"):
        if not item.is_file() or any(part in EXCLUDED for part in item.relative_to(overlays).parts):
            continue
        relative = item.relative_to(overlays)
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)

    # `uv sync --locked` remains compatible because original dependency pins
    # and original lockfile are retained without dependency modifications.
    required_final = ["main.py", "bandit_utils.py", "gaussian_bandit.py", "pyproject.toml", "uv.lock", "README.md", "pages/Analytics.py", "pages/Gaussian_Bandit.py", "experiment_results.csv"]
    if any(not (out / filename).is_file() for filename in required_final):
        raise RuntimeError("Assembly incomplete: some required files are missing.")

    print("\nProject assembled successfully:", out)
    print("Original three pages, pyproject.toml and uv.lock are included.")
    print("\nNext commands:")
    print(f'  cd "{out}"')
    print("  uv sync --locked")
    print("  uv run streamlit run main.py")
    print("\nWhen ready: create an empty GitHub repository and push this folder to main.")


if __name__ == "__main__":
    main()
