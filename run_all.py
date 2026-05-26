"""
run_all.py — Single entry point for the full pipeline.
Usage: python run_all.py
"""

import subprocess
import sys
import os

DATA_DIR = "data"

# Prefer the venv Python if it exists alongside this script
_venv_python = os.path.join(os.path.dirname(__file__), ".venv", "bin", "python")
PYTHON = _venv_python if os.path.exists(_venv_python) else sys.executable


def run(cmd: list, label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=True)
    return result


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    run(
        [PYTHON, "etl_pipeline.py", "--output-dir", DATA_DIR],
        "Step 1/4 — ETL Pipeline"
    )
    run(
        [PYTHON, "feature_engineering.py", "--output-dir", DATA_DIR],
        "Step 2/4 — Feature Engineering"
    )
    _venv_jupyter = os.path.join(os.path.dirname(__file__), ".venv", "bin", "jupyter")
    JUPYTER = _venv_jupyter if os.path.exists(_venv_jupyter) else "jupyter"
    run(
        [
            JUPYTER, "nbconvert",
            "--to", "notebook",
            "--execute", "model_training.ipynb",
            "--output", "model_training_executed.ipynb",
            "--ExecutePreprocessor.timeout=300",
        ],
        "Step 3/4 — Model Training (Jupyter notebook)"
    )
    run(
        [PYTHON, "executive_summary_generator.py"],
        "Step 4/4 — Executive Summary PDF"
    )

    print("\n" + "="*60)
    print("  All steps complete!")
    print("="*60)
    print(f"\n  Outputs saved in:  ./{DATA_DIR}/")
    print(f"  Executive summary: ./{DATA_DIR}/executive_summary.pdf")
    print(f"  Tableau/PBI CSV:   ./{DATA_DIR}/dashboard_data.csv")
    print("\n  To launch the dashboard:")
    print("    streamlit run dashboard_app.py")
    print()
