# Franka Mass-Matrix Grammar Analysis

Reproducibility artifacts for the paper *Principled Grammar Specification for Symbolic Regression of Robot Mass-Matrix Dynamics* (Hanif, 2026).

## Contents

- `franka_M*.csv` (28 files) — Mass-matrix diagonal datasets from 2500 PyBullet configurations on the Franka Panda. 7 diagonals × 4 conditions.
- `franka_mass_manifest.json` — Dataset specification.
- `M11_*_analysis.json` (3 files) — Lasso analysis outputs backing Tables 2, 3, 5, 6.
- `structural_analysis.json` — Backing Section 5.
- `gen_franka_mass_data.py` — Data generation script (PyBullet).
- `m11_*.py` (3 files) — Lasso analysis scripts.

## Reproducing the paper's tables

    python gen_franka_mass_data.py
    python m11_triple_analysis.py
    python m11_sin2q_analysis.py
    python m11_final_grammar.py
