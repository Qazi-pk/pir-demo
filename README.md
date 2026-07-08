# PIR — Physics Intermediate Representation

**Automated symbolic law discovery from noisy tabular data.**  
Classical pipeline: monomial-basis log-linearization + optimal transport scoring + dimensional soft scoring + template augmentation.

Submit tabular physics data → receive the discovered symbolic law + hidden physics
sensitivity score (Δs).

> **Note (2026-05):** This repository previously advertised results attributed to
> JEPA and Langevin diffusion components. A 2026-05 artifact audit found those
> components were never wired in or executed in any published run. The "over-diffusion"
> ablation and T=500 optimum have no artifact support and have been retracted.
> All results below come from the **classical PIR pipeline only**.

## Live Demo

🔬 **[Try it on Hugging Face Spaces](https://huggingface.co/spaces/Qazihanif/pir-jepa)**

## Papers (current, citable)

| Paper | DOI |
|---|---|
| PIR: Physics Intermediate Representation (v3.4) | [10.5281/zenodo.21231641](https://doi.org/10.5281/zenodo.21231641) |
| PIR Architecture v3.1 | [10.5281/zenodo.19723561](https://doi.org/10.5281/zenodo.19723561) |
| PIR-Bench v3.2 (gplearn head-to-head) | [10.5281/zenodo.20062682](https://doi.org/10.5281/zenodo.20062682) |
| PIR-Bench v3.3 (Feynman Tier A blind) | [10.5281/zenodo.20195566](https://doi.org/10.5281/zenodo.20195566) |
| PhysicsGPT v3 | [10.5281/zenodo.19428391](https://doi.org/10.5281/zenodo.19428391) |
| Franka mass-matrix grammar | [10.5281/zenodo.20616114](https://doi.org/10.5281/zenodo.20616114) |

**External validation:** Submitted to the SRBench living benchmark — [cavalab/srbench PR #210](https://github.com/cavalab/srbench/pull/210).

**Withdrawn / do not cite:**
- PIR-JEPA SPARC (DOIs 19561908, 19602481, 19614469) — analysis ran on synthetic data, not real SPARC observations. Withdrawn.
- PIR-LIGO (DOIs 19586642, 19596888) — artifact audit found no Stage 1–3 results; Stage 4 all failed. Withdrawn.

## Key Results (classical PIR, artifact-supported)

### SRBench Feynman Tier A — blind protocol (v3.4)

Strict blind protocol, 5 seeds, no formula peeking, symbolic-ratio classifier:

| Metric | Value |
|---|---|
| Exact recovery | **12 / 44 EXACT** (symbolic ratio == 1) |
| Correct form (secondary) | +12 / 44 FORM_NUMERIC (transcendental constant folded as decimal) |
| Seed wobble | Zero |
| Protocol | Blind (no ground-truth access at configuration time) |

FORM_NUMERIC is reported as a transparent secondary figure and never summed into the
primary count. Prior figures of "27.3%" / "13/44" are retracted (formula-peeking); the
honest blind headline is **12/44 EXACT**.

### PIR-Bench v3.1 — 20 physics tasks

5 seeds, n=200, noise=0.01, hybrid OT (α=0.7, β=0.3), dimensional soft scoring (γ=0.2):

| Task | DR% | DCS |
|---|---|---|
| newton | 100% | 0.992 |
| kepler_third_law | 100% | — |
| inverse_square | 100% | 0.974 |
| gravity | 100% | 0.806 |
| pendulum | 100% | 0.985 |
| orbit_ax | 100% | — |
| orbit_ay | 100% | — |
| planar_robot_fk | 100% | — |
| hamiltonian | 100% | 1.000 |
| *(all 20 tasks)* | **100%** | **mean 0.840** |

## Architecture

PIR uses a classical pipeline:

**1. F3 log-linearization gate** — recovers power-law monomials via OLS in log-space
(R² ≥ 0.99 acceptance). This is the primary recovery mechanism for pure multiplicative laws.

**2. Template augmentation** — physics-informed expression families:
- Nonlinear damping: v·|v|, v², |v|
- Exponential / Boltzmann forms
- Logarithmic patterns
- Non-integer power-law cross terms

**3. Optimal transport scoring:**

```
s_total(c) = -(α·MSE + β·W₁) - γ·dim_penalty
α=0.7, β=0.3, γ=0.2  (soft penalty only; no hard dimensional rejection)
```

**4. Hidden physics detector (Δs):**

```
Δs = s(rank-1) − s(rank-2)
```

Small Δs < 0.15 flags the dataset as a hidden physics candidate — the rank-2
expression is a plausible symbolic correction to the dominant law, consistent
with dimensional analysis. Δs is real classical code with artifact support; it is
a candidate-margin diagnostic, not a validated hidden-physics detector.

## Quick Start

```
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 7860
```

Open `http://localhost:7860/docs` for the interactive Swagger UI.

## API

### POST /discover

```json
{
  "data": [[0.1, 0.5, -0.35], [-0.3, 1.2, 0.67], ...],
  "variables": ["x", "v"],
  "target": "F"
}
```

**Response:**

```json
{
  "expression": "-1.0*x - 0.5*v*Abs(v)",
  "mae": 0.0082,
  "dcs": 0.994,
  "delta_s": 0.107,
  "rank2_expression": "-1.0*x - 0.48*v*Abs(v) + 0.02*log(Abs(v)+1)",
  "hidden_physics_flag": true,
  "runtime_seconds": 1.3
}
```

### GET /health

Returns engine status and version.

## Benchmark Datasets

```bash
python bench/run_discovery_benchmark.py \
    --experiments oog_damped_oscillator \
    --noise-levels 0.01 --dataset-sizes 200 \
    --repeats 5 --hybrid-ot
```

## Citation

```bibtex
@misc{hanif2026pir,
  author    = {Hanif, Qazi},
  title     = {{PIR}: Physics Intermediate Representation for
               Automated Discovery of Physical Laws},
  year      = {2026},
  doi       = {10.5281/zenodo.21231641},
  publisher = {Zenodo},
  note      = {ORCID: 0009-0003-2818-5449}
}
```

## Note on Code Availability

The classical inference engine is publicly available at
[github.com/Qazi-pk/physics-engine](https://github.com/Qazi-pk/physics-engine) (MIT, tag v3.4.1).

This repository provides:

- The public API wrapper (`api.py`)
- Interactive demo UI (`index.html`)
- Benchmark datasets and evaluation scripts (`bench/`)
