# PIR — Physics Intermediate Representation

**Automated symbolic law discovery from noisy tabular data.**  
Classical pipeline: optimal transport scoring + dimensional soft scoring + template augmentation.

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
| PIR Architecture v3.1 | [10.5281/zenodo.19723561](https://doi.org/10.5281/zenodo.19723561) |
| PIR-Bench v3.1 (20 tasks, 100% DR) | [10.5281/zenodo.19723561](https://doi.org/10.5281/zenodo.19723561) |
| PIR-Bench v3.2 (gplearn head-to-head) | [10.5281/zenodo.20062682](https://doi.org/10.5281/zenodo.20062682) |
| PIR-Bench v3.3 (Feynman Tier A blind) | [10.5281/zenodo.20195566](https://doi.org/10.5281/zenodo.20195566) |
| PhysicsGPT v3 | [10.5281/zenodo.19428391](https://doi.org/10.5281/zenodo.19428391) |

**Withdrawn / do not cite:**
- PIR-JEPA SPARC (DOIs 19561908, 19602481, 19614469) — analysis ran on synthetic data, not real SPARC observations. Withdrawn.
- PIR-LIGO (DOIs 19586642, 19596888) — artifact audit found no Stage 1–3 results; Stage 4 all failed. Withdrawn.

## Key Results (classical PIR, artifact-supported)

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

### SRBench Feynman Tier A — blind protocol (v3.3)

5 seeds, held-out Feynman subset, no formula peeking:

| Metric | Value |
|---|---|
| Tasks stable | 7 / 44 (15.9%) |
| Mean recoveries | 7.6 / 44 (17.3%) |
| Protocol | Blind (no ground-truth access) |

## Architecture

PIR uses a classical two-stage pipeline:

**1. Template augmentation** — physics-informed expression families:
- Nonlinear damping: v·|v|, v², |v|
- Exponential / Boltzmann forms
- Logarithmic patterns
- Non-integer power-law cross terms

**2. Optimal transport scoring:**

```
s_total(c) = -(α·MSE + β·W₁) - γ·dim_penalty
α=0.7, β=0.3, γ=0.2  (soft penalty only; no hard dimensional rejection)
```

**3. Hidden physics detector (Δs):**

```
Δs = s(rank-1) − s(rank-2)
```

Small Δs < 0.15 flags the dataset as a hidden physics candidate — the rank-2
expression is a plausible symbolic correction to the dominant law, consistent
with dimensional analysis. Δs is real classical code with artifact support.

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
  author    = {Qazi Muhammad Hanif},
  title     = {{PIR}: Physics Intermediate Representation for
               Automated Symbolic Law Discovery},
  year      = {2026},
  doi       = {10.5281/zenodo.19723561},
  publisher = {Zenodo}
}
```

## Note on Code Availability

The core inference engine is in a private research repository.
This repository provides:

- The public API wrapper (`api.py`)
- Interactive demo UI (`index.html`)
- Benchmark datasets and evaluation scripts (`bench/`)
