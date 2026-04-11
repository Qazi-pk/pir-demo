# pir-jepa
PIR-JEPA: Physics Law Discovery API and Benchmark

**Physics Intermediate Representation** with Joint Embedding Predictive Architecture prior.

Submit tabular physics data → receive the discovered symbolic law + hidden physics sensitivity score.

## Live Demo

🔬 **[Try it on Hugging Face Spaces](https://huggingface.co/spaces/qmhanif/pir-jepa)**

## Papers

| Paper | DOI |
|-------|-----|
| PIR Architecture v3 | [10.5281/zenodo.19428230](https://doi.org/10.5281/zenodo.19428230) |
| PIR-Bench v3.1 | [10.5281/zenodo.19130521](https://doi.org/10.5281/zenodo.19130521) |
| PhysicsGPT v3 | [10.5281/zenodo.19130163](https://doi.org/10.5281/zenodo.19130163) |
| PIR-JEPA | [10.5281/zenodo.19428230](https://doi.org/10.5281/zenodo.19428230) |

## Quick Start

```bash
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
  "target": "F",
  "use_jepa": true
}
```

**Response:**
```json
{
  "expression": "-1.0*x - 0.5*v*Abs(v)",
  "mae": 0.008,
  "dcs": 0.885,
  "delta_s": 0.107,
  "rank2_expression": "-1.0*x - 0.48*v*Abs(v) + 0.02*log(Abs(v)+1)",
  "hidden_physics_flag": true,
  "runtime_seconds": 1.3,
  "jepa_active": true
}
```

**`delta_s`** (Δs): scoring margin between rank-1 and rank-2 candidates.
Small Δs < 0.15 flags the dataset as a hidden physics candidate —
the rank-2 expression is the most plausible symbolic correction
to the dominant law, consistent with dimensional analysis.

### GET /example

Returns a ready-to-use request body for the out-of-grammar benchmark task
`oog_damped_oscillator` (F = −kx − bv|v|).

### GET /health

Returns engine status and JEPA availability.

## Benchmark Datasets

All PIR-Bench benchmark datasets and evaluation scripts are in `/bench/`.

```bash
python bench/run_discovery_benchmark.py \
    --experiments oog_damped_oscillator \
    --noise-levels 0.01 \
    --dataset-sizes 200 \
    --repeats 5 \
    --hybrid-ot --use-jepa
```

## Results (5-seed ablation, April 2026)

| Task | Baseline DR | PIR-JEPA DR | MAE reduction |
|------|-------------|-------------|---------------|
| newton | 100% | 100% | — |
| kepler_third_law | 100% | 100% | — |
| gravity | 100% | 100% | — |
| pendulum | 100% | 100% | — |
| oog_damped_oscillator | 0% | **80%** | **100×** |

## Citation

```bibtex
@misc{hanif2026pirjepa,
  author    = {Muhammad Hanif},
  title     = {{PIR-JEPA}: Joint Embedding Predictive Architecture
               as a Physics Manifold Prior for
               Out-of-Grammar Symbolic Law Discovery},
  year      = {2026},
  doi       = {10.5281/zenodo.19428230},
  publisher = {Zenodo}
}
```

## Note on Code Availability

The core inference engine is proprietary. This repository provides:
- The public API wrapper (`api.py`)
- Benchmark datasets and evaluation scripts (`bench/`)
- A hosted demo for reproducibility verification

This satisfies arXiv reproducibility standards. For collaboration or
licensing enquiries: qmhanif70@gmail.com
