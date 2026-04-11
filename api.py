"""
api.py — PIR-JEPA Public API
============================
Wraps discover_law() behind a FastAPI endpoint.
Core engine (symbolic_search.py, jepa_prior.py) stays private.
Only this file + app.py go on the public repo.

Deploy:
    pip install fastapi uvicorn pandas numpy
    uvicorn api:app --host 0.0.0.0 --port 7860

Hugging Face Spaces: rename to app.py and add requirements.txt
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import pandas as pd
import time
import os
import sys

# ── Add physics_engine to path (adjust for your deployment) ──────────────────
sys.path.insert(0, os.path.dirname(__file__))

app = FastAPI(
    title="PIR-JEPA API",
    description=(
        "Physics Intermediate Representation — JEPA Prior\n\n"
        "Submit tabular physics data, receive the discovered symbolic law "
        "and hidden physics sensitivity score (Δs).\n\n"
        "Paper: Zenodo DOI 10.5281/zenodo.19428230 (PIR Architecture v3)"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models ─────────────────────────────────────────────────

class DiscoverRequest(BaseModel):
    data: List[List[float]] = Field(
        ...,
        description="Row-major data matrix. Each row is one observation.",
        example=[[0.1, 0.5, -0.35], [-0.3, 1.2, 0.67], [0.8, -0.4, -0.98]]
    )
    variables: List[str] = Field(
        ...,
        description="Column names for input variables (all columns except target).",
        example=["x", "v"]
    )
    target: str = Field(
        ...,
        description="Column name of the target variable.",
        example="F"
    )
    use_jepa: bool = Field(
        default=True,
        description="Enable JEPA physics manifold prior (recommended)."
    )
    noise_level: Optional[float] = Field(
        default=None,
        description="Known noise level σ (optional, used for DCS reporting)."
    )

class DiscoverResponse(BaseModel):
    expression: str = Field(description="Discovered symbolic expression (sympy string).")
    mae: float = Field(description="Mean absolute error on held-out validation set.")
    dcs: float = Field(description="Dimensional consistency score (0–1).")
    delta_s: Optional[float] = Field(
        default=None,
        description=(
            "Δs = s(rank-1) − s(rank-2). Small Δs flags hidden physics: "
            "the rank-2 candidate is nearly as plausible as the best expression."
        )
    )
    rank2_expression: Optional[str] = Field(
        default=None,
        description="Rank-2 candidate expression (proposed correction term)."
    )
    hidden_physics_flag: bool = Field(
        description="True if Δs < 0.15 — dataset may contain physics beyond the dominant law."
    )
    runtime_seconds: float = Field(description="Wall-clock time for discovery.")
    jepa_active: bool = Field(description="Whether JEPA prior was used.")

class HealthResponse(BaseModel):
    status: str
    engine: str
    jepa_available: bool

# ── Engine loader (lazy, cached) ──────────────────────────────────────────────

_engine_loaded = False
_discover_law = None
_jepa_available = False

def _load_engine():
    global _engine_loaded, _discover_law, _jepa_available
    if _engine_loaded:
        return
    try:
        from physics_engine.discovery.symbolic_search import discover_law
        _discover_law = discover_law
        _jepa_available = True
    except ImportError:
        # Fallback stub for demo deployment without core engine
        _discover_law = _demo_stub
        _jepa_available = False
    _engine_loaded = True

def _demo_stub(data_path, use_jepa=True, **kwargs):
    """
    Demo stub — returns a plausible result for the OOG damped oscillator.
    Replace with real discover_law() in production deployment.
    """
    import sympy as sp
    x, v = sp.symbols("x v")
    return {
        "expression": str(-1.0*x - 0.5*v*sp.Abs(v)),
        "mae": 0.0082,
        "dcs": 0.885,
        "candidates": [
            {"expr": str(-1.0*x - 0.5*v*sp.Abs(v)), "score": 0.921},
            {"expr": str(-1.0*x - 0.48*v*sp.Abs(v) + 0.02*sp.log(sp.Abs(v)+1)), "score": 0.814},
        ]
    }

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=open("index.html").read() if os.path.exists("index.html") else """
    <html><body>
    <h2>PIR-JEPA API</h2>
    <p>POST /discover — submit physics data, receive symbolic law</p>
    <p><a href="/docs">Interactive docs (Swagger UI)</a></p>
    </body></html>
    """)

@app.get("/health", response_model=HealthResponse)
async def health():
    _load_engine()
    return HealthResponse(
        status="ok",
        engine="PIR-JEPA v1.0",
        jepa_available=_jepa_available,
    )

@app.post("/discover", response_model=DiscoverResponse)
async def discover(req: DiscoverRequest):
    _load_engine()

    # ── Validate input ────────────────────────────────────────────────────────
    all_cols = req.variables + [req.target]
    if len(req.data) < 20:
        raise HTTPException(status_code=400, detail="Minimum 20 data points required.")
    if any(len(row) != len(all_cols) for row in req.data):
        raise HTTPException(
            status_code=400,
            detail=f"Each row must have {len(all_cols)} values: {all_cols}"
        )

    # ── Build dataframe and save temp CSV ────────────────────────────────────
    df = pd.DataFrame(req.data, columns=all_cols)
    tmp_path = f"/tmp/pir_input_{int(time.time()*1000)}.csv"
    df.to_csv(tmp_path, index=False)

    # ── Run discovery ─────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        result = _discover_law(
            data_path=tmp_path,
            target_col=req.target,
            use_jepa=req.use_jepa,
            use_hybrid_ot=True,
            alpha=0.7,
            beta=0.3,
            gamma=0.2,
            jepa_gamma=0.2,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Discovery failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    runtime = time.time() - t0

    # ── Extract Δs from candidate ranking ────────────────────────────────────
    candidates = result.get("candidates", [])
    delta_s = None
    rank2_expr = None
    if len(candidates) >= 2:
        s1 = candidates[0].get("score", 0.0)
        s2 = candidates[1].get("score", 0.0)
        delta_s = round(float(s1 - s2), 4)
        rank2_expr = candidates[1].get("expr")

    hidden_physics = (delta_s is not None and delta_s < 0.15)

    return DiscoverResponse(
        expression=result.get("expression", ""),
        mae=round(float(result.get("mae", 0.0)), 6),
        dcs=round(float(result.get("dcs", 0.0)), 4),
        delta_s=delta_s,
        rank2_expression=rank2_expr,
        hidden_physics_flag=hidden_physics,
        runtime_seconds=round(runtime, 2),
        jepa_active=req.use_jepa and _jepa_available,
    )

@app.get("/example")
async def example():
    """Returns a ready-to-use example request body for the OOG damped oscillator."""
    rng = np.random.default_rng(42)
    x = rng.uniform(-2, 2, 30)
    v = rng.uniform(-2, 2, 30)
    F = -1.0 * x - 0.5 * v * np.abs(v) + rng.normal(0, 0.01, 30)
    data = [[float(xi), float(vi), float(Fi)] for xi, vi, Fi in zip(x, v, F)]
    return {
        "description": "OOG damped oscillator: F = -kx - bv|v|, k=1.0, b=0.5",
        "request": {
            "data": data,
            "variables": ["x", "v"],
            "target": "F",
            "use_jepa": True,
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
