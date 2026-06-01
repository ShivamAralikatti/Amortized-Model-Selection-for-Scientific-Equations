#!/usr/bin/env python3
"""
cv_threeway_pk.py — LOOCV Three-Way Comparison for the Theophylline PK Dataset
================================================================================

This is the PHARMACOKINETICS analog of cv_threeway.py. It performs the same
leave-one-out three-way comparison (no-input MLP vs TRF cold vs TRF warm)
for two competing PK models:

    one_compartment  — 3 parameters: V, ka, ke
    two_compartment  — 4 parameters: ka, alpha, extra, A_frac


TWO MLP INPUT VARIANTS (the key PK-specific design choice)
----------------------------------------------------------
Rheology and PK have a real structural asymmetry that must be acknowledged.
In rheology, "material identity" is a meaningful test-time input: a new
specimen comes from a known material family (e.g., another piece of CuNiSi),
so the one-hot material code carries learned weight that transfers. In PK,
"subject identity" is NOT meaningful at test time in the same way: a new
patient is a genuinely new human, not "another instance of patient 7."

To preserve the comparison's integrity, this script runs the SAME LOOCV
three-way pipeline under two MLP input variants. Pick one per run via
the --no-subject-id flag.

    --- DEFAULT (subject-id variant) ---
    MLP input = [one-hot subject ID, normalized total dose σ0]
    Interpretation: the rheology-parallel upper bound. Tells us what
    amortized inference COULD achieve if categorical replicate-structured
    population data existed for PK. Not a valid test-time predictor for a
    truly new patient.

    --- --no-subject-id (population variant) ---
    MLP input = [normalized body weight, normalized total dose σ0]
    Interpretation: a clinically realistic predictor. Generalizes to a
    truly new patient via continuous covariates (weight, dose). This is
    the variant a clinician could actually deploy.

Both variants share architecture, optimizer, training procedure, fraction
grid, and TRF baseline. They differ only in input encoding. The expected
empirical pattern: the subject-id variant produces lower test MSE
(because it can memorize per-subject parameters) while the population
variant tests whether the equation's parameterization supports
amortization from continuous covariates alone. The gap between the two
quantifies the value of categorical population structure for amortized
inference, and is reported as a substantive cross-domain finding.


KEY INVARIANT (same as rheology, unchanged)
-------------------------------------------
The MLP receives NO observations from the test curve under either variant.
Its MSE is therefore CONSTANT across observation fractions. The crossover
with TRF emerges because TRF improves with more observations while the
MLP stays flat.


Fraction grid
-------------
With only 11 time points per subject, "fraction" maps to n_obs as
ceil(fraction * 11). The default fractions correspond to:
    0.10 -> n_obs=2  (extreme low-data)
    0.20 -> n_obs=3
    0.30 -> n_obs=4
    0.45 -> n_obs=5
    0.55 -> n_obs=7
    0.75 -> n_obs=9
    1.00 -> n_obs=11 (all points)
Picked to give roughly even spacing on a count basis.


Usage
-----
    # RECOMMENDED: run every variant x seed in one command
    python cv_threeway_pk.py --csv theoph.csv --all-variants \\
        --output-dir results_pk_all

    This runs three configurations --
        subjid_sigma0        : one-hot subject ID + dose      (upper bound)
        population_sigma0    : weight + dose                  (clinically real)
        population_nosigma0  : weight only                    (sigma0 ablation)
    -- each across 5 seeds (42-46 by default), then aggregates across seeds.
    Master outputs: MASTER_SUMMARY.txt, MASTER_all_folds_raw.csv, and per-config
    SEED_AGGREGATED.csv / SEED_EQUATION_WINS.csv.

    # Change the seed list
    python cv_threeway_pk.py --csv theoph.csv --all-variants \\
        --seeds 1,2,3 --output-dir results_pk_all

    # SINGLE RUN (one config) -- still supported for quick checks
    python cv_threeway_pk.py --csv theoph.csv \\
        --output-dir results_pk_subjid                 # subject-id variant
    python cv_threeway_pk.py --csv theoph.csv --no-subject-id \\
        --output-dir results_pk_population              # population variant
    python cv_threeway_pk.py --csv theoph.csv --no-subject-id --no-sigma0 \\
        --output-dir results_pk_pop_nosigma0            # sigma0 ablation

    # One model only (faster)
    python cv_threeway_pk.py --csv theoph.csv --model one_compartment

TRF restarts default to 50 (was 10 in earlier versions): see --trf-restarts.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import least_squares

import pk_pipeline as ppl
import pk_physics as pkp

EPS         = 1e-8
UNKNOWN_MAT = "__UNKNOWN__"

VALID_MODELS = ["one_compartment", "two_compartment", "both"]


# ===========================================================================
#  Utilities (parallel to cv_threeway.py)
# ===========================================================================

def set_seed(seed: int) -> None:
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def inverse_softplus(y: float) -> float:
    y = max(float(y), EPS)
    return y if y > 30 else math.log(math.expm1(y))


def parse_fractions(text: str) -> List[float]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part: continue
        is_pct = part.endswith("%")
        v = float(part[:-1]) if is_pct else float(part)
        if is_pct: v /= 100.0
        elif v > 1.0: v /= 100.0
        if not 0.0 < v <= 1.0:
            raise ValueError(f"Invalid fraction: {part!r}")
        out.append(v)
    if not out:
        raise ValueError("Provide at least one fraction.")
    return sorted(set(out))


# ===========================================================================
#  Curve attribute access (PKCurve exposes the same attribute names)
# ===========================================================================

def get_time(c):     return np.asarray(c.time, dtype=np.float64)
def get_stress(c):   return np.asarray(c.stress, dtype=np.float64)   # = concentration
def get_material(c): return c.material
def get_name(c):     return c.name
def get_sigma0(c):   return float(c.sigma0)                          # = total dose mg
def get_weight_kg(c):return float(c.weight_kg)                       # body weight, kg
def get_temperature_K(c): return float(c.temperature_K)              # held constant, ignored


# ===========================================================================
#  Material (= subject) encoding
# ===========================================================================

def build_material_map(curves: List) -> Dict[str, int]:
    mats = sorted({get_material(c) for c in curves})
    mapping = {m: i for i, m in enumerate(mats)}
    mapping[UNKNOWN_MAT] = len(mapping)
    return mapping


def one_hot(i: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32); v[i] = 1.0; return v


def make_input(curve, material_map: Dict[str,int],
               sigma0_mean: float, sigma0_std: float,
               weight_mean: float, weight_std: float,
               include_sigma0: bool,
               include_subject_id: bool) -> np.ndarray:
    """
    Build the MLP input vector for one curve.

    Two variants (chosen via include_subject_id):
      include_subject_id=True   -> [one_hot_subject_id, sigma0_norm?]
        Rheology-parallel upper bound. Encodes per-subject identity.
      include_subject_id=False  -> [weight_norm, sigma0_norm?]
        Population variant. Pure continuous-covariate predictor. Generalises
        to a truly new patient via weight and dose.

    The include_sigma0 flag is honoured under BOTH variants. Default usage
    is include_sigma0=True. Disabling it would leave the population variant
    with weight alone (1 input) and the subject-id variant with the
    one-hot alone — both available for ablations.
    """
    if include_subject_id:
        n_mat = len(material_map)
        mid   = int(material_map.get(get_material(curve),
                                     material_map[UNKNOWN_MAT]))
        oh    = one_hot(mid, n_mat)
        if include_sigma0:
            s0_norm = (get_sigma0(curve) - sigma0_mean) / max(sigma0_std, EPS)
            return np.concatenate([oh, [s0_norm]]).astype(np.float32)
        return oh

    # Population variant — continuous covariates only
    w_norm = (get_weight_kg(curve) - weight_mean) / max(weight_std, EPS)
    feats  = [w_norm]
    if include_sigma0:
        s0_norm = (get_sigma0(curve) - sigma0_mean) / max(sigma0_std, EPS)
        feats.append(s0_norm)
    return np.array(feats, dtype=np.float32)


def input_dim_for(n_materials: int,
                  include_sigma0: bool,
                  include_subject_id: bool) -> int:
    """Compute the MLP input dimension consistent with make_input's variants."""
    if include_subject_id:
        return n_materials + (1 if include_sigma0 else 0)
    # Population variant: always at least weight; optionally + sigma0
    return 1 + (1 if include_sigma0 else 0)


# ===========================================================================
#  No-input MLP
# ===========================================================================

class NoInputMLP(nn.Module):
    def __init__(self, input_dim: int, n_outputs: int,
                 output_bias_init: Tuple[float,...]):
        super().__init__()
        assert len(output_bias_init) == n_outputs
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 32),        nn.ReLU(),
            nn.Linear(32, n_outputs),
        )
        with torch.no_grad():
            bias = torch.tensor([inverse_softplus(v) for v in output_bias_init],
                                 dtype=torch.float32)
            self.net[-1].bias.copy_(bias)
            self.net[-1].weight.zero_()

    def forward(self, x):
        return self.net(x)


def predict_curve_torch(model: NoInputMLP, x: torch.Tensor,
                        sigma0: torch.Tensor, t: torch.Tensor,
                        kind: str) -> torch.Tensor:
    raw = model(x)
    if kind == "one_compartment":
        return pkp.one_compartment_torch(sigma0, raw, t)
    return pkp.two_compartment_torch(sigma0, raw, t)


def predict_numpy(params: np.ndarray, sigma0: float, t: np.ndarray,
                  kind: str) -> np.ndarray:
    if kind == "one_compartment":
        return pkp.one_compartment_numpy(params, sigma0, t)
    return pkp.two_compartment_numpy(params, sigma0, t)


# ===========================================================================
#  Training (full curve, no fraction parameter — same invariant as rheology)
# ===========================================================================

def train_model(train_curves: List, material_map: Dict[str,int],
                kind: str, include_sigma0: bool, include_subject_id: bool,
                epochs: int, lr: float, wd: float, device: torch.device,
                sigma0_mean: float, sigma0_std: float,
                weight_mean: float, weight_std: float,
                log_every: int = 100) -> NoInputMLP:
    """
    Train the no-input MLP on the FULL time-series of every training subject.
    No fraction parameter — the MLP must see full curves so its loss has a
    long-horizon signal. See cv_threeway.py for the full rationale.

    The input variant (subject-id vs population) is fixed by
    include_subject_id and propagated through make_input + input_dim_for.
    """
    n_out = pkp.n_outputs(kind)
    bias  = pkp.initial_bias(kind)
    in_dim = input_dim_for(len(material_map), include_sigma0, include_subject_id)
    model = NoInputMLP(in_dim, n_out, bias).to(device)
    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.5)

    cached = []
    for c in train_curves:
        cached.append({
            "x":      torch.tensor(make_input(c, material_map,
                                              sigma0_mean, sigma0_std,
                                              weight_mean, weight_std,
                                              include_sigma0,
                                              include_subject_id),
                                   device=device),
            "t":      torch.tensor(get_time(c).astype(np.float32),   device=device),
            "y":      torch.tensor(get_stress(c).astype(np.float32), device=device),
            "sigma0": torch.tensor(get_sigma0(c), dtype=torch.float32, device=device),
        })

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        losses = [
            torch.mean((predict_curve_torch(model, c["x"], c["sigma0"],
                                            c["t"], kind) - c["y"])**2)
            for c in cached
        ]
        loss = torch.stack(losses).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step(loss.item())
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(f"      epoch {epoch:4d}/{epochs}  train MSE = {loss.item():.4f}")

    return model


@torch.no_grad()
def evaluate_mlp(model: NoInputMLP, test_curves: List,
                 material_map: Dict[str,int], kind: str,
                 include_sigma0: bool, include_subject_id: bool,
                 device: torch.device,
                 sigma0_mean: float, sigma0_std: float,
                 weight_mean: float, weight_std: float) -> List[dict]:
    model.eval()
    rows = []
    for c in test_curves:
        x    = torch.tensor(make_input(c, material_map,
                                       sigma0_mean, sigma0_std,
                                       weight_mean, weight_std,
                                       include_sigma0, include_subject_id),
                            device=device)
        t    = torch.tensor(get_time(c).astype(np.float32),   device=device)
        y    = torch.tensor(get_stress(c).astype(np.float32), device=device)
        s0   = torch.tensor(get_sigma0(c), dtype=torch.float32, device=device)
        pred = predict_curve_torch(model, x, s0, t, kind)
        mse  = float(torch.mean((pred - y)**2).item())
        rows.append({"material": get_material(c), "file": get_name(c),
                     "sigma0": get_sigma0(c), "mse_mlp": mse})
    return rows


@torch.no_grad()
def mlp_params_numpy(model: NoInputMLP, curve, material_map: Dict[str,int],
                     sigma0_mean: float, sigma0_std: float,
                     weight_mean: float, weight_std: float,
                     include_sigma0: bool, include_subject_id: bool,
                     device: torch.device, kind: str) -> np.ndarray:
    """Return MLP-predicted parameters in physical units.

    For one_compartment: returns (V, ka, ke) after softplus.
    For two_compartment: returns (ka, alpha, extra, A_frac_raw). Note that
    A_frac_raw is left in its raw form so TRF can search in the
    unconstrained sigmoid pre-image — matching the parameterization of
    pk_physics.two_compartment_numpy.
    """
    model.eval()
    x   = torch.tensor(make_input(curve, material_map,
                                   sigma0_mean, sigma0_std,
                                   weight_mean, weight_std,
                                   include_sigma0, include_subject_id),
                       device=device)
    raw = model(x).cpu().numpy().astype(np.float64)

    if kind == "one_compartment":
        # All three outputs go through softplus
        pos = np.where(raw > 30, raw,
                       np.log1p(np.exp(-np.abs(raw))) + np.maximum(raw, 0.0))
        return pos + EPS

    # two_compartment: first three through softplus, fourth raw (sigmoid downstream)
    pos = np.where(raw[:3] > 30, raw[:3],
                   np.log1p(np.exp(-np.abs(raw[:3]))) + np.maximum(raw[:3], 0.0))
    return np.concatenate([pos + EPS, [raw[3]]])


# ===========================================================================
#  TRF baseline
# ===========================================================================

def trf_fit(curve, kind: str, fraction: float, n_restarts: int,
            rng: np.random.Generator,
            warm_start: Optional[np.ndarray] = None) -> dict:
    t_full   = get_time(curve)
    sig_full = get_stress(curve)
    sigma0   = get_sigma0(curve)
    n_obs    = max(2, int(np.ceil(len(t_full) * fraction)))
    t_obs, s_obs = t_full[:n_obs], sig_full[:n_obs]

    bounds_lo, bounds_hi = pkp.trf_bounds(kind)

    def resid(p):
        return predict_numpy(p, sigma0, t_obs, kind) - s_obs

    if warm_start is not None:
        lo = np.array(bounds_lo, dtype=np.float64)
        # clip only the positive-constrained entries
        ws = warm_start.copy()
        n_pos = (3 if kind == "one_compartment" else 3)  # both: first 3 pos
        ws[:n_pos] = np.maximum(ws[:n_pos], lo[:n_pos] + EPS)
        inits = [ws]
    else:
        inits = [pkp.trf_random_init(kind, rng) for _ in range(n_restarts)]

    best_params, best_cost = None, np.inf
    for x0 in inits:
        try:
            res = least_squares(resid, x0, method="trf",
                                bounds=(bounds_lo, bounds_hi), max_nfev=500)
            if res.cost < best_cost:
                best_cost, best_params = float(res.cost), res.x.copy()
        except Exception:
            pass

    if best_params is None:
        best_params = inits[0]
    pred = predict_numpy(best_params, sigma0, t_full, kind)
    return {"mse": float(np.mean((pred - sig_full)**2)), "n_obs": n_obs}


# ===========================================================================
#  Fold runner (LOOCV)
# ===========================================================================

def loo_folds(curves: List) -> List[Tuple[str, List, List]]:
    out = []
    for i, c in enumerate(curves):
        train = [other for other in curves if other is not c]
        out.append((f"LOO_{i:02d}_{safe_name(get_name(c))}", train, [c]))
    return out


def run_fold(fold_label: str, train_curves: List, test_curves: List,
             all_curves: List, kinds: List[str], fractions: List[float],
             include_sigma0: bool, include_subject_id: bool,
             epochs: int, lr: float, wd: float,
             n_restarts: int, device: torch.device, seed: int,
             log_every: int) -> List[dict]:
    material_map = build_material_map(all_curves)
    s0_vals  = np.array([get_sigma0(c)    for c in all_curves], dtype=np.float64)
    w_vals   = np.array([get_weight_kg(c) for c in all_curves], dtype=np.float64)
    s0_mean  = float(s0_vals.mean()); s0_std = float(s0_vals.std() + EPS)
    w_mean   = float(w_vals.mean());  w_std  = float(w_vals.std()  + EPS)

    variant = "subject_id" if include_subject_id else "population"

    rows = []
    rng  = np.random.default_rng(seed)

    for kind in kinds:
        print(f"    [{fold_label}] Training {kind} ({variant}) on "
              f"{len(train_curves)} full PK curves ...")
        model = train_model(train_curves, material_map, kind,
                            include_sigma0, include_subject_id,
                            epochs, lr, wd, device,
                            s0_mean, s0_std, w_mean, w_std,
                            log_every=log_every)

        mlp_rows = evaluate_mlp(model, test_curves, material_map, kind,
                                include_sigma0, include_subject_id, device,
                                s0_mean, s0_std, w_mean, w_std)
        mlp_mse_by = {(r["file"], r["material"]): r["mse_mlp"] for r in mlp_rows}

        mlp_params_by = {}
        for c in test_curves:
            mlp_params_by[(get_name(c), get_material(c))] = mlp_params_numpy(
                model, c, material_map,
                s0_mean, s0_std, w_mean, w_std,
                include_sigma0, include_subject_id, device, kind)

        for frac in fractions:
            for c in test_curves:
                key       = (get_name(c), get_material(c))
                mlp_mse   = mlp_mse_by[key]
                mlp_par   = mlp_params_by[key]

                cold = trf_fit(c, kind, frac, n_restarts, rng, warm_start=None)
                warm = trf_fit(c, kind, frac, 1,           rng, warm_start=mlp_par)

                rows.append({
                    "fold":     fold_label,
                    "variant":  variant,
                    "equation": kind,
                    "material": get_material(c),
                    "file":     get_name(c),
                    "sigma0":   get_sigma0(c),
                    "weight":   get_weight_kg(c),
                    "fraction": frac,
                    "mlp_only": mlp_mse,
                    "trf_cold": cold["mse"],
                    "trf_warm": warm["mse"],
                    "n_obs":    cold["n_obs"],
                    "n_train":  len(train_curves),
                    "n_test":   len(test_curves),
                })
    return rows


# ===========================================================================
#  Aggregation + reporting
# ===========================================================================

def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (eq, frac), grp in df.groupby(["equation","fraction"]):
        for method in ["mlp_only","trf_cold","trf_warm"]:
            vals = grp[method].dropna()
            out.append({
                "equation": eq, "fraction": frac, "method": method,
                "mean_mse": float(vals.mean()),
                "std_mse":  float(vals.std(ddof=1)) if len(vals)>1 else 0.0,
                "n_obs":    int(len(vals)),
            })
    return pd.DataFrame(out)


def print_threeway_summary(agg: pd.DataFrame, kinds: List[str],
                           fractions: List[float]) -> None:
    print("\n=== Three-way comparison (mean ± std across LOO folds) ===")
    for kind in kinds:
        sub = agg[agg["equation"] == kind]
        print(f"\n--- {kind} ---")
        print(f"  {'fraction':>10}  {'n_obs':>5}  {'mlp_only':>14}  "
              f"{'trf_cold':>14}  {'trf_warm':>14}  {'best':>10}")
        for frac in sorted(fractions):
            row = {}
            for method in ["mlp_only","trf_cold","trf_warm"]:
                r = sub[(sub["fraction"]==frac) & (sub["method"]==method)]
                row[method] = ((float(r["mean_mse"].iloc[0]),
                                float(r["std_mse"].iloc[0]))
                               if len(r) else (float("nan"), float("nan")))
            best  = min(row, key=lambda m: row[m][0])
            parts = {m: f"{row[m][0]:8.4f}±{row[m][1]:6.4f}" for m in row}
            n_obs = max(2, int(np.ceil(11 * frac)))
            print(f"  {frac:>10.4f}  {n_obs:>5d}  {parts['mlp_only']:>14}  "
                  f"{parts['trf_cold']:>14}  {parts['trf_warm']:>14}  {best:>10}")


def print_crossover(agg: pd.DataFrame, kinds: List[str]) -> None:
    print("\n=== Crossover (TRF cold first beats MLP) ===")
    for kind in kinds:
        sub  = agg[agg["equation"] == kind]
        mlp  = sub[sub["method"]=="mlp_only"][["fraction","mean_mse"]].set_index("fraction")
        cold = sub[sub["method"]=="trf_cold"][["fraction","mean_mse"]].set_index("fraction")
        j    = mlp.join(cold, lsuffix="_mlp", rsuffix="_cold").sort_index()
        crossed = j[j["mean_mse_cold"] < j["mean_mse_mlp"]]
        if len(crossed) == 0:
            print(f"  {kind:18s}: TRF cold never beats MLP in tested range")
        else:
            print(f"  {kind:18s}: TRF cold first beats MLP at fraction "
                  f"{crossed.index[0]:.3f}  "
                  f"(n_obs={max(2,int(np.ceil(11*crossed.index[0])))})")


def print_per_subject_summary(df: pd.DataFrame, kinds: List[str]) -> None:
    print("\n=== Per-subject mean MSE across LOO folds ===")
    for kind in kinds:
        print(f"\n--- {kind} ---")
        sub  = df[df["equation"] == kind]
        files = sorted(sub["file"].unique())
        print(f"  {'subject':<25}  {'dose_mg':>8}  "
              f"{'mlp_only':>10}  {'trf_cold':>10}  {'trf_warm':>10}")
        for f in files:
            m = sub[sub["file"] == f]
            print(f"  {f:<25}  {m['sigma0'].iloc[0]:>8.2f}  "
                  f"{m['mlp_only'].mean():>10.4f}  "
                  f"{m['trf_cold'].mean():>10.4f}  "
                  f"{m['trf_warm'].mean():>10.4f}")


def print_equation_comparison(df: pd.DataFrame, kinds: List[str]) -> None:
    """Direct head-to-head: for each subject, which equation gives a lower
    MLP-only MSE? This is the analog of the rheology 'Guiu-Pratt wins on
    4/5 materials' result."""
    if len(kinds) < 2:
        return
    print("\n=== Equation comparison: which PK model wins per subject? ===")
    # Take the MLP-only MSE at fraction 1.0 (or whatever max fraction is
    # available; since MLP is fraction-independent, any fraction works).
    fmax = df["fraction"].max()
    sub = df[df["fraction"] == fmax]

    pivot = sub.pivot_table(index="file", columns="equation",
                            values="mlp_only", aggfunc="mean")
    if len(kinds) == 2 and set(kinds) <= set(pivot.columns):
        a, b = kinds[0], kinds[1]
        pivot["winner"] = np.where(pivot[a] < pivot[b], a, b)
        a_wins = (pivot["winner"] == a).sum()
        b_wins = (pivot["winner"] == b).sum()
        print(pivot.to_string())
        print(f"\n  {a} wins on {a_wins}/{len(pivot)} subjects")
        print(f"  {b} wins on {b_wins}/{len(pivot)} subjects")


# ===========================================================================
#  Experiment runners
# ===========================================================================

def run_experiment(curves: List, output_dir: Path, kinds: List[str],
                   include_sigma0: bool, include_subject_id: bool,
                   epochs: int, lr: float, wd: float,
                   fractions: List[float], n_restarts: int,
                   master_seed: int, device: torch.device,
                   log_every: int, verbose: bool = True) -> pd.DataFrame:
    """
    Run one complete LOOCV three-way experiment for a single configuration
    (one input variant, one sigma0 setting, one master seed).

    Writes per-fold CSVs, all_folds_raw.csv, aggregated_summary.csv, and
    VARIANT.txt into output_dir. Returns the full per-curve DataFrame so a
    caller (e.g. the --all-variants orchestrator) can aggregate further.
    """
    variant_label = "subject_id" if include_subject_id else "population"
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(master_seed)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  PK Three-Way Comparison: Theophylline (LOOCV, 12 subjects)")
        print(f"  Models           : {kinds}")
        print(f"  INPUT VARIANT    : {variant_label.upper()}")
        if include_subject_id:
            print(f"                     input = [one-hot subject ID"
                  f"{', sigma0' if include_sigma0 else ''}]")
        else:
            print(f"                     input = [weight"
                  f"{', sigma0' if include_sigma0 else ''}]")
        print(f"  Include sigma0   : {include_sigma0}")
        print(f"  TRF fractions    : {fractions}")
        print(f"  TRF restarts     : {n_restarts}")
        print(f"  Epochs           : {epochs}  |  LR: {lr}")
        print(f"  Master seed      : {master_seed}  |  Device: {device}")
        print(f"{'='*70}")

    folds = loo_folds(curves)

    all_rows = []
    for fold_i, (label, train_c, test_c) in enumerate(folds):
        if verbose:
            print(f"  Fold {fold_i+1}/{len(folds)}  [{label}]  "
                  f"variant={variant_label}  seed={master_seed}")
        fold_seed = master_seed + fold_i * 7919
        set_seed(fold_seed)
        rows = run_fold(label, train_c, test_c, curves, kinds, fractions,
                        include_sigma0, include_subject_id,
                        epochs, lr, wd, n_restarts, device,
                        fold_seed, log_every)
        all_rows.extend(rows)
        pd.DataFrame(rows).to_csv(
            output_dir / f"fold_{safe_name(label)}.csv", index=False)

    full = pd.DataFrame(all_rows)
    # Tag every row with the configuration so concatenated CSVs stay unambiguous
    full["include_sigma0"] = include_sigma0
    full["master_seed"]    = master_seed
    full.to_csv(output_dir / "all_folds_raw.csv", index=False)

    agg = aggregate_results(full)
    agg.to_csv(output_dir / "aggregated_summary.csv", index=False)

    (output_dir / "VARIANT.txt").write_text(
        f"variant={variant_label}\n"
        f"include_subject_id={include_subject_id}\n"
        f"include_sigma0={include_sigma0}\n"
        f"master_seed={master_seed}\n"
        f"trf_restarts={n_restarts}\n"
    )

    if verbose:
        print(f"  [SAVED] {output_dir}/  ({len(full)} rows)")
        print_threeway_summary(agg, kinds, fractions)
        print_crossover(agg, kinds)
        print_per_subject_summary(full, kinds)
        print_equation_comparison(full, kinds)

    return full


# ===========================================================================
#  Cross-seed / cross-variant aggregation (for --all-variants mode)
# ===========================================================================

# The three configurations the paper needs. Each is (label, include_subject_id,
# include_sigma0). The subject-id σ0-ablation is omitted by default because, with
# subject identity available, the network can memorise per-subject parameters and
# the σ0 feature contributes little — making that ablation hard to interpret. Add
# it here if you want the full 2x2 grid.
ALL_VARIANT_CONFIGS = [
    # label                    include_subject_id   include_sigma0
    ("subjid_sigma0",          True,                True),
    ("population_sigma0",      False,               True),
    ("population_nosigma0",    False,               False),
]


def aggregate_across_seeds(per_seed_frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Given one all_folds_raw frame per seed (same configuration), return a
    per-(equation, fraction, method) table with mean and std taken ACROSS
    SEEDS of the per-seed mean MSE. This separates seed variability from
    fold variability."""
    rows = []
    # First reduce each seed to its per-(equation,fraction,method) mean
    per_seed_means = []
    for df in per_seed_frames:
        seed = int(df["master_seed"].iloc[0])
        for (eq, frac), grp in df.groupby(["equation", "fraction"]):
            for method in ["mlp_only", "trf_cold", "trf_warm"]:
                per_seed_means.append({
                    "seed": seed, "equation": eq, "fraction": frac,
                    "method": method, "mean_mse": float(grp[method].mean()),
                })
    psm = pd.DataFrame(per_seed_means)
    for (eq, frac, method), grp in psm.groupby(["equation", "fraction", "method"]):
        vals = grp["mean_mse"]
        rows.append({
            "equation": eq, "fraction": frac, "method": method,
            "mean_over_seeds": float(vals.mean()),
            "std_over_seeds":  float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "n_seeds":         int(len(vals)),
        })
    return pd.DataFrame(rows)


def aggregate_equation_wins_across_seeds(
        per_seed_frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate the per-subject 'which equation wins' result across seeds.
    For each subject, report how many seeds each equation won, plus the
    mean MLP MSE per equation across seeds."""
    if not per_seed_frames or len(per_seed_frames[0]["equation"].unique()) < 2:
        return pd.DataFrame()

    fmax = per_seed_frames[0]["fraction"].max()
    win_counter = defaultdict(lambda: defaultdict(int))   # file -> eq -> wins
    mse_acc     = defaultdict(lambda: defaultdict(list))  # file -> eq -> [mse]

    for df in per_seed_frames:
        sub = df[df["fraction"] == fmax]
        piv = sub.pivot_table(index="file", columns="equation",
                              values="mlp_only", aggfunc="mean")
        eqs = list(piv.columns)
        for f, row in piv.iterrows():
            best = min(eqs, key=lambda e: row[e])
            win_counter[f][best] += 1
            for e in eqs:
                mse_acc[f][e].append(float(row[e]))

    rows = []
    for f in sorted(win_counter):
        rec = {"file": f}
        for e in mse_acc[f]:
            arr = np.array(mse_acc[f][e])
            rec[f"{e}_mlp_mean"] = float(arr.mean())
            rec[f"{e}_mlp_std"]  = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            rec[f"{e}_wins"]     = int(win_counter[f].get(e, 0))
        rows.append(rec)
    return pd.DataFrame(rows)


def run_all_variants(curves: List, base_output_dir: Path, kinds: List[str],
                     epochs: int, lr: float, wd: float,
                     fractions: List[float], n_restarts: int,
                     seeds: List[int], device: torch.device,
                     log_every: int) -> None:
    """Orchestrate every (variant config x seed) combination, then aggregate
    across seeds within each config and write a master summary."""
    base_output_dir.mkdir(parents=True, exist_ok=True)
    n_configs = len(ALL_VARIANT_CONFIGS)
    n_total   = n_configs * len(seeds)
    print(f"\n{'#'*70}")
    print(f"#  ALL-VARIANTS RUN")
    print(f"#  Configurations : {[c[0] for c in ALL_VARIANT_CONFIGS]}")
    print(f"#  Seeds          : {seeds}")
    print(f"#  Total runs     : {n_configs} configs x {len(seeds)} seeds "
          f"= {n_total}")
    print(f"#  TRF restarts   : {n_restarts}")
    print(f"{'#'*70}")

    run_idx = 0
    master_rows = []          # combined per-curve rows across everything
    per_config_seedframes = {}  # config_label -> list of per-seed frames

    for cfg_label, inc_sid, inc_s0 in ALL_VARIANT_CONFIGS:
        seed_frames = []
        for seed in seeds:
            run_idx += 1
            out_dir = base_output_dir / cfg_label / f"seed_{seed}"
            print(f"\n{'='*70}")
            print(f"  RUN {run_idx}/{n_total}  "
                  f"config={cfg_label}  seed={seed}")
            print(f"{'='*70}")
            full = run_experiment(
                curves, out_dir, kinds,
                include_sigma0=inc_s0, include_subject_id=inc_sid,
                epochs=epochs, lr=lr, wd=wd,
                fractions=fractions, n_restarts=n_restarts,
                master_seed=seed, device=device,
                log_every=log_every, verbose=False)
            full["config"] = cfg_label
            seed_frames.append(full)
            master_rows.append(full)
            # brief per-run line so progress is visible
            agg = aggregate_results(full)
            for kind in kinds:
                mlp_row = agg[(agg.equation == kind) &
                              (agg.method == "mlp_only")]
                if len(mlp_row):
                    mlp_val = mlp_row["mean_mse"].iloc[0]
                    print(f"    {cfg_label} seed={seed} {kind:18s} "
                          f"MLP MSE = {mlp_val:.4f}")

        per_config_seedframes[cfg_label] = seed_frames

        # Aggregate across seeds for this config
        cfg_dir = base_output_dir / cfg_label
        seed_agg = aggregate_across_seeds(seed_frames)
        seed_agg.insert(0, "config", cfg_label)
        seed_agg.to_csv(cfg_dir / "SEED_AGGREGATED.csv", index=False)

        wins_agg = aggregate_equation_wins_across_seeds(seed_frames)
        if len(wins_agg):
            wins_agg.insert(0, "config", cfg_label)
            wins_agg.to_csv(cfg_dir / "SEED_EQUATION_WINS.csv", index=False)
        print(f"\n  [SAVED] {cfg_dir}/SEED_AGGREGATED.csv")

    # Master combined raw file
    master = pd.concat(master_rows, ignore_index=True)
    master.to_csv(base_output_dir / "MASTER_all_folds_raw.csv", index=False)

    # Master human-readable summary
    _write_master_summary(base_output_dir, per_config_seedframes,
                          kinds, fractions, seeds, n_restarts)
    print(f"\n{'#'*70}")
    print(f"#  ALL-VARIANTS RUN COMPLETE")
    print(f"#  Master files in: {base_output_dir.resolve()}")
    print(f"#    MASTER_all_folds_raw.csv   — every row, every config, every seed")
    print(f"#    MASTER_SUMMARY.txt         — human-readable cross-seed summary")
    print(f"#    <config>/SEED_AGGREGATED.csv      — per-config cross-seed table")
    print(f"#    <config>/SEED_EQUATION_WINS.csv   — per-config cross-seed wins")
    print(f"{'#'*70}")


def _write_master_summary(base_dir: Path, per_config_seedframes: Dict,
                          kinds: List[str], fractions: List[float],
                          seeds: List[int], n_restarts: int) -> None:
    """Write a plain-text master summary across all configs and seeds."""
    lines = []
    lines.append("=" * 72)
    lines.append("  PK ALL-VARIANTS MASTER SUMMARY")
    lines.append("=" * 72)
    lines.append(f"  Seeds       : {seeds}")
    lines.append(f"  TRF restarts: {n_restarts}")
    lines.append(f"  Configs     : {list(per_config_seedframes.keys())}")
    lines.append("")

    for cfg_label, seed_frames in per_config_seedframes.items():
        lines.append("-" * 72)
        lines.append(f"  CONFIG: {cfg_label}")
        lines.append("-" * 72)
        seed_agg = aggregate_across_seeds(seed_frames)
        for kind in kinds:
            lines.append(f"  --- {kind} ---")
            lines.append(f"    {'fraction':>9}  {'method':>9}  "
                          f"{'mean(seeds)':>13}  {'std(seeds)':>11}")
            sub = seed_agg[seed_agg.equation == kind]
            for frac in sorted(fractions):
                for method in ["mlp_only", "trf_cold", "trf_warm"]:
                    r = sub[(sub.fraction == frac) & (sub.method == method)]
                    if len(r):
                        lines.append(
                            f"    {frac:>9.4f}  {method:>9}  "
                            f"{r['mean_over_seeds'].iloc[0]:>13.4f}  "
                            f"{r['std_over_seeds'].iloc[0]:>11.4f}")
            lines.append("")

        # Equation-wins summary across seeds
        wins = aggregate_equation_wins_across_seeds(seed_frames)
        if len(wins):
            lines.append(f"  Equation wins per subject (across {len(seeds)} seeds):")
            win_cols = [c for c in wins.columns if c.endswith("_wins")]
            for _, row in wins.iterrows():
                detail = "  ".join(
                    f"{c.replace('_wins','')}={int(row[c])}" for c in win_cols)
                lines.append(f"    {row['file']:<26}  {detail}")
            # tally
            lines.append("")
            for c in win_cols:
                eq = c.replace("_wins", "")
                total = int(wins[c].sum())
                lines.append(f"    TOTAL {eq:18s}: {total} subject-seed wins "
                             f"out of {len(wins)*len(seeds)}")
        lines.append("")

    (base_dir / "MASTER_SUMMARY.txt").write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv",          type=Path, required=True,
                   help="Path to theoph.csv (exported from R's Theoph dataset)")
    p.add_argument("--output-dir",   type=Path, default=Path("results_pk_threeway"))
    p.add_argument("--model",        choices=VALID_MODELS, default="both")
    p.add_argument("--epochs",       type=int,   default=500)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--log-every",    type=int,   default=100)
    p.add_argument("--no-sigma0",    action="store_true",
                   help="Drop the sigma0 (total dose) feature. Default: include it. "
                        "Ignored when --all-variants is set.")
    p.add_argument("--no-subject-id", action="store_true",
                   help="Run the POPULATION variant: input = "
                        "[weight_normalized, sigma0_normalized]. "
                        "Default (off): SUBJECT-ID variant with one-hot subject. "
                        "Ignored when --all-variants is set.")
    p.add_argument("--all-variants", action="store_true",
                   help="Orchestration mode. Run every (variant config x seed) "
                        "combination in one invocation and aggregate across "
                        "seeds. Configs: subjid_sigma0, population_sigma0, "
                        "population_nosigma0. Use --seeds to set the seed list. "
                        "Overrides --no-sigma0 / --no-subject-id / --master-seed.")
    p.add_argument("--seeds", type=str, default="42,43,44,45,46",
                   help="Comma-separated master seeds for --all-variants mode "
                        "(default: 42,43,44,45,46). Ignored in single-run mode.")
    p.add_argument("--trf-fractions", type=str,
                   default="0.10,0.20,0.30,0.45,0.55,0.75,1.00")
    p.add_argument("--trf-restarts",  type=int, default=50,
                   help="TRF cold-start random restarts per (curve, fraction). "
                        "Default 50; was 10 in earlier versions. The higher value "
                        "reduces catastrophic variance at small n_obs where the "
                        "3- or 4-parameter PK model approaches underdetermination.")
    p.add_argument("--master-seed",   type=int, default=42,
                   help="Seed for single-run mode. Ignored when --all-variants set.")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    kinds = (["one_compartment","two_compartment"] if args.model == "both"
             else [args.model])
    fractions = parse_fractions(args.trf_fractions)
    device    = torch.device(args.device)

    print("\n-- Loading PK data --")
    curves = ppl.load_curves(args.csv)
    print(f"  Loaded {len(curves)} subjects from {args.csv}")
    for c in curves:
        print(f"    {c}")

    if args.all_variants:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        if not seeds:
            raise ValueError("--seeds produced an empty list.")
        run_all_variants(
            curves, args.output_dir, kinds,
            epochs=args.epochs, lr=args.lr, wd=args.weight_decay,
            fractions=fractions, n_restarts=args.trf_restarts,
            seeds=seeds, device=device, log_every=args.log_every)
    else:
        include_s0         = not args.no_sigma0
        include_subject_id = not args.no_subject_id
        full = run_experiment(
            curves, args.output_dir, kinds,
            include_sigma0=include_s0, include_subject_id=include_subject_id,
            epochs=args.epochs, lr=args.lr, wd=args.weight_decay,
            fractions=fractions, n_restarts=args.trf_restarts,
            master_seed=args.master_seed, device=device,
            log_every=args.log_every, verbose=True)
        variant = "subject_id" if include_subject_id else "population"
        print(f"\nDone. Outputs -> {args.output_dir.resolve()}  "
              f"(variant={variant})")


if __name__ == "__main__":
    main()
