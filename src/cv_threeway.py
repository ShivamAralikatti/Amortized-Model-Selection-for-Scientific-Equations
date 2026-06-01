#!/usr/bin/env python3
"""
Cross-Validation Three-Way Comparison: No-Input MLP vs TRF Cold vs TRF Warm
=============================================================================

IMPORTANT NOTE ON THE "FRACTION" AXIS
-------------------------------------
"Fraction" in this script means ONE thing only: the fraction of the TEST
curve's time-series points that TRF is allowed to observe at evaluation
time. It is a property of the TRF baseline's observation budget, nothing
else.

The MLP receives NO observations from the test curve. Its inputs are
(one-hot material, sigma0) only. Therefore the MLP's prediction --
and hence its test MSE -- is identical at every fraction. This is BY
DESIGN, not a bug. The MLP column in the three-way table will be a
constant value per fold per equation; the crossover with TRF emerges
because TRF improves with fraction while the MLP stays flat.

Do NOT modify train_model to truncate training curves to the fraction
window. The MLP must always be trained on the FULL training curves so
that its loss has a long-horizon signal to learn from. Truncating
training curves to match TRF's evaluation budget conflates two unrelated
quantities (training-data sparsity vs. test-time observation budget) and
destroys the methodology.

The MLP is trained ONCE per (fold, kind). The fractions loop runs only
over TRF evaluation.

Usage:
    # LOO over all curves, both models
    python cv_threeway.py --data-dir data --cv-mode loo --split-variant all

    # 5-fold CV (5 seeds), duplicates-only test set, Guiu-Pratt
    python cv_threeway.py --data-dir data --cv-mode kfold \\
        --split-variant duplicates --model guiu_pratt --n-seeds 5

    # LOO, duplicates, custom fractions
    python cv_threeway.py --data-dir data --cv-mode loo \\
        --split-variant duplicates \\
        --trf-fractions 0.001,0.005,0.01,0.02,0.05,0.10,0.20,0.40
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

import guiu_pratt_pipeline as gpp

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
EPS         = 1e-8
R_GAS       = 8.314
UNKNOWN_MAT = "__UNKNOWN__"

VALID_CV_MODES       = ["loo", "kfold"]
VALID_SPLIT_VARIANTS = ["all", "duplicates"]
VALID_MODELS         = ["guiu_pratt", "thermal", "both"]


# ===========================================================================
#  UTILITIES
# ===========================================================================

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
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
        if not part:
            continue
        is_pct = part.endswith("%")
        v = float(part[:-1]) if is_pct else float(part)
        if is_pct:
            v /= 100.0
        elif v > 1.0:
            v /= 100.0
        if not 0.0 < v <= 1.0:
            raise ValueError(f"Invalid fraction: {part!r}")
        out.append(v)
    if not out:
        raise ValueError("Provide at least one fraction.")
    return sorted(set(out))


# ===========================================================================
#  CURVE ATTRIBUTE ACCESS
# ===========================================================================

def _get_attr(obj, candidates: List[str], required: bool = True, default=None):
    for name in candidates:
        if hasattr(obj, name):
            return getattr(obj, name)
    if required:
        raise AttributeError(
            f"None of {candidates} found on {type(obj).__name__}. "
            f"Available: {[a for a in dir(obj) if not a.startswith('_')]}"
        )
    return default


def get_time(c)     -> np.ndarray: return np.asarray(_get_attr(c, ["time","t","times"]))
def get_stress(c)   -> np.ndarray: return np.asarray(_get_attr(c, ["stress","sigma","stresses"]))
def get_material(c) -> str:        return str(_get_attr(c, ["material","werkstoff","alloy"]))
def get_name(c)     -> str:        return str(_get_attr(c, ["name","filename","stem"], required=False, default=""))


def get_sigma0(c) -> float:
    val = _get_attr(c, ["sigma0","sigma_0","s0","S0"], required=False)
    if val is not None:
        return float(val)
    return float(get_stress(c)[0])


def get_temperature_K(c) -> float:
    val = _get_attr(c, ["temperature_K","T_K","T_kelvin","temp_K"], required=False)
    if val is not None:
        return float(val)
    val = _get_attr(c, ["temperature","T","temp"], required=False)
    if val is not None:
        v = float(val)
        return v + 273.15 if v < 250.0 else v
    val = _get_attr(c, ["temperature_C","T_C","temp_C","pruftemperatur"], required=False)
    if val is not None:
        return float(val) + 273.15
    return 373.15


# ===========================================================================
#  DATASET FILTERING VARIANTS
# ===========================================================================

def group_by_material(curves: List) -> Dict[str, List]:
    groups: Dict[str, List] = defaultdict(list)
    for c in curves:
        groups[get_material(c)].append(c)
    return dict(groups)


def eligible_test_curves(curves: List, split_variant: str) -> List:
    if split_variant == "all":
        return list(curves)
    groups = group_by_material(curves)
    return [c for c in curves if len(groups[get_material(c)]) >= 2]


def print_dataset_summary(curves: List, eligible: List, split_variant: str) -> None:
    groups = group_by_material(curves)
    print(f"\n  Dataset summary  (split_variant={split_variant})")
    print(f"  {'─'*52}")
    print(f"  {'Material':<20}  {'# files':>7}  {'eligible as test':>16}")
    for mat in sorted(groups):
        n    = len(groups[mat])
        elig = "YES" if split_variant == "all" or n >= 2 else "NO  (singleton)"
        print(f"  {mat:<20}  {n:>7}  {elig:>16}")
    print(f"  {'─'*52}")
    print(f"  Total files   : {len(curves)}")
    print(f"  Eligible test : {len(eligible)}")
    print(f"  Always-train  : {len(curves) - len(eligible)}")


# ===========================================================================
#  CROSS-VALIDATION FOLD GENERATORS
# ===========================================================================

def loo_folds(eligible: List, non_eligible: List) -> List[Tuple[str, List, List]]:
    """
    Leave-One-Out: each eligible curve is the test set exactly once.
    Train = all other eligible + all non-eligible (always-train) curves.

    BUG 2 FIX: non_eligible is passed in directly so singleton-material
    curves are never accidentally dropped from any training set.
    """
    folds = []
    for i, test_c in enumerate(eligible):
        train = [c for c in eligible if id(c) != id(test_c)] + non_eligible
        label = f"LOO_{i:03d}_{safe_name(get_name(test_c))}"
        folds.append((label, train, [test_c]))
    return folds


def kfold_folds(
    eligible: List,
    k: int,
    seed: int,
    non_eligible: List,
) -> List[Tuple[str, List, List]]:
    """
    K-Fold: partition eligible curves into k folds.
    Test = that fold's eligible curves.
    Train = remaining eligible + all non-eligible.
    Warns when a material appears only in the test fold (no training rep).
    """
    rng    = np.random.default_rng(seed)
    idx    = rng.permutation(len(eligible))
    splits = np.array_split(idx, k)

    folds = []
    for fold_i, test_idx in enumerate(splits):
        test_set       = {int(i) for i in test_idx}
        train_eligible = [eligible[i] for i in range(len(eligible)) if i not in test_set]
        test_curves    = [eligible[i] for i in test_idx]
        train_curves   = train_eligible + non_eligible

        # Warn if any test material has no training representative
        train_mats = {get_material(c) for c in train_curves}
        missing    = {get_material(c) for c in test_curves} - train_mats
        if missing:
            print(f"  [WARN] Fold K{k}F{fold_i+1} seed{seed}: "
                  f"material(s) {missing} have NO training representative. "
                  f"MLP will use UNKNOWN embedding.")

        folds.append((f"K{k}F{fold_i+1}_seed{seed}", train_curves, test_curves))
    return folds


# ===========================================================================
#  PHYSICS LAYERS
# ===========================================================================

def guiu_pratt_torch(sigma0: torch.Tensor, raw: torch.Tensor,
                     t: torch.Tensor) -> torch.Tensor:
    E     = nn.functional.softplus(raw[0]) + EPS
    xi    = nn.functional.softplus(raw[1]) + EPS
    theta = nn.functional.softplus(raw[2]) + EPS
    beta  = sigma0 / (E * theta)
    tau   = xi / theta
    return sigma0 - beta * torch.log1p(t / tau)


def thermal_torch(sigma0: torch.Tensor, raw: torch.Tensor,
                  t: torch.Tensor, T_K: torch.Tensor) -> torch.Tensor:
    V = nn.functional.softplus(raw[0]) + EPS
    C = nn.functional.softplus(raw[1]) + EPS
    return sigma0 - (R_GAS * T_K / V) * torch.log1p(t / C)


def predict_numpy(params: np.ndarray, sigma0: float, t: np.ndarray,
                  T_K: float, kind: str) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    if kind == "guiu_pratt":
        E, xi, theta = (max(float(params[0]), EPS),
                        max(float(params[1]), EPS),
                        max(float(params[2]), EPS))
        return sigma0 - (sigma0/(E*theta)) * np.log1p(t/(xi/theta))
    V, C = max(float(params[0]), EPS), max(float(params[1]), EPS)
    return sigma0 - (R_GAS*T_K/V) * np.log1p(t/C)


# ===========================================================================
#  MATERIAL ENCODING
# ===========================================================================

def build_material_map(curves: List) -> Dict[str, int]:
    mats    = sorted({get_material(c) for c in curves})
    mapping = {m: i for i, m in enumerate(mats)}
    mapping[UNKNOWN_MAT] = len(mapping)
    return mapping


def one_hot(i: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32); v[i] = 1.0; return v


def make_input(curve, material_map: Dict[str,int],
               sigma0_mean: float, sigma0_std: float,
               include_sigma0: bool) -> np.ndarray:
    n_mat = len(material_map)
    mid   = int(material_map.get(get_material(curve), material_map[UNKNOWN_MAT]))
    oh    = one_hot(mid, n_mat)
    if include_sigma0:
        s0_norm = (get_sigma0(curve) - sigma0_mean) / max(sigma0_std, EPS)
        return np.concatenate([oh, [s0_norm]]).astype(np.float32)
    return oh


# ===========================================================================
#  NO-INPUT MLP
# ===========================================================================

class NoInputMLP(nn.Module):
    """Maps (one-hot material [+ sigma0_zscore]) -> raw constitutive params."""

    def __init__(self, n_materials: int, n_outputs: int,
                 include_sigma0: bool = True,
                 output_bias_init: Tuple[float,...] = (80.0, 15.0, 1.0)):
        super().__init__()
        assert len(output_bias_init) == n_outputs
        self.include_sigma0 = include_sigma0
        self.n_materials    = n_materials
        input_dim = n_materials + (1 if include_sigma0 else 0)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def predict_curve_torch(model: NoInputMLP, x: torch.Tensor,
                        sigma0: torch.Tensor, t: torch.Tensor,
                        T_K: torch.Tensor, kind: str) -> torch.Tensor:
    raw = model(x)
    if kind == "guiu_pratt":
        return guiu_pratt_torch(sigma0, raw, t)
    return thermal_torch(sigma0, raw, t, T_K)


# ===========================================================================
#  TRAINING
# ===========================================================================

def train_model(train_curves: List, material_map: Dict[str,int],
                kind: str, include_sigma0: bool, epochs: int,
                lr: float, wd: float, device: torch.device,
                sigma0_mean: float, sigma0_std: float,
                log_every: int = 100) -> NoInputMLP:
    """
    Train the no-input MLP on the FULL time-series of every training curve.

    The MLP receives (one-hot material [+ sigma0]) as input and outputs raw
    constitutive parameters. The differentiable physics layer applies the
    chosen constitutive equation at every time point of the training curve,
    and the loss is the MSE over the FULL curve. Backprop through the physics
    layer drives the MLP to produce parameters that extrapolate well.

    DO NOT pass any "fraction" or "observation window" argument here. The
    MLP's loss signal comes from the full long-horizon behavior of the
    training curves, not from a truncated initial window. Truncating training
    curves to match TRF's test-time observation budget destroys the loss
    signal and produces meaningless MLP predictions.
    """
    n_out, bias = ((3, (80.0, 15.0, 1.0)) if kind == "guiu_pratt"
                   else (2, (300.0, 100.0)))
    model = NoInputMLP(len(material_map), n_out, include_sigma0, bias).to(device)
    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.5)

    cached = []
    for c in train_curves:
        cached.append({
            "x":      torch.tensor(make_input(c, material_map, sigma0_mean,
                                              sigma0_std, include_sigma0), device=device),
            "t":      torch.tensor(get_time(c).astype(np.float32),   device=device),
            "y":      torch.tensor(get_stress(c).astype(np.float32), device=device),
            "sigma0": torch.tensor(get_sigma0(c),        dtype=torch.float32, device=device),
            "T_K":    torch.tensor(get_temperature_K(c), dtype=torch.float32, device=device),
        })

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        losses = [
            torch.mean((predict_curve_torch(model, c["x"], c["sigma0"],
                                            c["t"], c["T_K"], kind) - c["y"])**2)
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


# ===========================================================================
#  MLP EVALUATION
# ===========================================================================

@torch.no_grad()
def evaluate_mlp(model: NoInputMLP, test_curves: List,
                 material_map: Dict[str,int], kind: str,
                 include_sigma0: bool, device: torch.device,
                 sigma0_mean: float, sigma0_std: float) -> List[dict]:
    model.eval()
    rows = []
    for c in test_curves:
        x    = torch.tensor(make_input(c, material_map, sigma0_mean,
                                       sigma0_std, include_sigma0), device=device)
        t    = torch.tensor(get_time(c).astype(np.float32),   device=device)
        y    = torch.tensor(get_stress(c).astype(np.float32), device=device)
        s0   = torch.tensor(get_sigma0(c),        dtype=torch.float32, device=device)
        T_K  = torch.tensor(get_temperature_K(c), dtype=torch.float32, device=device)
        pred = predict_curve_torch(model, x, s0, t, T_K, kind)
        mse  = float(torch.mean((pred - y)**2).item())
        rows.append({"material": get_material(c), "file": get_name(c),
                     "sigma0": get_sigma0(c), "mse_mlp": mse})
    return rows


@torch.no_grad()
def mlp_params_numpy(model: NoInputMLP, curve, material_map: Dict[str,int],
                     sigma0_mean: float, sigma0_std: float,
                     include_sigma0: bool, device: torch.device) -> np.ndarray:
    model.eval()
    x   = torch.tensor(make_input(curve, material_map, sigma0_mean,
                                   sigma0_std, include_sigma0), device=device)
    raw = model(x).cpu().numpy().astype(np.float64)
    pos = np.where(raw > 30, raw,
                   np.log1p(np.exp(-np.abs(raw))) + np.maximum(raw, 0.0))
    return pos + EPS


# ===========================================================================
#  TRF BASELINE
# ===========================================================================

def trf_random_init(kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind == "guiu_pratt":
        return np.array([10**rng.uniform(0,3), 10**rng.uniform(-1,4),
                         10**rng.uniform(-2,2)], dtype=np.float64)
    return np.array([10**rng.uniform(1,4), 10**rng.uniform(-1,4)], dtype=np.float64)


def trf_fit(curve, kind: str, fraction: float, n_restarts: int,
            rng: np.random.Generator,
            warm_start: Optional[np.ndarray] = None) -> dict:
    t_full   = get_time(curve).astype(np.float64)
    sig_full = get_stress(curve).astype(np.float64)
    sigma0   = get_sigma0(curve)
    T_K      = get_temperature_K(curve)
    n_obs    = max(2, int(np.ceil(len(t_full) * fraction)))
    t_obs, s_obs = t_full[:n_obs], sig_full[:n_obs]

    bounds_lo = [EPS] * (3 if kind == "guiu_pratt" else 2)
    bounds_hi = [np.inf] * len(bounds_lo)

    def resid(p): return predict_numpy(p, sigma0, t_obs, T_K, kind) - s_obs

    inits = ([np.maximum(warm_start, np.array(bounds_lo) + EPS)]
             if warm_start is not None
             else [trf_random_init(kind, rng) for _ in range(n_restarts)])

    best_params, best_cost = None, np.inf
    for x0 in inits:
        try:
            res = least_squares(resid, x0, method="trf",
                                bounds=(bounds_lo, bounds_hi), max_nfev=300)
            if res.cost < best_cost:
                best_cost, best_params = float(res.cost), res.x.copy()
        except Exception:
            pass

    if best_params is None:
        best_params = inits[0]
    pred = predict_numpy(best_params, sigma0, t_full, T_K, kind)
    return {"mse": float(np.mean((pred - sig_full)**2)), "n_obs": n_obs}


# ===========================================================================
#  ONE FOLD RUN  --  FIXED
# ===========================================================================

def run_fold(fold_label: str, train_curves: List, test_curves: List,
             all_curves: List, kinds: List[str], fractions: List[float],
             include_sigma0: bool, epochs: int, lr: float, wd: float,
             n_restarts: int, device: torch.device, seed: int,
             log_every: int) -> List[dict]:
    """
    For each constitutive equation, train ONE no-input MLP on the full
    training curves of this fold, then evaluate three methods on every
    test curve at every observation fraction:

        mlp_only : the trained MLP's full-curve prediction (no test-curve
                   observations used; identical at every fraction).
        trf_cold : nonlinear least-squares fit to the first ceil(f * N_test)
                   points of the test curve, with n random restarts.
        trf_warm : nonlinear least-squares initialized from the MLP's
                   parameter prediction, single call.

    BUG 2 FIX: sigma0_mean/std computed from all_curves (not just
    train_curves) so the feature encoding is stable across folds.
    """
    # Stable sigma0 statistics across all folds
    material_map = build_material_map(all_curves)
    sigma0_vals  = np.array([get_sigma0(c) for c in all_curves], dtype=np.float64)
    sigma0_mean  = float(sigma0_vals.mean())
    sigma0_std   = float(sigma0_vals.std() + EPS)

    rows = []
    rng  = np.random.default_rng(seed)

    for kind in kinds:

        # ── Train the MLP ONCE per (fold, kind) on full training curves ──
        # The MLP receives no test-curve observations, so its prediction is
        # the same at every fraction. Do not retrain inside the fractions
        # loop. Do not truncate training curves.
        print(f"    [{fold_label}] Training {kind} on "
              f"{len(train_curves)} full training curves ...")
        model = train_model(train_curves, material_map, kind,
                            include_sigma0, epochs, lr, wd, device,
                            sigma0_mean, sigma0_std, log_every=log_every)

        # MLP prediction is fraction-independent: evaluate once.
        mlp_rows = evaluate_mlp(model, test_curves, material_map,
                                 kind, include_sigma0, device,
                                 sigma0_mean, sigma0_std)
        mlp_mse_by_file = {(r["file"], r["material"]): r["mse_mlp"]
                           for r in mlp_rows}

        # Cache MLP-predicted parameters per test curve (for TRF warm start)
        mlp_params_by_test = {}
        for c in test_curves:
            mlp_params_by_test[(get_name(c), get_material(c))] = (
                mlp_params_numpy(model, c, material_map,
                                 sigma0_mean, sigma0_std,
                                 include_sigma0, device)
            )

        # ── TRF varies with fraction; loop only over fractions for TRF ──
        for frac in fractions:
            for c in test_curves:
                key        = (get_name(c), get_material(c))
                mlp_mse    = mlp_mse_by_file[key]
                mlp_params = mlp_params_by_test[key]

                cold = trf_fit(c, kind, frac, n_restarts, rng, warm_start=None)
                warm = trf_fit(c, kind, frac, 1,           rng, warm_start=mlp_params)

                rows.append({
                    "fold":        fold_label,
                    "equation":    kind,
                    "material":    get_material(c),
                    "file":        get_name(c),
                    "sigma0":      get_sigma0(c),
                    "fraction":    frac,
                    "mlp_only":    mlp_mse,      # constant per (fold, kind)
                    "trf_cold":    cold["mse"],
                    "trf_warm":    warm["mse"],
                    "n_obs":       cold["n_obs"],
                    "n_train":     len(train_curves),
                    "n_test":      len(test_curves),
                })

    return rows


# ===========================================================================
#  AGGREGATION AND PRINTING
# ===========================================================================

def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (eq, frac), grp in df.groupby(["equation","fraction"]):
        for method in ["mlp_only","trf_cold","trf_warm"]:
            vals = grp[method].dropna()
            out.append({
                "equation": eq,
                "fraction": frac,
                "method":   method,
                "mean_mse": float(vals.mean()),
                "std_mse":  float(vals.std(ddof=1)) if len(vals)>1 else 0.0,
                "n_obs":    int(len(vals)),
            })
    return pd.DataFrame(out)


def print_threeway_summary(agg: pd.DataFrame, kinds: List[str],
                           fractions: List[float]) -> None:
    print("\n=== Three-way comparison (mean ± std, averaged over folds/seeds) ===")
    for kind in kinds:
        sub = agg[agg["equation"] == kind].copy()
        print(f"\n--- {kind} ---")
        header = (f"  {'fraction':>10}  {'mlp_only':>14}  "
                  f"{'trf_cold':>14}  {'trf_warm':>14}  {'best':>10}")
        print(header)
        for frac in sorted(fractions):
            row = {}
            for method in ["mlp_only","trf_cold","trf_warm"]:
                r = sub[(sub["fraction"]==frac) & (sub["method"]==method)]
                row[method] = ((float(r["mean_mse"].iloc[0]),
                                float(r["std_mse"].iloc[0]))
                               if len(r) else (float("nan"), float("nan")))
            best  = min(row, key=lambda m: row[m][0])
            parts = {m: f"{row[m][0]:8.4f}±{row[m][1]:6.4f}" for m in row}
            print(f"  {frac:>10.4f}  {parts['mlp_only']:>14}  "
                  f"{parts['trf_cold']:>14}  {parts['trf_warm']:>14}  {best:>10}")


def print_crossover(agg: pd.DataFrame, kinds: List[str]) -> None:
    print("\n=== Crossover fractions "
          "(TRF cold first beats MLP, averaged over folds/seeds) ===")
    for kind in kinds:
        sub    = agg[agg["equation"] == kind]
        mlp    = sub[sub["method"]=="mlp_only"][["fraction","mean_mse"]].set_index("fraction")
        cold   = sub[sub["method"]=="trf_cold"][["fraction","mean_mse"]].set_index("fraction")
        joined = mlp.join(cold, lsuffix="_mlp", rsuffix="_cold").sort_index()
        crossed = joined[joined["mean_mse_cold"] < joined["mean_mse_mlp"]]
        if len(crossed) == 0:
            print(f"  {kind:12s}: TRF cold never beats MLP in tested range")
        else:
            print(f"  {kind:12s}: TRF cold first beats MLP at p = "
                  f"{crossed.index[0]*100:.3f}%")


def print_per_material_summary(df: pd.DataFrame, kinds: List[str],
                               fractions: List[float]) -> None:
    """
    BUG 2 FIX (display): Print both per-material and per-FILE breakdown.
    The per-file table includes mlp_only std and fold count so you can verify
    that the MLP MSE actually varies across folds (non-zero std), which
    confirms the model is fold-dependent.
    """
    print("\n=== Per-material mean test MSE (averaged over folds/seeds) ===")
    for kind in kinds:
        print(f"\n--- {kind} ---")
        sub  = df[df["equation"] == kind]
        mats = sorted(sub["material"].unique())
        header = (f"  {'material':<22}  {'mlp_only':>10}  "
                  f"{'trf_cold':>10}  {'trf_warm':>10}  {'n_folds':>7}")
        print(header)
        for mat in mats:
            m = sub[sub["material"] == mat]
            n = m["fold"].nunique()
            print(f"  {mat:<22}  {m['mlp_only'].mean():>10.4f}  "
                  f"{m['trf_cold'].mean():>10.4f}  "
                  f"{m['trf_warm'].mean():>10.4f}  {n:>7}")

    print("\n=== Per-file mean test MSE (averaged over folds/seeds) ===")
    print("    (mlp_std > 0 confirms the MLP is fold-dependent as expected)\n")
    for kind in kinds:
        print(f"\n--- {kind} ---")
        sub   = df[df["equation"] == kind]
        files = sorted(sub["file"].unique())
        header = (f"  {'file':<35}  {'material':<20}  "
                  f"{'mlp_only':>10}  {'trf_cold':>10}  {'trf_warm':>10}  "
                  f"{'mlp_std':>8}  {'n_folds':>7}")
        print(header)
        for fname in files:
            m       = sub[sub["file"] == fname]
            mat     = m["material"].iloc[0]
            n       = m["fold"].nunique()
            mlp_std = m["mlp_only"].std(ddof=1) if len(m) > 1 else 0.0
            print(f"  {fname:<35}  {mat:<20}  "
                  f"{m['mlp_only'].mean():>10.4f}  "
                  f"{m['trf_cold'].mean():>10.4f}  "
                  f"{m['trf_warm'].mean():>10.4f}  "
                  f"{mlp_std:>8.4f}  {n:>7}")


# ===========================================================================
#  MAIN
# ===========================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Data
    p.add_argument("--data-dir",    type=Path, required=True,
                   help="Directory containing .lis / .csv / .xlsx files")
    p.add_argument("--output-dir",  type=Path, default=Path("results_cv_threeway"))

    # CV strategy
    p.add_argument("--cv-mode",     choices=VALID_CV_MODES, default="loo")
    p.add_argument("--k-folds",     type=int, default=5)
    p.add_argument("--n-seeds",     type=int, default=5)
    p.add_argument("--master-seed", type=int, default=42)

    # Split variant
    p.add_argument("--split-variant", choices=VALID_SPLIT_VARIANTS, default="all")

    # Model
    p.add_argument("--model", choices=VALID_MODELS, default="both")

    # MLP settings
    p.add_argument("--no-sigma0",    action="store_true")
    p.add_argument("--epochs",       type=int,   default=500)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--log-every",    type=int,   default=100)

    # TRF settings
    p.add_argument("--trf-fractions", type=str,
                   default="0.001,0.005,0.01,0.02,0.05,0.10,0.20,0.40")
    p.add_argument("--trf-restarts",  type=int, default=50,
                   help="TRF cold-start random restarts per (curve, fraction). "
                        "Default 50; was 10 in earlier versions. The higher value "
                        "is more defensible for low-fraction comparisons where the "
                        "TRF cost surface has weak curvature.")

    # Device
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")

    args = p.parse_args()

    kinds          = (["guiu_pratt","thermal"] if args.model == "both" else [args.model])
    fractions      = parse_fractions(args.trf_fractions)
    device         = torch.device(args.device)
    include_s0     = not args.no_sigma0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.master_seed)

    print(f"\n{'='*65}")
    print(f"  CV Three-Way Comparison")
    print(f"  CV mode             : {args.cv_mode}"
          + (f"  (k={args.k_folds}, n_seeds={args.n_seeds})"
             if args.cv_mode == "kfold" else "  (deterministic)"))
    print(f"  Split variant       : {args.split_variant}")
    print(f"  Model(s)            : {kinds}")
    print(f"  Include sigma0      : {include_s0}")
    print(f"  MLP                 : trained ONCE per (fold, kind) on full")
    print(f"                        training curves. MLP MSE is CONSTANT")
    print(f"                        across fractions by design.")
    print(f"  TRF fractions       : {fractions}")
    print(f"  TRF restarts        : {args.trf_restarts}")
    print(f"  Epochs              : {args.epochs}  |  LR: {args.lr}")
    print(f"  Master seed         : {args.master_seed}")
    print(f"  Device              : {device}")
    print(f"{'='*65}")

    # ── load data ─────────────────────────────────────────────────────────
    print("\n-- Loading dataset --")
    loader_fn = (gpp.load_curves   if hasattr(gpp, "load_curves")   else
                 gpp.load_dataset  if hasattr(gpp, "load_dataset")  else None)
    if loader_fn is None:
        raise RuntimeError("Cannot find a loader in guiu_pratt_pipeline. "
                           "Expected load_curves or load_dataset.")
    all_curves = loader_fn(args.data_dir)
    print(f"  Loaded {len(all_curves)} curves")

    # ── eligible test set ─────────────────────────────────────────────────
    eligible     = eligible_test_curves(all_curves, args.split_variant)
    non_eligible = [c for c in all_curves
                    if id(c) not in {id(e) for e in eligible}]
    print_dataset_summary(all_curves, eligible, args.split_variant)

    if not eligible:
        print("[ERROR] No eligible test curves under the chosen split_variant.")
        return

    # ── build fold list ───────────────────────────────────────────────────
    all_folds: List[Tuple[str, List, List]] = []

    if args.cv_mode == "loo":
        # BUG 2 FIX B: pass non_eligible directly so they're always in training
        all_folds = loo_folds(eligible, non_eligible)
    else:
        seeds = [args.master_seed + i * 1000 for i in range(args.n_seeds)]
        for seed in seeds:
            all_folds.extend(kfold_folds(eligible, args.k_folds, seed, non_eligible))

    print(f"\n  Total folds to run : {len(all_folds)}")
    if args.cv_mode == "loo":
        tr, te = all_folds[0][1], all_folds[0][2]
        print(f"  Typical fold       : train={len(tr)}, test={len(te)}")
    else:
        train_sizes = [len(f[1]) for f in all_folds]
        test_sizes  = [len(f[2]) for f in all_folds]
        print(f"  Train size range   : {min(train_sizes)}--{max(train_sizes)}")
        print(f"  Test  size range   : {min(test_sizes)}--{max(test_sizes)}")
    print()

    # ── run all folds ─────────────────────────────────────────────────────
    all_rows: List[dict] = []

    for fold_i, (label, train_c, test_c) in enumerate(all_folds):
        print(f"\n{'─'*60}")
        print(f"  Fold {fold_i+1}/{len(all_folds)}  [{label}]")
        print(f"  Train: {len(train_c)} curves  |  Test: {len(test_c)} curves")
        print(f"  Train materials: {sorted({get_material(c) for c in train_c})}")
        print(f"  Test  curves   : {[get_name(c) for c in test_c]}")

        fold_seed = args.master_seed + fold_i * 7919
        set_seed(fold_seed)

        rows = run_fold(
            fold_label         = label,
            train_curves       = train_c,
            test_curves        = test_c,
            all_curves         = all_curves,
            kinds              = kinds,
            fractions          = fractions,
            include_sigma0     = include_s0,
            epochs             = args.epochs,
            lr                 = args.lr,
            wd                 = args.weight_decay,
            n_restarts         = args.trf_restarts,
            device             = device,
            seed               = fold_seed,
            log_every          = args.log_every,
        )
        all_rows.extend(rows)
        pd.DataFrame(rows).to_csv(
            args.output_dir / f"fold_{safe_name(label)}.csv", index=False)

    # ── save combined results ─────────────────────────────────────────────
    full_df = pd.DataFrame(all_rows)
    full_df.to_csv(args.output_dir / "all_folds_raw.csv", index=False)
    print(f"\n  [SAVED] all_folds_raw.csv  ({len(full_df)} rows)")

    agg = aggregate_results(full_df)
    agg.to_csv(args.output_dir / "aggregated_summary.csv", index=False)
    print(f"  [SAVED] aggregated_summary.csv")

    print_threeway_summary(agg, kinds, fractions)
    print_crossover(agg, kinds)
    print_per_material_summary(full_df, kinds, fractions)

    print(f"\nDone. All outputs -> {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()