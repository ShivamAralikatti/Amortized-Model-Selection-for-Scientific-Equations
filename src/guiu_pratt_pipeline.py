#!/usr/bin/env python3
"""
Guiu-Pratt Logarithmic Relaxation -- Physics-Informed Neural Network Pipeline
==============================================================================

Equation:
    sigma(t) = sigma0 - beta * ln(1 + t / tau_sigma)

Parameter relationships:
    sigma0     = stress at t=0  (read from data directly, NOT a NN output)
    beta       = (sigma0 / E) * theta
    tau_sigma  = xi / theta
    NN outputs : (E, xi, theta)

Three improvement options over the baseline (selectable via --mode):
    baseline     -- original pipeline (ReduceLROnPlateau, raw stress input)
    cosine       -- Option 1: CosineAnnealingLR scheduler, better LR warmup,
                    more epochs by default (300). Helps escape shallow plateaus.
    sigma0_input -- Option 2: sigma0 appended as extra scalar to Dense head.
                    Lets the network condition E/xi/theta on stress magnitude,
                    fixing the beta-scaling problem for high vs low sigma0 files.
    normalized   -- Option 3: input stress values divided by sigma0 before the
                    Conv1D. Network sees shape of decay (values in ~[0.85,1.0])
                    rather than raw magnitude. Most principled physics fix.
    combined     -- Options 2 + 3 together: normalized input AND sigma0 injected
                    into the Dense head.

Multi-draw sampling:
    For each (curve, fraction) per epoch, N_DRAWS=5 independent sets of 128
    stress values are sampled using fixed draw seeds.  Their MSE losses are
    averaged before backprop.

Draw seeds derived from master seed:
    DRAW_SEEDS = [seed+100, seed+200, ..., seed+500]
    e.g. --seed 42  ->  [142, 242, 342, 442, 542]

Usage:
    # Baseline (original):
    python guiu_pratt_pipeline.py --data-dir data --output-dir ./results_baseline --mode baseline

    # Option 1 -- cosine LR:
    python guiu_pratt_pipeline.py --data-dir data --output-dir ./results_cosine --mode cosine

    # Option 2 -- sigma0 as extra input:
    python guiu_pratt_pipeline.py --data-dir data --output-dir ./results_sigma0 --mode sigma0_input

    # Option 3 -- normalize stress by sigma0:
    python guiu_pratt_pipeline.py --data-dir data --output-dir ./results_norm --mode normalized

    # Combined (Options 2+3):
    python guiu_pratt_pipeline.py --data-dir data --output-dir ./results_combined --mode combined

    # All modes in one shot:
    for mode in baseline cosine sigma0_input normalized combined; do
        python guiu_pratt_pipeline.py --data-dir data --output-dir ./results_$mode --mode $mode
    done
"""

from __future__ import annotations

import argparse
import math
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from material_split_utils import (
    extract_material, material_split, random_split,
    print_split_summary, UNKNOWN_MATERIAL,
)

# ---------------------------------------------------------------------------
#  Global config
# ---------------------------------------------------------------------------
RANDOM_SEED        = 42
N_TRAIN            = 10
N_TEST             = 7
SAMPLING_FRACTIONS = [0.10, 0.20, 0.30, 0.40]
N_SAMPLES          = 128
N_DRAWS            = 5
DRAW_SEEDS: List[int] = [RANDOM_SEED + 100 * (i + 1) for i in range(N_DRAWS)]
KERNEL_SIZE        = 3
DEFAULT_EPOCHS     = 150          # overridden to 300 for cosine mode
DEFAULT_LR         = 1e-3
SUPPORTED_EXTS     = {".lis", ".csv", ".xls", ".xlsx"}
EPS                = 1e-6

VALID_MODES   = ["baseline", "cosine", "sigma0_input", "normalized", "combined"]
VALID_SPLITS  = ["random", "material"]

MODE_DESCRIPTIONS = {
    "baseline":     "Original pipeline (ReduceLROnPlateau, raw stress input, 150 epochs)",
    "cosine":       "Option 1: CosineAnnealingLR + warmup, 300 epochs default",
    "sigma0_input": "Option 2: sigma0 injected as scalar into Dense head",
    "normalized":   "Option 3: stress input divided by sigma0 before Conv1D",
    "combined":     "Options 2+3: normalized input + sigma0 in Dense head",
}


# ---------------------------------------------------------------------------
#  Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===========================================================================
#  DATA LOADING
# ===========================================================================

TIME_KEYWORDS   = ["zeit", "time", "elapsed", "sec", "second", "seconds", "s"]
STRESS_KEYWORDS = ["spannung", "stress", "sigma", "mpa", "n/mm2", "nmm2"]


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.strip().lower()
    text = text.replace("\u00b5", "u").replace("\u00b2", "2")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text)
    return text


def to_float(value: object) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text:
        return np.nan
    text = text.replace("\u00a0", " ").strip().replace(" ", "")
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not m:
        return np.nan
    try:
        return float(m.group(0))
    except ValueError:
        return np.nan


def score_column(name: str, keywords: List[str], preferred: List[str]) -> int:
    score = 0
    for kw in keywords:
        if kw in name:
            score += 3
    for kw in preferred:
        if kw == name:
            score += 5
        elif kw in name:
            score += 2
    return score


def find_time_stress_columns(columns) -> Tuple[Optional[object], Optional[object]]:
    normalized    = [normalize_text(c) for c in columns]
    time_scores   = [score_column(c, TIME_KEYWORDS,   ["zeit", "time", "elapsed time"]) for c in normalized]
    stress_scores = [score_column(c, STRESS_KEYWORDS, ["spannung", "stress"])           for c in normalized]
    ti = int(np.argmax(time_scores))   if time_scores   else -1
    si = int(np.argmax(stress_scores)) if stress_scores else -1
    tc = columns[ti] if ti >= 0 and time_scores[ti]   > 0 else None
    sc = columns[si] if si >= 0 and stress_scores[si] > 0 else None
    return (None, None) if tc == sc else (tc, sc)


def maybe_skip_units_row(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    first_row     = [normalize_text(x) for x in df.iloc[0].tolist()]
    numeric_ratio = np.mean([not math.isnan(to_float(x)) for x in df.iloc[0].tolist()])
    units_like    = any(tok in " ".join(first_row) for tok in ["sec", "mpa", "%", "mm", "kn", "s"])
    return df.iloc[1:].copy() if (units_like and numeric_ratio < 0.5) else df


@dataclass
class RelaxationCurve:
    name:     str
    time:     np.ndarray
    stress:   np.ndarray
    material: str = UNKNOWN_MATERIAL   # parsed from Werkstoff header field

    @property
    def sigma0(self) -> float:
        return float(self.stress[0])


def _build_curve(df: pd.DataFrame, name: str,
                 material: str = UNKNOWN_MATERIAL) -> Optional[RelaxationCurve]:
    tc, sc = find_time_stress_columns(df.columns)
    if tc is None or sc is None:
        return None
    out = pd.DataFrame({"time": df[tc].map(to_float), "stress": df[sc].map(to_float)})
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out.drop_duplicates(subset=["time", "stress"])
    if len(out) < 10:
        return None
    out = out.sort_values("time").reset_index(drop=True)
    if (out["time"] < 0).any() and (out["time"] >= 0).any():
        out = out[out["time"] >= 0].reset_index(drop=True)
    else:
        out["time"] = out["time"] - out["time"].min()
    return RelaxationCurve(name=name,
                           time=out["time"].to_numpy(dtype=np.float64),
                           stress=out["stress"].to_numpy(dtype=np.float64),
                           material=material)


def _find_header(df_raw: pd.DataFrame, max_rows: int = 30) -> Optional[int]:
    best_idx, best_score = None, -1
    for idx in range(min(max_rows, len(df_raw))):
        row   = " | ".join(normalize_text(x) for x in df_raw.iloc[idx])
        score = 0
        if any(kw in row for kw in ["zeit", "time", "elapsed"]):   score += 3
        if any(kw in row for kw in ["spannung", "stress", "mpa"]): score += 3
        if score > best_score:
            best_score, best_idx = score, idx
    return best_idx if best_score > 0 else None


def load_lis(path: Path) -> Optional[RelaxationCurve]:
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            text = path.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        return None
    lines    = text.splitlines()
    data_idx = next((i for i, ln in enumerate(lines) if normalize_text(ln) == "[daten]"), None)
    if data_idx is None:
        return None

    header_lines = lines[:data_idx]
    material     = extract_material(header_lines)

    rows_after = [ln for ln in lines[data_idx + 1:] if ln.strip()]
    if len(rows_after) < 3:
        return None

    def split_tab(ln: str) -> List[str]:
        return [p.strip() for p in re.split(r"\t+", ln.strip()) if p.strip()]

    headers    = split_tab(rows_after[0])
    data_lines = rows_after[2:]
    parsed     = [split_tab(ln)[:len(headers)] for ln in data_lines
                  if len(split_tab(ln)) >= len(headers)]
    if not parsed:
        return None
    return _build_curve(pd.DataFrame(parsed, columns=headers), path.stem, material)


def load_csv(path: Path) -> Optional[RelaxationCurve]:
    for enc in ["utf-8", "latin-1", "cp1252"]:
        for sep in [None, ";", ",", "\t"]:
            try:
                kwargs = dict(header=None, dtype=str, engine="python", encoding=enc)
                df_raw = pd.read_csv(path, sep=sep if sep else None, **kwargs)
                hi = _find_header(df_raw)
                if hi is None:
                    continue
                df = df_raw.iloc[hi + 1:].copy().reset_index(drop=True)
                df.columns = df_raw.iloc[hi].tolist()
                df = maybe_skip_units_row(df)
                c  = _build_curve(df, path.stem)
                if c is not None:
                    return c
            except Exception:
                continue
    return None


def load_excel(path: Path) -> Optional[RelaxationCurve]:
    try:
        sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=str)
    except Exception:
        return None
    for sheet_name, df_raw in sheets.items():
        if df_raw is None or df_raw.empty:
            continue
        hi = _find_header(df_raw)
        if hi is None:
            continue
        df = df_raw.iloc[hi + 1:].copy().reset_index(drop=True)
        df.columns = df_raw.iloc[hi].tolist()
        df = maybe_skip_units_row(df)
        name = path.stem if len(sheets) == 1 else f"{path.stem}__{sheet_name}"
        c    = _build_curve(df, name)
        if c is not None:
            return c
    return None


def load_file(path: Path) -> Optional[RelaxationCurve]:
    ext = path.suffix.lower()
    if ext == ".lis":             return load_lis(path)
    if ext == ".csv":             return load_csv(path)
    if ext in {".xls", ".xlsx"}: return load_excel(path)
    return None


def load_dataset(data_dir: Path) -> List[RelaxationCurve]:
    files  = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS)
    curves = []
    for f in files:
        c = load_file(f)
        if c is not None:
            curves.append(c)
            print(f"  [LOADED] {f.name}  "
                  f"({len(c.time)} pts, sigma0={c.sigma0:.2f} MPa, "
                  f"material={c.material})")
        else:
            print(f"  [SKIP]   {f.name}")
    return curves


def do_split(curves: List[RelaxationCurve],
             split_mode: str,
             seed: int) -> Tuple[List[RelaxationCurve], List[RelaxationCurve]]:
    """Dispatch to material_split or random_split based on --split-mode."""
    if split_mode == "material":
        return material_split(curves, seed)
    return random_split(curves, n_train=N_TRAIN, seed=seed)


# ===========================================================================
#  NEURAL NETWORK  (mode-aware)
# ===========================================================================

class GuiuPrattNet(nn.Module):
    """
    Conv1D network -> (E, xi, theta).

    mode controls input preprocessing and head architecture:

    baseline / cosine:
        Input  : raw stress values  (128,)
        Head   : Dense(32) -> Dense(3)

    sigma0_input:
        Input  : raw stress values  (128,)
        Head   : Dense(32+1) -> Dense(32) -> Dense(3)
                 sigma0 concatenated to GAP output before Dense layers

    normalized:
        Input  : stress / sigma0    (128,)  -- shape only, magnitude removed
        Head   : Dense(32) -> Dense(3)

    combined:
        Input  : stress / sigma0    (128,)
        Head   : Dense(32+1) -> Dense(32) -> Dense(3)
                 sigma0 concatenated to GAP output
    """

    def __init__(self, mode: str, kernel_size: int = KERNEL_SIZE):
        super().__init__()
        self.mode = mode
        use_sigma0_in_head = mode in ("sigma0_input", "combined")

        self.conv_block = nn.Sequential(
            nn.Conv1d(1,  16, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
        )

        # GAP output is 32-dim; optionally append sigma0 scalar -> 33-dim
        head_in = 33 if use_sigma0_in_head else 32
        self.head = nn.Sequential(
            nn.Linear(head_in, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x: torch.Tensor,
                sigma0_scalar: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x             : (B, 128) -- raw or normalised stress values
        sigma0_scalar : (B, 1)   -- required for sigma0_input / combined modes
        """
        x = x.unsqueeze(1)         # (B, 1, 128)
        x = self.conv_block(x)     # (B, 32, 128)
        x = x.mean(dim=2)          # Global Average Pooling -> (B, 32)

        if self.mode in ("sigma0_input", "combined"):
            assert sigma0_scalar is not None, \
                f"sigma0_scalar required for mode={self.mode}"
            x = torch.cat([x, sigma0_scalar], dim=1)   # (B, 33)

        return self.head(x)        # (B, 3)


# ===========================================================================
#  PHYSICS LAYER
# ===========================================================================

def guiu_pratt(sigma0: float,
               E:      torch.Tensor,
               xi:     torch.Tensor,
               theta:  torch.Tensor,
               t:      torch.Tensor) -> torch.Tensor:
    """
    sigma(t) = sigma0 - beta * ln(1 + t / tau_sigma)
    beta      = (sigma0 / E) * theta
    tau_sigma = xi / theta   (always > 0 via softplus)
    """
    xi_pos    = nn.functional.softplus(xi)    + EPS
    theta_pos = nn.functional.softplus(theta) + EPS
    beta      = (sigma0 / (E + EPS)) * theta_pos
    tau       = xi_pos / theta_pos
    return sigma0 - beta * torch.log(1.0 + t / tau)


def physics_predict(sigma0: float,
                    params: torch.Tensor,
                    t_full: torch.Tensor) -> torch.Tensor:
    E, xi, theta = params[0:1], params[1:2], params[2:3]
    return guiu_pratt(sigma0, E, xi, theta, t_full)


def decode_params(params: torch.Tensor,
                  sigma0: float) -> Tuple[float, float, float, float, float]:
    """Returns (E_raw, xi_raw, theta_raw, beta, tau_sigma)."""
    E_raw     = params[0].item()
    xi_pos    = float(nn.functional.softplus(params[1])) + EPS
    theta_pos = float(nn.functional.softplus(params[2])) + EPS
    beta      = (sigma0 / (E_raw + EPS)) * theta_pos
    tau       = xi_pos / theta_pos
    return E_raw, params[1].item(), params[2].item(), beta, tau


# ===========================================================================
#  INPUT PREPARATION  (mode-aware)
# ===========================================================================

def prepare_input(sampled_stress: np.ndarray,
                  sigma0: float,
                  mode: str,
                  device: torch.device
                  ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns (x, sigma0_tensor) ready for GuiuPrattNet.forward().

    baseline / cosine:
        x = raw stress  (1, 128)
        sigma0_tensor = None

    sigma0_input:
        x = raw stress  (1, 128)
        sigma0_tensor = [[sigma0]]  (1, 1)

    normalized:
        x = stress / sigma0  (1, 128)
        sigma0_tensor = None

    combined:
        x = stress / sigma0  (1, 128)
        sigma0_tensor = [[sigma0]]  (1, 1)
    """
    if mode in ("normalized", "combined"):
        values = sampled_stress / (sigma0 + EPS)
    else:
        values = sampled_stress

    x = torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0)

    if mode in ("sigma0_input", "combined"):
        s0t = torch.tensor([[sigma0]], dtype=torch.float32, device=device)
    else:
        s0t = None

    return x, s0t


# ===========================================================================
#  SAMPLING HELPERS
# ===========================================================================

def partial_curve(curve: RelaxationCurve, fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    n = max(2, int(math.ceil(len(curve.time) * fraction)))
    return curve.time[:n], curve.stress[:n]


def random_sample_stress(stress_partial: np.ndarray, n: int,
                         rng: np.random.Generator) -> np.ndarray:
    replace = len(stress_partial) < n
    return stress_partial[rng.choice(len(stress_partial), size=n, replace=replace)]


# ===========================================================================
#  TRAINING & EVALUATION
# ===========================================================================

def run_epoch(model:     GuiuPrattNet,
              curves:    List[RelaxationCurve],
              fractions: List[float],
              draw_rngs: List[np.random.Generator],
              optimizer: Optional[optim.Optimizer],
              device:    torch.device,
              train:     bool) -> Dict[float, float]:
    """
    One epoch over all curves x all fractions.
    For each (curve, frac), N_DRAWS=5 draws are averaged before backprop.
    """
    model.train(train)
    losses_per_frac: Dict[float, List[float]] = {f: [] for f in fractions}

    for curve in curves:
        t_full = torch.tensor(curve.time,   dtype=torch.float32, device=device)
        s_full = torch.tensor(curve.stress, dtype=torch.float32, device=device)
        sigma0 = curve.sigma0

        for frac in fractions:
            _, stress_partial = partial_curve(curve, frac)

            draw_losses: List[torch.Tensor] = []
            for rng in draw_rngs:
                sampled      = random_sample_stress(stress_partial, N_SAMPLES, rng)
                x, s0_tensor = prepare_input(sampled, sigma0, model.mode, device)

                with torch.set_grad_enabled(train):
                    params = model(x, s0_tensor).squeeze(0)
                    s_pred = physics_predict(sigma0, params, t_full)
                    loss   = nn.functional.mse_loss(s_pred, s_full)

                draw_losses.append(loss)

            mean_loss = torch.stack(draw_losses).mean()

            if train:
                optimizer.zero_grad()
                mean_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            losses_per_frac[frac].append(mean_loss.item())

    return {f: float(np.mean(v)) for f, v in losses_per_frac.items()}


# ===========================================================================
#  INFERENCE
# ===========================================================================

def infer_curve(model:     GuiuPrattNet,
                curve:     RelaxationCurve,
                frac:      float,
                draw_rngs: List[np.random.Generator],
                device:    torch.device
                ) -> Tuple[np.ndarray, float, float, float, float, float]:
    """
    Inference averaged over N_DRAWS draws.
    Returns (s_pred_mean, E_raw, xi_raw, theta_raw, beta, tau_sigma).
    """
    model.eval()
    t_full = torch.tensor(curve.time, dtype=torch.float32, device=device)
    sigma0 = curve.sigma0
    _, stress_partial = partial_curve(curve, frac)

    all_preds  = []
    all_params = {"E": [], "xi": [], "theta": [], "beta": [], "tau": []}

    with torch.no_grad():
        for rng in draw_rngs:
            sampled      = random_sample_stress(stress_partial, N_SAMPLES, rng)
            x, s0_tensor = prepare_input(sampled, sigma0, model.mode, device)
            params       = model(x, s0_tensor).squeeze(0)
            s_pred       = physics_predict(sigma0, params, t_full).cpu().numpy()
            E_r, xi_r, theta_r, beta, tau = decode_params(params, sigma0)

            all_preds.append(s_pred)
            all_params["E"].append(E_r);    all_params["xi"].append(xi_r)
            all_params["theta"].append(theta_r)
            all_params["beta"].append(beta); all_params["tau"].append(tau)

    return (np.mean(all_preds, axis=0),
            float(np.mean(all_params["E"])),
            float(np.mean(all_params["xi"])),
            float(np.mean(all_params["theta"])),
            float(np.mean(all_params["beta"])),
            float(np.mean(all_params["tau"])))


# ===========================================================================
#  SCHEDULER FACTORY  (mode-aware)
# ===========================================================================

def build_scheduler(optimizer: optim.Optimizer,
                    mode: str,
                    n_epochs: int) -> object:
    """
    baseline     -> ReduceLROnPlateau (patience=15, factor=0.5)
    cosine       -> LinearLR warmup (10 epochs) + CosineAnnealingLR
    sigma0_input -> ReduceLROnPlateau
    normalized   -> ReduceLROnPlateau
    combined     -> ReduceLROnPlateau
    """
    if mode == "cosine":
        warmup    = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=10)
        cosine    = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs - 10, eta_min=1e-5)
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[10])
        return scheduler
    else:
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=15, factor=0.5)


def step_scheduler(scheduler: object, mean_train_loss: float, epoch: int) -> None:
    """Step the scheduler correctly depending on its type."""
    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(mean_train_loss)
    else:
        scheduler.step()


# ===========================================================================
#  PLOTTING HELPERS
# ===========================================================================

def plot_predictions(model:      GuiuPrattNet,
                     curves:     List[RelaxationCurve],
                     fractions:  List[float],
                     draw_rngs:  List[np.random.Generator],
                     device:     torch.device,
                     output_dir: Path,
                     split_tag:  str,
                     mode:       str) -> None:
    pred_dir = output_dir / "predictions" / split_tag
    pred_dir.mkdir(parents=True, exist_ok=True)

    for curve in curves:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
        axes = axes.flatten()

        for ax, frac in zip(axes, fractions):
            s_pred, E_r, xi_r, theta_r, beta, tau = infer_curve(
                model, curve, frac, draw_rngs, device)

            mse       = float(np.mean((s_pred - curve.stress) ** 2))
            log_t     = np.log10(curve.time + 1.0)
            partial_n = max(2, int(math.ceil(len(curve.time) * frac)))

            ax.plot(log_t, curve.stress, color="steelblue", lw=1.6, label="Actual")
            ax.plot(log_t, s_pred,       color="tomato",    lw=1.6, linestyle="--",
                    label=f"Predicted (avg {N_DRAWS} draws)")
            ax.axvline(log_t[partial_n - 1], color="gray", lw=1.0, linestyle=":",
                       label=f"Input cutoff ({int(frac*100)}%)")

            ax.set_title(
                f"Sampling {int(frac*100)}%  |  MSE={mse:.4f}\n"
                f"E={E_r:.4f}  beta={beta:.4f}  tau_sigma={tau:.4f}",
                fontsize=8)
            ax.set_xlabel("log10(t+1)  [s]", fontsize=8)
            ax.set_ylabel("Stress sigma (MPa)", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"{curve.name}  --  {split_tag.upper()}  [mode: {mode}]",
                     fontsize=10, fontweight="bold")
        fig.tight_layout()
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", curve.name)
        fig.savefig(pred_dir / f"{safe}_predictions.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_loss_curves(train_history: Dict[float, List[float]],
                     test_history:  Dict[float, List[float]],
                     output_dir:    Path,
                     mode:          str) -> None:
    loss_dir = output_dir / "loss_curves"
    loss_dir.mkdir(parents=True, exist_ok=True)

    fractions = sorted(train_history.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for ax, frac in zip(axes, fractions):
        tr     = train_history[frac]
        te     = test_history[frac]
        epochs = range(1, len(tr) + 1)
        ax.plot(epochs, tr, label="Train", color="steelblue", lw=1.6)
        ax.plot(epochs, te, label="Test",  color="tomato",    lw=1.6, linestyle="--")
        ax.set_title(f"Sampling {int(frac*100)}%", fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"MSE Loss (avg {N_DRAWS} draws)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    fig.suptitle(
        f"Train vs Test Loss -- {mode}  "
        f"({N_DRAWS} draws x {N_SAMPLES} samples)",
        fontsize=10, fontweight="bold")
    fig.tight_layout()
    out = loss_dir / "loss_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] Loss curves -> {out}")


def save_param_table(model:      GuiuPrattNet,
                     curves:     List[RelaxationCurve],
                     fractions:  List[float],
                     draw_rngs:  List[np.random.Generator],
                     device:     torch.device,
                     output_dir: Path,
                     split_tag:  str) -> None:
    rows = []
    for curve in curves:
        for frac in fractions:
            s_pred, E_r, xi_r, theta_r, beta, tau = infer_curve(
                model, curve, frac, draw_rngs, device)
            mse = float(np.mean((s_pred - curve.stress) ** 2))
            rows.append({
                "split":         split_tag,
                "file":          curve.name,
                "material":      curve.material,
                "fraction":      f"{int(frac*100)}%",
                "mode":          model.mode,
                "sigma0":        round(curve.sigma0,  6),
                "E_raw_avg":     round(E_r,    6),
                "xi_raw_avg":    round(xi_r,   6),
                "theta_raw_avg": round(theta_r, 6),
                "beta_avg":      round(beta,   6),
                "tau_sigma_avg": round(tau,    6),
                "mse_avg":       round(mse,    6),
                "n_draws":       N_DRAWS,
            })

    df       = pd.DataFrame(rows)
    out_path = output_dir / f"parameters_{split_tag}.csv"
    df.to_csv(out_path, index=False)
    print(f"  [SAVED] Parameter table -> {out_path}")


# ===========================================================================
#  MAIN
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Guiu-Pratt PINN pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {k:15s}: {v}" for k, v in MODE_DESCRIPTIONS.items()))
    parser.add_argument("--data-dir",    required=True)
    parser.add_argument("--output-dir",  default="./results")
    parser.add_argument("--mode",        default="baseline", choices=VALID_MODES,
                        help="Which improvement option to use (default: baseline)")
    parser.add_argument("--split-mode",  default="random", choices=VALID_SPLITS,
                        help="'random' = 10 train / 7 test (default); "
                             "'material' = 1 file per unique Werkstoff -> train, rest -> test")
    parser.add_argument("--epochs",      type=int,   default=None,
                        help="Training epochs. Default: 300 for cosine, 150 for others.")
    parser.add_argument("--lr",          type=float, default=DEFAULT_LR)
    parser.add_argument("--seed",        type=int,   default=RANDOM_SEED)
    args = parser.parse_args()

    # Mode-specific defaults
    n_epochs = args.epochs if args.epochs is not None else (300 if args.mode == "cosine" else 150)

    global DRAW_SEEDS
    DRAW_SEEDS = [args.seed + 100 * (i + 1) for i in range(N_DRAWS)]

    set_seed(args.seed)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Mode        : {args.mode}")
    print(f"  Description : {MODE_DESCRIPTIONS[args.mode]}")
    print(f"  Epochs      : {n_epochs}")
    print(f"  LR          : {args.lr}")
    print(f"  Master seed : {args.seed}")
    print(f"  Draw seeds  : {DRAW_SEEDS}")
    print(f"  Device      : {device}")
    print(f"  Split mode  : {args.split_mode}")
    print(f"{'='*60}")

    # ── 1. Load data ────────────────────────────────────────────
    print("\n-- Loading dataset --")
    data_dir = Path(args.data_dir).expanduser().resolve()
    curves   = load_dataset(data_dir)

    if args.split_mode == "random" and len(curves) < N_TRAIN + N_TEST:
        print(f"[ERROR] random split needs at least {N_TRAIN + N_TEST} files, "
              f"found {len(curves)}.")
        return 1
    if len(curves) < 2:
        print("[ERROR] Need at least 2 files.")
        return 1

    train_curves, test_curves = do_split(curves, args.split_mode, args.seed)
    print_split_summary(train_curves, test_curves, args.split_mode)

    # ── 2. Build model, optimizer, scheduler ────────────────────
    draw_rngs = [np.random.default_rng(s) for s in DRAW_SEEDS]
    model     = GuiuPrattNet(mode=args.mode, kernel_size=KERNEL_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = build_scheduler(optimizer, args.mode, n_epochs)

    train_history: Dict[float, List[float]] = {f: [] for f in SAMPLING_FRACTIONS}
    test_history:  Dict[float, List[float]] = {f: [] for f in SAMPLING_FRACTIONS}

    # ── 3. Training loop ─────────────────────────────────────────
    print(f"\n-- Training ({args.mode}) for {n_epochs} epochs on {device} --")
    for epoch in range(1, n_epochs + 1):
        train_losses = run_epoch(model, train_curves, SAMPLING_FRACTIONS,
                                 draw_rngs, optimizer, device, train=True)
        test_losses  = run_epoch(model, test_curves,  SAMPLING_FRACTIONS,
                                 draw_rngs, None,      device, train=False)

        for f in SAMPLING_FRACTIONS:
            train_history[f].append(train_losses[f])
            test_history[f].append(test_losses[f])

        mean_train = float(np.mean(list(train_losses.values())))
        step_scheduler(scheduler, mean_train, epoch)

        if epoch % 10 == 0 or epoch == 1:
            lr_now   = optimizer.param_groups[0]["lr"]
            frac_str = "  ".join(
                f"{int(f*100)}%: tr={train_losses[f]:.4f} te={test_losses[f]:.4f}"
                for f in SAMPLING_FRACTIONS)
            print(f"  Epoch {epoch:4d}/{n_epochs}  lr={lr_now:.2e}  |  {frac_str}")

    # ── 4. Save outputs ──────────────────────────────────────────
    print("\n-- Saving outputs --")
    plot_loss_curves(train_history, test_history, output_dir, args.mode)

    infer_rngs = [np.random.default_rng(s) for s in DRAW_SEEDS]
    print("  Saving prediction plots ...")
    plot_predictions(model, train_curves, SAMPLING_FRACTIONS,
                     infer_rngs, device, output_dir, "train", args.mode)

    infer_rngs = [np.random.default_rng(s) for s in DRAW_SEEDS]
    plot_predictions(model, test_curves,  SAMPLING_FRACTIONS,
                     infer_rngs, device, output_dir, "test",  args.mode)

    infer_rngs = [np.random.default_rng(s) for s in DRAW_SEEDS]
    save_param_table(model, train_curves, SAMPLING_FRACTIONS,
                     infer_rngs, device, output_dir, "train")

    infer_rngs = [np.random.default_rng(s) for s in DRAW_SEEDS]
    save_param_table(model, test_curves,  SAMPLING_FRACTIONS,
                     infer_rngs, device, output_dir, "test")

    torch.save({
        "epoch":       n_epochs,
        "mode":        args.mode,
        "split_mode":  args.split_mode,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "train_split": [c.name for c in train_curves],
        "test_split":  [c.name for c in test_curves],
        "seed":        args.seed,
        "draw_seeds":  DRAW_SEEDS,
    }, output_dir / "model_checkpoint.pt")
    print(f"  [SAVED] Checkpoint -> {output_dir / 'model_checkpoint.pt'}")

    print(f"\nDone. Outputs -> {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
