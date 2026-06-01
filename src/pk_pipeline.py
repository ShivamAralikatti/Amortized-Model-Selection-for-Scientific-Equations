#!/usr/bin/env python3
"""
pk_pipeline.py — Theophylline PK data loader
=============================================

This module is the pharmacokinetics analog of guiu_pratt_pipeline.py.
It exposes:

    class PKCurve            — one subject's concentration-time series
    function load_curves()   — load all 12 subjects from a CSV

Expected CSV format (the theophylline data):

    Subject,Wt,Dose,Time,conc
    1,79.6,4.02,0.00,0.74
    1,79.6,4.02,0.25,2.84
    ...

Columns:
    Subject : integer 1..12, subject identifier
    Wt      : body weight in kg
    Dose    : dose per kg (mg/kg)
    Time    : hours since administration
    conc    : serum theophylline concentration in mg/L

How to obtain the CSV
---------------------
In R:
    write.csv(Theoph, "theoph.csv", row.names = FALSE)

The dataset ships with base R; nothing extra to install. The CSV is also
available from the Rdatasets project on GitHub.

A built-in verification check confirms Subject 1's values match two
independent published references (rdocumentation.org and
dpastoor.github.io/r-pharmsciences). If the verification fails the loader
raises an error and prints which row disagreed.

Mapping to the rheology RelaxationCurve interface (cv_threeway.py)
-------------------------------------------------------------------
This module exposes the same attribute names as RelaxationCurve, so
cv_threeway.py works with no changes except a one-line swap of the
import statement. The mapping is:

    rheology                        →  pharmacokinetics
    --------                           ----------------
    time     (seconds)              →  time     (hours)
    stress   (MPa)                  →  stress   (mg/L)         -- reused for conc
    material (Werkstoff str)        →  material (subject str)
    sigma0   (initial stress, MPa)  →  sigma0   (Dose * Wt mg) -- total dose
    temperature_K (K)               →  temperature_K (held = 310.15 K, ignored)

The reuse of attribute names is deliberate: it lets the existing CV
framework treat both domains uniformly. The substantive interpretation
is determined by which physics layer is plugged in (Guiu-Pratt vs.
thermal for rheology; one-compartment vs. two-compartment for PK).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List


# Verification fixture: Subject 1 values, double-verified against
# two independent published sources.
_SUBJECT_1_VERIFY = [
    # (Time hr, conc mg/L)
    (0.00,  0.74),
    (0.25,  2.84),
    (0.57,  6.57),
    (1.12, 10.50),
    (2.02,  9.66),
    (3.82,  8.58),
    (5.10,  8.36),
    (7.03,  7.47),
    (9.05,  6.89),
    (12.12, 5.94),
    (24.37, 3.28),
]
_SUBJECT_1_WT_KG    = 79.6
_SUBJECT_1_DOSE_MGKG = 4.02


class PKCurve:
    """One subject's theophylline concentration-time series.

    Attributes mirror RelaxationCurve so the existing CV framework works
    without modification:

        time           : 1D numpy array of time points (hours)
        stress         : 1D numpy array of concentrations (mg/L) -- reused name
        material       : str, subject identifier ("Subj_01" .. "Subj_12")
        sigma0         : float, TOTAL dose in mg (Dose_per_kg * Wt_kg)
                         -- this is the analog of the initial stress and
                         appears as a multiplicative prefactor in the
                         one-compartment PK equation.
        temperature_K  : float, held at body temperature 310.15 K and ignored
                         by the PK physics layers. Kept only for interface
                         compatibility with rheology code.

    Extra PK-specific attributes (not required by cv_threeway.py):

        weight_kg      : float, subject body weight
        dose_per_kg    : float, mg per kg administered
        subject_id     : int, 1..12
        name           : str, filename-like identifier for logging
    """

    __slots__ = ("time", "stress", "material", "sigma0", "temperature_K",
                 "weight_kg", "dose_per_kg", "subject_id", "name")

    def __init__(self, subject_id: int, weight_kg: float, dose_per_kg: float,
                 times, concentrations):
        import numpy as np
        self.subject_id    = int(subject_id)
        self.weight_kg     = float(weight_kg)
        self.dose_per_kg   = float(dose_per_kg)
        self.time          = np.asarray(times,          dtype=np.float64)
        self.stress        = np.asarray(concentrations, dtype=np.float64)
        self.material      = f"Subj_{self.subject_id:02d}"
        self.sigma0        = self.dose_per_kg * self.weight_kg   # total dose in mg
        self.temperature_K = 310.15
        self.name          = f"theoph_subject_{self.subject_id:02d}"

    def __repr__(self) -> str:
        return (f"PKCurve(subject={self.subject_id}, "
                f"n_points={len(self.time)}, dose_total={self.sigma0:.2f} mg, "
                f"Cmax={self.stress.max():.2f} mg/L)")


def _verify_subject_1(rows_for_subject_1) -> None:
    """Sanity check Subject 1's (Time, conc) values against published refs.

    Raises ValueError with a clear message if any cell disagrees beyond
    a tiny floating-point tolerance.
    """
    if len(rows_for_subject_1) != len(_SUBJECT_1_VERIFY):
        raise ValueError(
            f"Subject 1 should have {len(_SUBJECT_1_VERIFY)} rows in the "
            f"published Theoph dataset, but found {len(rows_for_subject_1)}."
        )
    for i, (row, (ref_t, ref_c)) in enumerate(zip(rows_for_subject_1,
                                                   _SUBJECT_1_VERIFY)):
        t, c = float(row["Time"]), float(row["conc"])
        if abs(t - ref_t) > 1e-6 or abs(c - ref_c) > 1e-6:
            raise ValueError(
                f"Subject 1 row {i+1} mismatch:\n"
                f"  expected (Time={ref_t}, conc={ref_c})\n"
                f"  found    (Time={t}, conc={c})\n"
                f"Loaded CSV does not match the canonical Theoph dataset. "
                f"Re-export via R: write.csv(Theoph, 'theoph.csv', row.names=FALSE)"
            )


def load_curves(csv_path) -> List[PKCurve]:
    """Load the theophylline dataset from a CSV file.

    The CSV must have columns Subject, Wt, Dose, Time, conc (case-sensitive,
    in any order). All other columns are ignored. Output is a list of
    PKCurve objects, one per subject, ordered by subject ID.

    Subject 1's values are verified against two independent published
    references; a mismatch raises ValueError.

    Parameters
    ----------
    csv_path : str or Path or directory containing 'theoph.csv'
        If a directory is given, the loader looks for 'theoph.csv' inside.
    """
    p = Path(csv_path)
    if p.is_dir():
        candidate = p / "theoph.csv"
        if not candidate.exists():
            raise FileNotFoundError(
                f"Expected 'theoph.csv' in directory {p}. "
                f"Generate with R: write.csv(Theoph, 'theoph.csv', row.names=FALSE)"
            )
        p = candidate

    # Read all rows, group by Subject
    by_subj = {}
    with open(p, newline="") as f:
        reader = csv.DictReader(f)
        required = {"Subject", "Wt", "Dose", "Time", "conc"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"CSV missing required columns: {sorted(missing)}. "
                f"Found columns: {reader.fieldnames}"
            )
        for row in reader:
            sid = int(float(row["Subject"]))   # tolerate '1', '1.0', '"1"'
            by_subj.setdefault(sid, []).append({
                "Wt":   float(row["Wt"]),
                "Dose": float(row["Dose"]),
                "Time": float(row["Time"]),
                "conc": float(row["conc"]),
            })

    if not by_subj:
        raise ValueError(f"No rows found in {p}.")
    if 1 not in by_subj:
        raise ValueError(
            f"Subject 1 not found in {p}. Found subjects: {sorted(by_subj)}."
        )

    # Verify Subject 1 against the published reference
    _verify_subject_1(sorted(by_subj[1], key=lambda r: r["Time"]))

    # Build PKCurve objects
    curves = []
    for sid in sorted(by_subj):
        rows = sorted(by_subj[sid], key=lambda r: r["Time"])
        wt   = rows[0]["Wt"]
        dose = rows[0]["Dose"]
        # Sanity check: weight and dose should be constant within a subject
        for r in rows:
            if r["Wt"] != wt or r["Dose"] != dose:
                raise ValueError(
                    f"Subject {sid} has non-constant Wt or Dose across rows."
                )
        curves.append(PKCurve(
            subject_id   = sid,
            weight_kg    = wt,
            dose_per_kg  = dose,
            times        = [r["Time"] for r in rows],
            concentrations = [r["conc"] for r in rows],
        ))
    return curves


# ---------------------------------------------------------------------------
#  CLI: quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "theoph.csv"
    try:
        curves = load_curves(path)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr); sys.exit(1)
    print(f"Loaded {len(curves)} subjects from {path}")
    print(f"Verification check on Subject 1: PASSED")
    for c in curves:
        print(f"  {c}")
