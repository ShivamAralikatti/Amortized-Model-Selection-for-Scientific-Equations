# Amortized Model Selection for Scientific Equations

**A Predictive Validation Procedure with Cross-Domain Evidence**

This repository contains the code and experimental pipeline for the paper **“Amortized Model Selection for Scientific Equations: A Predictive Validation Procedure with Cross-Domain Evidence.”**

The project studies how to choose between competing scientific equations when only a short early portion of a time-series curve is available. Instead of fitting each candidate equation separately to the test curve, the proposed method trains a neural network to predict equation parameters from subject-level information alone, then ranks equations by their long-horizon prediction error.

---

## Authors

* **Manoranjan Dash**
  FLAME University, Pune, India
  `manoranjan.dash@flame.edu.in`

* **Shivam Aralikatti**
  FLAME University, Pune, India
  `shivam.aralikatti@flame.edu.in`

---

## Overview

Many scientific processes are represented by time-series curves:

* stress relaxation in metals,
* drug concentration over time,
* voltage decay,
* biological response curves,
* mechanical or material degradation curves.

Often, multiple equations can fit the same observed data. Classical model selection usually fits each equation to the observed part of a curve and compares residual error, AIC, or BIC. This works when data is dense, but it can fail when only a short early segment of the curve is available.

This repository implements a different selection criterion:

> **Choose the equation whose parameters can be predicted from population-level conditions and whose predictions generalize best to the unobserved long-horizon curve.**

The method is based on **amortized inference**. A small neural network is trained once across a population of curves. At test time, it predicts parameters for a held-out curve without observing any points from that curve.

---

## Core Idea

For each candidate equation, we train a separate neural network:

```text
subject descriptor + driving condition  --->  MLP  --->  equation parameters
```

The predicted parameters are passed into a differentiable physics layer that implements the candidate equation.

The candidate equations are then compared using full-curve held-out mean squared error.

```text
Candidate equation A  --->  predicted parameters  --->  full-curve prediction  --->  MSE
Candidate equation B  --->  predicted parameters  --->  full-curve prediction  --->  MSE
```

The equation with lower long-horizon error is preferred.

---

## Key Concept: Driving-Condition Coupling

The paper identifies a structural property called **driving-condition coupling**.

An equation is said to have driving-condition coupling when the curve-level driving condition appears multiplicatively inside the equation’s parameter structure.

Examples:

* In stress relaxation, the driving condition is the initial stress `sigma0`.
* In pharmacokinetics, the driving condition is the administered dose `D`.

The central empirical finding is:

> Equations whose parameter structure embeds the driving condition multiplicatively are easier to learn through amortized inference and generalize better to unseen curves.

---

## Domains Studied

This repository contains experiments from two scientific domains.

---

### 1. Stress Relaxation in Copper Alloys

The first domain studies stress relaxation curves in copper alloys.

The candidate equations are:

| Equation                              | Script Label | Description                                       |
| ------------------------------------- | ------------ | ------------------------------------------------- |
| Guiu-Pratt logarithmic relaxation law | `guiu_pratt` | Coupled to initial stress                         |
| Thermally activated relaxation law    | `thermal`    | Does not contain the same multiplicative coupling |

The Guiu-Pratt equation has the form:

```text
sigma(t) = sigma0 - beta * log(1 + t / tau_sigma)
```

where the amplitude term depends on the initial stress:

```text
beta = sigma0 / (E * theta_j)
```

This makes the equation structurally suitable for amortized inference.

---

### 2. Theophylline Pharmacokinetics

The second domain studies concentration-time curves from the theophylline pharmacokinetic dataset.

The candidate equations are:

| Equation              | Script Label      | Parameters                       |
| --------------------- | ----------------- | -------------------------------- |
| One-compartment model | `one_compartment` | `V`, `ka`, `ke`                  |
| Two-compartment model | `two_compartment` | `ka`, `alpha`, `extra`, `A_frac` |

The one-compartment model is:

```text
C(t) = D * ka / (V * (ka - ke)) * (exp(-ke*t) - exp(-ka*t))
```

Here, the administered dose `D` appears as a multiplicative prefactor, making the model suitable for amortized prediction.

---

## Three-Way Evaluation

Each candidate equation is evaluated using three estimators.

---

### 1. No-Input MLP

The MLP receives only subject-level information and the driving condition.

It does **not** receive any observations from the held-out test curve.

For rheology:

```text
MLP input = one-hot material identity + normalized initial stress
```

For pharmacokinetics, two variants are supported:

```text
Subject-ID variant:
MLP input = one-hot subject ID + normalized dose

Population variant:
MLP input = normalized body weight + normalized dose
```

Because the MLP receives no test-curve observations, its prediction is constant across all observation fractions.

---

### 2. TRF Cold-Start

The Trust Region Reflective optimizer fits the candidate equation directly to the first observed fraction of the test curve.

It uses:

* test-curve observations,
* no population information,
* multiple random restarts.

This is the classical nonlinear least-squares baseline.

---

### 3. TRF Warm-Start

TRF warm-start uses the MLP-predicted parameters as the initialization for nonlinear least squares.

It combines:

* population-level information from the MLP,
* test-curve observations from the observed fraction.

---

## Observation Fraction

The observation fraction controls how much of the test curve TRF is allowed to see.

For example:

```text
fraction = 0.10
```

means TRF sees only the first 10% of the test curve.

The MLP does not use the test curve, so its result is independent of this fraction.

---

## Crossover Fraction

The **crossover fraction** is the smallest observation fraction at which TRF beats the no-input MLP.

Interpretation:

| Region          | Preferred Method                   |
| --------------- | ---------------------------------- |
| Below crossover | No-input MLP / amortized inference |
| Above crossover | TRF / direct curve fitting         |

This gives a practical recommendation:

> Use amortized inference when only a very short early segment is available. Use direct fitting once enough of the curve has been observed.

---

## Repository Structure

A recommended repository layout is:

```text
.
├── README.md
├── requirements.txt
├── paper/
│   └── IEEE_ICDM_2026.pdf
├── data/
│   ├── rheology/
│   │   └── stress_relaxation_files can be found at: https://dataportal.material-digital.de/dataset/?q=StressRelaxation&sort=score+desc%2C+metadata_modified+desc
│   └── theoph.csv
├── src/
│   ├── cv_threeway.py
│   ├── cv_threeway_pk.py
│   ├── guiu_pratt_pipeline.py
│   ├── pk_pipeline.py
│   └── pk_physics.py
└── results/
    ├── rheology/
    └── pharmacokinetics/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ShivamAralikatti/Amortized-Model-Selection-for-Scientific-Equations.git
cd Amortized-Model-Selection-for-Scientific-Equations
```

Create a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Or with Conda:

```bash
conda create -n amortized-equations python=3.10 -y
conda activate amortized-equations
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Requirements

A minimal `requirements.txt` should contain:

```text
numpy
pandas
scipy
torch
openpyxl
matplotlib
```

Depending on your data-loading scripts, you may also need:

```text
scikit-learn
tqdm
```

---

## Running the Rheology Experiments

The rheology experiment is implemented in:

```text
src/cv_threeway.py
```

Run leave-one-out cross-validation over all curves:

```bash
python src/cv_threeway.py \
  --data-dir data/rheology \
  --output-dir results/rheology_loo \
  --cv-mode loo \
  --split-variant all \
  --model both
```

Run only the Guiu-Pratt model:

```bash
python src/cv_threeway.py \
  --data-dir data/rheology \
  --output-dir results/rheology_guiu_pratt \
  --cv-mode loo \
  --split-variant all \
  --model guiu_pratt
```

Run only the thermal model:

```bash
python src/cv_threeway.py \
  --data-dir data/rheology \
  --output-dir results/rheology_thermal \
  --cv-mode loo \
  --split-variant all \
  --model thermal
```

Run with custom TRF observation fractions:

```bash
python src/cv_threeway.py \
  --data-dir data/rheology \
  --output-dir results/rheology_custom_fractions \
  --cv-mode loo \
  --split-variant duplicates \
  --trf-fractions 0.001,0.005,0.01,0.02,0.05,0.10,0.20,0.40
```

---

## Running the Pharmacokinetics Experiments

The pharmacokinetics experiment is implemented in:

```text
src/cv_threeway_pk.py
```

The recommended command runs all PK variants across multiple seeds:

```bash
python src/cv_threeway_pk.py \
  --csv data/theoph.csv \
  --all-variants \
  --output-dir results/pk_all
```

This runs:

```text
subjid_sigma0      = one-hot subject ID + dose
population_sigma0  = body weight + dose
population_nosigma0 = body weight only
```

Run the subject-ID variant:

```bash
python src/cv_threeway_pk.py \
  --csv data/theoph.csv \
  --output-dir results/pk_subject_id
```

Run the population variant:

```bash
python src/cv_threeway_pk.py \
  --csv data/theoph.csv \
  --no-subject-id \
  --output-dir results/pk_population
```

Run the population variant without dose:

```bash
python src/cv_threeway_pk.py \
  --csv data/theoph.csv \
  --no-subject-id \
  --no-sigma0 \
  --output-dir results/pk_population_no_dose
```

Run only the one-compartment model:

```bash
python src/cv_threeway_pk.py \
  --csv data/theoph.csv \
  --model one_compartment \
  --output-dir results/pk_one_compartment
```

Run only the two-compartment model:

```bash
python src/cv_threeway_pk.py \
  --csv data/theoph.csv \
  --model two_compartment \
  --output-dir results/pk_two_compartment
```

---

## Important Methodological Note

The MLP does **not** use the observation fraction.

This is intentional.

The observation fraction belongs only to the TRF baselines. It controls how much of the test curve TRF can observe.

The MLP is trained on full training curves, but at test time it receives only subject-level information:

```text
MLP test input = subject descriptor + driving condition
```

It receives:

```text
no observed test-curve points
```

Therefore:

```text
MLP error is constant across fractions.
TRF error changes across fractions.
```

Do not modify the MLP training function to truncate training curves to the fraction window. That would mix two different concepts:

1. training-data availability, and
2. test-time observation budget.

---

## Main Output Files

The scripts generate CSV and text summaries in the selected output directory.

Typical rheology outputs:

```text
results/rheology_loo/
├── all_folds_raw.csv
├── aggregated_summary.csv
└── fold_level_outputs.csv
```

Typical PK outputs:

```text
results/pk_all/
├── MASTER_SUMMARY.txt
├── MASTER_all_folds_raw.csv
├── SEED_AGGREGATED.csv
├── SEED_EQUATION_WINS.csv
└── per_config_outputs/
```

---

## Output Columns

The raw result tables contain columns such as:

```text
fold
variant
equation
material
file
sigma0
weight
fraction
mlp_only
trf_cold
trf_warm
n_obs
n_train
n_test
```

For rheology, `material` refers to the alloy or material family.

For pharmacokinetics, `material` is used as a shared internal name for the subject identifier.

---

## How to Interpret Results

Each row compares three methods:

```text
mlp_only
trf_cold
trf_warm
```

Lower MSE is better.

Example interpretation:

```text
If mlp_only < trf_cold at low fractions:
    amortized inference is better in the low-data regime.

If trf_cold < mlp_only at higher fractions:
    direct per-curve fitting becomes better once enough observations are available.

If trf_warm < trf_cold:
    the MLP provides a useful initialization for nonlinear least squares.
```

---

## Expected Findings

The expected pattern is:

```text
Coupled equations perform better under amortized inference.
Uncoupled equations require direct curve fitting.
```

In rheology:

```text
Guiu-Pratt supports amortized prediction.
Thermal relaxation does not support amortized prediction as strongly.
```

In pharmacokinetics:

```text
The one-compartment model supports amortized prediction.
The two-compartment model is harder to predict without test-curve observations.
```

---

## Reproducing the Paper Experiments

A complete reproduction workflow is:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run rheology experiments
python src/cv_threeway.py \
  --data-dir data/rheology \
  --output-dir results/rheology_loo \
  --cv-mode loo \
  --split-variant all \
  --model both

# 3. Run pharmacokinetics experiments
python src/cv_threeway_pk.py \
  --csv data/theoph.csv \
  --all-variants \
  --output-dir results/pk_all

# 4. Inspect summaries
cat results/pk_all/MASTER_SUMMARY.txt
```

---

## Command-Line Arguments

### `cv_threeway.py`

| Argument          | Description                               |
| ----------------- | ----------------------------------------- |
| `--data-dir`      | Directory containing rheology curve files |
| `--output-dir`    | Directory for result files                |
| `--cv-mode`       | Cross-validation mode: `loo` or `kfold`   |
| `--split-variant` | Use `all` curves or only `duplicates`     |
| `--model`         | `guiu_pratt`, `thermal`, or `both`        |
| `--trf-fractions` | Comma-separated observation fractions     |
| `--trf-restarts`  | Number of TRF random restarts             |
| `--epochs`        | MLP training epochs                       |
| `--lr`            | Learning rate                             |
| `--weight-decay`  | AdamW weight decay                        |
| `--device`        | `cpu`, `cuda`, or `mps`                   |

---

### `cv_threeway_pk.py`

| Argument          | Description                                          |
| ----------------- | ---------------------------------------------------- |
| `--csv`           | Path to `theoph.csv`                                 |
| `--output-dir`    | Directory for result files                           |
| `--model`         | `one_compartment`, `two_compartment`, or `both`      |
| `--all-variants`  | Run all PK input variants                            |
| `--no-subject-id` | Use population variant instead of subject-ID variant |
| `--no-sigma0`     | Remove dose from MLP input                           |
| `--seeds`         | Comma-separated seeds for all-variants mode          |
| `--trf-fractions` | Comma-separated observation fractions                |
| `--trf-restarts`  | Number of TRF random restarts                        |
| `--epochs`        | MLP training epochs                                  |
| `--lr`            | Learning rate                                        |
| `--weight-decay`  | AdamW weight decay                                   |
| `--device`        | `cpu`, `cuda`, or `mps`                              |

---

## Notes on Data

The rheology data should provide:

```text
time
stress
material identity
initial stress sigma0
temperature, if available
```

The pharmacokinetic data should provide:

```text
subject ID
time
concentration
dose
body weight
```

The PK script expects a CSV file such as:

```text
data/theoph.csv
```

---

## Troubleshooting

### The MLP has the same MSE at every fraction

This is expected.

The MLP receives no test-curve observations. Its prediction is independent of the observation fraction.

---

### TRF performs poorly at very low fractions

This is expected in low-data regimes.

With too few observed points, nonlinear least squares may not identify the parameters reliably.

Increase:

```bash
--trf-restarts
```

or use larger observation fractions.

---

### CUDA is not available

Run on CPU:

```bash
python src/cv_threeway.py \
  --data-dir data/rheology \
  --device cpu
```

or:

```bash
python src/cv_threeway_pk.py \
  --csv data/theoph.csv \
  --device cpu
```

---

### Missing module error

Example:

```text
ModuleNotFoundError: No module named 'pk_pipeline'
```

Make sure all source files are in the correct location:

```text
src/
├── cv_threeway.py
├── cv_threeway_pk.py
├── guiu_pratt_pipeline.py
├── pk_pipeline.py
└── pk_physics.py
```

Run scripts from the repository root.

---

## Citation

```bibtex
@inproceedings{dash2026amortized,
  title     = {Amortized Model Selection for Scientific Equations: A Predictive Validation Procedure with Cross-Domain Evidence},
  author    = {Dash, Manoranjan and Aralikatti, Shivam},
  booktitle = {IEEE International Conference on Data Mining},
  year      = {2026},
  note      = {Manuscript under review}
}
```

---

## Keywords

```text
amortized inference
scientific machine learning
model selection
physics-informed neural networks
differentiable physics
Trust Region Reflective
stress relaxation
pharmacokinetics
Guiu-Pratt
theophylline
low-data regime
cross-validation
```

---


## Acknowledgement

This work was conducted at **FLAME University, Pune, India**.

The repository accompanies research on amortized model selection for scientific equations across rheology and pharmacokinetics.
