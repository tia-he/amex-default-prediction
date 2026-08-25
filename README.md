# American Express Default Prediction

Predicting whether a credit card customer will default, based on their longitudinal monthly
financial behavior. Built on Kaggle's [American Express - Default Prediction](https://www.kaggle.com/competitions/amex-default-prediction)
competition dataset — but structured as a credit-risk modeling project, not a leaderboard
chase.

Each customer here isn't a single snapshot: they show up as a sequence of monthly account
statements, and whether they default is a judgment that has to be inferred from both their
current state and how they got there. That longitudinal-to-customer-level transformation —
and whether it actually yields useful predictive signal — is the central question this
project works through.

**Central question:** can a customer's longitudinal financial behavior be transformed into
useful customer-level signals for predicting future default?

## Project Overview

```
Monthly Customer Statements
        ↓
Data Understanding & EDA
        ↓
Customer-Level Feature Engineering
        ↓
Stratified Validation
        ↓
Logistic Regression · LightGBM · CatBoost
        ↓
AMEX Evaluation Metric
```

This repository currently covers **Phases 1–3**: data understanding, feature engineering,
and baseline modeling. Each phase is one fully executed, self-contained notebook.

## Dataset

Source: Kaggle's AMEX Default Prediction training data, verified directly from the raw
files (`notebooks/01_data_exploration.ipynb`):

- **5,531,451** statement-level rows, **190** columns
- **458,913** unique customers
- Up to **13** monthly statements per customer — **84.1%** of customers have the full
  13-statement history
- Statement dates span **2017-03-01** to **2018-03-31**
- Default rate: **25.89%**

One raw row represents one customer's financial state at one monthly statement date — not
one row per customer. Features fall into five families by prefix:

| Prefix | Family | Count |
|---|---|---|
| `D_` | Delinquency | 96 |
| `B_` | Balance | 40 |
| `R_` | Risk | 28 |
| `S_` | Spend | 21 |
| `P_` | Payment | 3 |

Missingness is substantial and unevenly distributed — 33 features are missing in over 25%
of statements — and Phase 1 treats that as a signal worth preserving rather than a defect
to immediately impute away.

## Feature Engineering

The raw data is statement-level (~5.5M rows); the prediction target is customer-level.
Phase 2 (`notebooks/02_feature_engineering.ipynb`) collapses one into the other:

```
5,531,451 monthly observations
            ↓
   Customer-level aggregation
            ↓
       458,913 customers
```

**Continuous features** get seven aggregates each: `mean`, `std`, `min`, `max`, `first`,
`last`, and `change` (`last - first`). Each captures something different — `mean` is the
customer's typical historical state, `last` is where they stand most recently, and `change`
is the direction they've been moving. A customer with an unremarkable average balance but a
sharply rising `change` looks very different from one whose average looks the same but is
declining.

**Categorical features** (both true string categories and small-integer-coded ones,
identified through explicit cardinality and value inspection rather than assumed from
dtype) get three fields each: `first`, `last`, and the number of distinct categories
observed.

A selective set of customer-level **missing-rate** features and two general **history**
features (statement count, history span in days) round out the set. In total, Phase 2
produces **1,299 engineered features** across 458,913 customers before any
modeling-stage exclusions.

## Memory-Efficient Processing

The raw training CSV is ~15GB; the development machine has ~8GB of RAM. Naively loading
the full file was never an option, so the pipeline is built around **streaming, chunked
processing**: Phase 2's entire customer-level feature set — all 1,299 features — is
produced in a **single full pass** over the raw CSV (56 chunks, 1.84 minutes), using
boundary-aware grouping to aggregate correctly across chunk edges without ever holding the
full dataset in memory.

The resulting customer-level table is saved locally as Parquet
(`data/processed/train_features.parquet`, ~2.6GB) — Phase 3 loads that instead of
re-reading the raw CSV. Both raw and processed competition data are excluded from version
control (see [Reproducibility / Data Access](#reproducibility--data-access)).

## Validation Strategy

Phase 3 (`notebooks/03_baseline_modeling.ipynb`) uses a reproducible **customer-level
stratified holdout**:

- 80% train / 20% validation, stratified by `target`, random seed 42
- **367,130** training customers, **91,783** validation customers
- Default rate: **25.89%** in both splits

This is a **random stratified split, not a temporal or out-of-time split** — the processed
dataset has no defensible customer application-time ordering to validate against, so no
claim of temporal generalization is made here. All three models are trained and evaluated
on the exact same validation customers, making the comparison direct.

## Evaluation Metric

Results are reported using the official AMEX competition metric, which combines two
components:

```
AMEX Metric = 0.5 × Normalized Weighted Gini + 0.5 × Top-4% Default Capture
```

Gini measures overall ranking quality across the full population; top-4% capture measures
whether the model concentrates actual defaulters near the very top of its risk ranking —
the slice a real credit operation could act on. Plain ROC AUC is also reported for
interpretability, but on its own it can't distinguish models that rank the bulk of
customers similarly yet disagree specifically on the highest-risk tail.

## Baseline Models

Three model families, trained on identical features and the identical validation split,
with sensible baseline configurations rather than tuned ones:

- **Logistic Regression** — a linear benchmark.
- **LightGBM** — gradient-boosted trees, non-linear interactions, native categorical
  support.
- **CatBoost** — gradient-boosted trees with a different native categorical-handling
  approach.

## Results

Holdout validation results (not Kaggle leaderboard scores):

| Model | ROC AUC | Normalized Gini | Top-4% Capture | AMEX Metric | Train Time |
|---|---|---|---|---|---|
| Logistic Regression | 0.9604 | 0.9207 | 0.6506 | 0.7856 | 27.0s |
| LightGBM | 0.9621 | 0.9242 | 0.6594 | 0.7918 | 170.8s |
| CatBoost | 0.9626 | 0.9252 | 0.6620 | 0.7936 | 1278.9s |

CatBoost is the strongest current baseline, with LightGBM close behind and training
substantially faster. Logistic Regression is surprisingly competitive given its much
simpler functional form — the gap to the tree-based models is real but modest, not a
landslide. That's itself informative: it suggests the engineered customer-level features
already carry substantial linearly-separable predictive structure, and the trees are
extracting incremental (not transformative) value on top of it.

## Key Findings

**Recent behavior dominates feature importance.** `last`-aggregated features are the large
majority of both models' top 20 by importance — 13 of 20 for LightGBM, 15 of 20 for
CatBoost. A customer's most recent statement appears markedly more informative than their
historical average.

**Payment behavior is the single strongest signal.** `P_2_last` ranks #1 in both LightGBM's
and CatBoost's feature importance, by a wide margin. (`P_2`'s specific business definition
isn't established in the dataset documentation available to this project, so no semantic
claim is made beyond what the ranking itself shows.)

**Balance and delinquency features appear frequently.** `B_` and `D_` prefixed features
make up the majority of both models' top-20 lists, alongside a smaller but consistent
presence of `R_` (risk) features.

**Model complexity yields incremental, not transformative, gains.** CatBoost improves the
AMEX metric over Logistic Regression by roughly 0.008 — real, but modest relative to the
47x difference in training time. None of this implies causation; it describes what these
particular models weighted heavily on this split, not why default happens.

## Repository Structure

```
amex-default-prediction/
├── notebooks/
│   ├── 01_data_exploration.ipynb       # Phase 1 — EDA
│   ├── 02_feature_engineering.ipynb    # Phase 2 — customer-level features
│   └── 03_baseline_modeling.ipynb      # Phase 3 — validation & baselines
├── figures/                            # exported plots from Phase 1
├── .gitignore
├── README.md
└── requirements.txt
```

Competition data (raw and processed) lives locally under `data/` and is not tracked in
this repository — see below.

## Reproducibility / Data Access

The dataset comes from Kaggle's [American Express - Default Prediction](https://www.kaggle.com/competitions/amex-default-prediction)
competition. Because it's governed by Kaggle's competition terms, raw and processed data
are **not included in this repository** and are not redistributed here.

To reproduce this project, obtain the data directly from Kaggle (competition access
required) and place it locally as:

```
data/
├── train_data.csv
├── train_labels.csv
├── test_data.csv
└── sample_submission.csv
```

Notebooks expect this layout relative to the repository root. Install dependencies with
`pip install -r requirements.txt`.

## Current Status / Next Steps

**Completed:**
- Data Understanding & EDA
- Customer-Level Feature Engineering
- Baseline Modeling (Logistic Regression, LightGBM, CatBoost)

**Possible future work** (none of the below has been started):
- Feature ablation (e.g. testing whether the missing-rate features add value)
- Feature-engineering refinement
- Lightweight hyperparameter tuning
- Model ensembling
- Final inference / evaluation
