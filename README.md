# Modeling Air Quality with Multi-Output Gaussian Processes

This project explores whether **Multi-Output Gaussian Processes (MOGPs)** can outperform independent single-output models on urban air quality prediction — and whether they can guide efficient data collection through Bayesian optimization.

The dataset is hourly air quality readings from an Italian city (UCI Air Quality, De Vito et al. 2008). The goal is to jointly predict four pollutant concentrations — CO, Benzene (C6H6), NOx, and NO2 — from cheap metal-oxide sensor readings and meteorological variables. Reference analysers that provide ground-truth measurements are expensive to deploy, so every labelled example counts.

---

## The Setup

**Features (10):** 5 metal-oxide sensor readings + temperature, relative humidity, absolute humidity + cyclic hour encoding (sin/cos)

**Outputs (4):** CO(GT), C6H6(GT), NOx(GT), NO2(GT) — reference analyser measurements

**Models compared:**
- Independent GP — one ARD-RBF GP per output, no cross-output sharing
- ICM (Q=1) — Intrinsic Coregionalization Model, single shared latent
- LCM (Q=2) — Linear Coregionalization Model, two latent GPs with independent lengthscales
- Deep Ensemble MLP — 15 MLPs for uncertainty estimation (sklearn only)

**Evaluation:** RMSE (accuracy) and NLPD (calibration quality, lower = better)

---

## 1. Exploratory Data Analysis

The raw dataset has missing values encoded as −200 sentinels. After cleaning, ~6000 usable rows remain. NMHC(GT) is dropped entirely (>90% missing).

The four outputs are strongly correlated — all combustion/traffic-related pollutants that tend to peak together. This is the key premise for using MOGPs: if the outputs move together, a model that learns their joint covariance should outperform one that treats them independently.

<p align="center">
  <img src="outputs/01c_output_correlation.png" width="420"/>
  <img src="outputs/01b_output_distributions.png" width="560"/>
</p>

There's also clear temporal structure — morning rush hour peaks for CO and NOx, summer ozone-driven patterns for NO2. This motivated including cyclic hour features (sin/cos encoding) rather than raw hour integers.

<p align="center">
  <img src="outputs/01d_temporal_patterns.png" width="760"/>
</p>

---

## 2. Baseline: Independent GPs

One ARD-RBF GP per output, fitted independently. The ARD lengthscales reveal which sensors each pollutant is most sensitive to — useful as a sanity check and for feature interpretation.

<p align="center">
  <img src="outputs/02b_igp_parity.png" width="700"/>
</p>

<p align="center">
  <img src="outputs/02c_ard_lengthscales.png" width="700"/>
</p>

The sensor features dominate over meteorological ones for all outputs. Short lengthscales on S1(CO) and S2(NMHC) confirm these are the most informative sensors for CO and Benzene prediction.

**Baseline RMSE:** CO=0.419, C6H6=0.031, NOx=79.6, NO2=23.8

---

## 3. Multi-Output GPs: ICM and LCM

Both models learn a joint covariance structure across all four outputs. The ICM uses a single shared kernel scaled by a 4×4 coregionalization matrix B; the LCM uses two latent GPs with independent lengthscales mixed into the outputs via a learned weight matrix W.

<p align="center">
  <img src="outputs/03a_model_comparison.png" width="800"/>
</p>

The learned B matrix from ICM confirms the strong positive correlations between all pollutants, with the off-diagonal entries roughly grouping (CO, C6H6) together and (NOx, NO2) together — matching what you'd expect from their shared combustion sources.

<p align="center">
  <img src="outputs/03b_icm_coregionalization.png" width="380"/>
  <img src="outputs/03c_lcm_mixing_and_lengthscales.png" width="580"/>
</p>

**Full-data results:**

| Model | CO RMSE | C6H6 RMSE | NOx RMSE | NO2 RMSE |
|-------|---------|-----------|---------|---------|
| Independent GP | 0.419 | 0.031 | 79.6 | 23.8 |
| ICM (Q=1) | **0.385** | 0.202 | **69.5** | **20.9** |
| LCM (Q=2) | 0.403 | **0.039** | 80.5 | 24.0 |

ICM wins on three of four outputs at full data. LCM's implicit regularization from a single shared latent actually helps at larger n — the second latent in LCM adds parameters without adding enough signal when there's plenty of data.

---

## 4. Low-Data Regime

The more interesting question: does the MOGP advantage grow when data is scarce?

Training sizes {20, 40, 80, 160, 320, full} are tested with 5 random seeds each. The shaded bands show ±1 std.

<p align="center">
  <img src="outputs/04a_low_data_rmse.png" width="760"/>
</p>

LCM is clearly better at small n — especially on NOx (Y3) where it cuts RMSE by ~32% over independent GPs at n=20. The advantage shrinks as more data becomes available, which makes sense: with enough observations per output, independent GPs don't need to borrow strength from other outputs.

The uncertainty comparison at n=40 makes this concrete — LCM produces tighter predictive intervals because it uses information from all four outputs simultaneously:

<p align="center">
  <img src="outputs/04c_uncertainty_n40.png" width="760"/>
</p>

---

## 5. Bayesian Optimization: Pareto Front Discovery

Can the MOGP surrogate guide efficient data collection in a multi-objective setting?

We frame this as a pool-based active learning problem: given a large pool of unlabelled time points (cheap sensor readings only), sequentially select which ones to label with the expensive reference analyser to map out the CO–NO2 Pareto front as quickly as possible.

**Acquisition:** Thompson sampling — draw a posterior sample over the pool, find its Pareto front, pick a random point from it. Three strategies:
- Random search (baseline)
- Independent GP + Thompson sampling
- LCM + Thompson sampling

Starting from 30 initial random points, each strategy gets 40 additional evaluations.

<p align="center">
  <img src="outputs/05a_hypervolume_trace.png" width="700"/>
</p>

LCM consistently reaches higher hypervolume faster than both baselines. The random strategy plateaus early; independent GP improves but misses high-CO/high-NO2 co-pollution events that the MOGP surrogate specifically targets by modelling the joint distribution.

<p align="center">
  <img src="outputs/05c_pareto_fronts.png" width="900"/>
</p>

The true Pareto front for reference:

<p align="center">
  <img src="outputs/05d_true_pareto_front.png" width="700"/>
</p>

---

## 6. Neural Network Comparison

A deep ensemble of 15 MLPs provides the non-GP baseline for uncertainty. Architecture selected via grid search on validation RMSE.

<p align="center">
  <img src="outputs/06b_model_comparison_bar.png" width="900"/>
</p>

The MLP completely fails on C6H6 (RMSE=0.777 vs 0.031 for independent GP) — likely because C6H6 has a tight, low-variance distribution that requires well-calibrated uncertainty to predict well, and the ensemble variance is poorly estimated here. GP calibration (NLPD) is consistently better across all outputs.

<p align="center">
  <img src="outputs/06c_nn_low_data_rmse.png" width="760"/>
</p>

In the low-data regime, the MLP degrades quickly and erratically. GPs — particularly LCM — degrade much more gracefully.

**Final comparison (full data):**

| Model | CO RMSE | C6H6 RMSE | NOx RMSE | NO2 RMSE | CO NLPD | NOx NLPD |
|-------|---------|-----------|---------|---------|---------|---------|
| Independent GP | 0.419 | **0.031** | 79.6 | 23.8 | 0.552 | 5.878 |
| ICM (Q=1) | **0.385** | 0.202 | **69.5** | **20.9** | **0.434** | 5.901 |
| LCM (Q=2) | 0.403 | 0.039 | 80.5 | 24.0 | 0.502 | **5.825** |
| Deep Ensemble MLP | 0.423 | 0.777 | 78.8 | 22.4 | 0.489 | 7.680 |

---

## How to Run

```bash
pip install gpy numpy pandas scikit-learn matplotlib seaborn requests
```

Run the scripts in order:

```bash
python 01_eda.py
python 02_independent_gps.py
python 03_icm_vs_lcm.py
python 04_low_data_regime.py
python 05_pareto_optimization.py   # slowest — ~15–20 min
python 06_nn_comparison.py
```

All outputs (plots + JSON results) are saved to `outputs/`. The dataset is downloaded automatically from UCI on first run.

---

## Dependencies

- [GPy](https://github.com/SheffieldML/GPy) — GP models
- NumPy, pandas, scikit-learn, matplotlib, seaborn
- Python 3.10+
