# Modeling Air Quality with Multi-Output Gaussian Processes

This project investigates whether Multi-Output Gaussian Processes (MOGPs) can outperform independent models on urban air quality prediction, and whether they can guide efficient data collection through Bayesian optimization. The central question is practical: reference-grade air quality analysers are expensive to deploy, so can exploiting inter-pollutant correlations extract more information from fewer measurements?

I use the UCI Air Quality dataset (De Vito et al., 2008): hourly readings from an Italian city between March 2004 and February 2005, comprising 5 metal-oxide sensor channels, 3 meteorological variables, and 4 reference analyser outputs (CO, Benzene/C6H6, NOx, NO2). After cleaning, 6,941 usable rows remain. This run uses the full dataset (4,858 train / 694 val / 1,389 test), with GP inference accelerated on an A100 GPU via the CSF3 HPC cluster.

I compare three GP model families: independent GPs (one ARD-RBF GP per output), ICM (Intrinsic Coregionalization Model, Q=1 latent), and LCM (Linear Coregionalization Model, Q=2 latents with ARD), along with a deep ensemble of MLPs as a non-GP baseline.

**Main findings (full data):** Contrary to the theoretical promise of coregionalization, independent GPs outperform both ICM and LCM on all four outputs by RMSE at this data scale. With 4,858 training points each GP has sufficient data to learn the input space efficiently without borrowing strength across outputs. The coregionalization overhead — additional parameters, a harder optimization landscape — is not compensated by cross-output information at this scale. However, MOGPs do provide a calibration advantage in the low-data regime (n=20), and GP-based acquisition functions (both independent and LCM variants) dramatically outperform random search in the Bayesian optimization experiment.

---

## 1. Exploratory Data Analysis

### Missing Data

The dataset uses -200 as a sentinel for missing values. NMHC(GT) is >90% missing and is dropped entirely. The four reference analyser outputs (CO, C6H6, NOx, NO2) each have 15–20% missingness; rows with any missing target are dropped, bringing the full dataset from 9,358 to 6,941 rows. Sensor and meteorological columns have <10% missingness and are median-imputed using training-set statistics only, after the train/test split, to prevent data leakage. The asymmetry is noteworthy: the expensive reference analysers are more often offline than the cheap metal-oxide sensors.

<p align="center">
  <img src="outputs/01a_missing_data.png" width="750"/>
</p>

### Output Distributions

<p align="center">
  <img src="outputs/01b_output_distributions.png" width="800"/>
</p>

All four outputs are right-skewed with heavy tails, typical of pollutant concentration data. CO has a median of roughly 1.5 mg/m³ with occasional spikes above 8; C6H6 is tightly clustered near 6 µg/m³ with a long tail; NOx has a median around 166 ppb but extends past 1,000 ppb during high-traffic events; NO2 is the most Gaussian-looking with a median around 110 µg/m³. The substantial scale differences across outputs — from single-digit CO to triple-digit NOx — directly influence how coregionalization matrices get weighted during optimization, a point returned to in Section 3.

### Output Correlations

<p align="center">
  <img src="outputs/01c_output_correlation.png" width="420"/>
</p>

The inter-pollutant correlations are the core justification for attempting a MOGP. All off-diagonal correlations are positive and substantial: CO–C6H6 (r=0.930), CO–NOx (r=0.786), C6H6–NOx (r=0.718), NOx–NO2 (r=0.743), NO2–CO (r=0.674), NO2–C6H6 (r=0.603). The CO–C6H6 coupling is particularly strong; both are direct combustion products that peak simultaneously during traffic events. NO2 is the weakest correlate of the group because it involves secondary photochemical formation and has a distinct diurnal cycle. MOGPs are theoretically most beneficial when outputs are positively correlated, so these numbers validate the premise — at least in principle.

<p align="center">
  <img src="outputs/01f_output_pairplot.png" width="650"/>
</p>

The pairplot confirms the correlations are not purely linear. The CO–C6H6 and CO–NOx joint densities show elongated, slightly curved structures, and there is visible heteroscedasticity: variance grows with the mean for most pairs. This is the kind of structure that a kernel-based model can capture through learned lengthscales.

### Temporal Structure

<p align="center">
  <img src="outputs/01d_temporal_patterns.png" width="800"/>
</p>

The hourly profiles are physically informative. CO and NOx show classic double-peak patterns: morning rush (7–9 am) and evening peak (6–8 pm), with a midday dip as atmospheric mixing disperses surface-level pollutants. C6H6 follows CO almost exactly. NO2 rises through the day and peaks in the late evening, reflecting its dependence on photochemical reactions and secondary formation from NOx oxidation. Seasonally, CO and NOx are elevated in winter (November to February), consistent with heating emissions and a reduced atmospheric boundary layer height. These patterns motivated cyclic hour encoding (sin/cos of hour-of-day) rather than raw integer hour, which would impose a spurious 23→0 discontinuity on the kernel.

### Feature Importance

<p align="center">
  <img src="outputs/01e_feature_output_correlation.png" width="500"/>
</p>

The metal-oxide sensor columns dominate. S1(CO) is strongly correlated with CO (r≈0.88) and C6H6 (r≈0.91). S2(NMHC) shows similar behaviour. Meteorological variables have weak correlations with all outputs: temperature has a modest negative relationship with NOx and NO2, consistent with greater photochemical activity in warmer conditions, but RH and AH are nearly orthogonal to the pollutant concentrations. The cyclic hour features show moderate correlations, confirming time-of-day carries predictive signal beyond the sensor readings alone.

---

## 2. Baseline: Independent GPs

Before introducing any cross-output structure, I fit one ARD-RBF GP independently to each output using GPyTorch, with 3 random restarts to avoid local optima in hyperparameter optimization. This is a strong baseline: it learns feature-specific lengthscales per output and produces calibrated uncertainty estimates, but treats the four pollutants as entirely unrelated.

<p align="center">
  <img src="outputs/02a_igp_predictions.png" width="800"/>
</p>

<p align="center">
  <img src="outputs/02b_igp_parity.png" width="800"/>
</p>

**Test-set results:**

| Output | RMSE | NLPD |
|--------|------|------|
| CO (mg/m³) | 0.371 | 0.469 |
| C6H6 (µg/m³) | 0.032 | -0.508 |
| NOx (ppb) | 51.85 | 5.308 |
| NO2 (µg/m³) | 15.76 | 4.258 |

C6H6 is predicted almost perfectly (RMSE=0.032, NLPD=-0.508); its tight, well-clustered distribution and strong sensor correlation make it a tractable target even for a simple kernel. CO achieves RMSE=0.371. NOx has a high absolute RMSE (51.85 ppb), but this must be read against its dynamic range — values span from near-zero to over 1,000 ppb, and the relative fit is reasonable. NO2 is the most challenging output: its secondary photochemical nature means sensor readings are a noisier proxy, and the model's uncertainty is correspondingly wider (NLPD=4.258).

### ARD Lengthscales

<p align="center">
  <img src="outputs/02c_ard_lengthscales.png" width="750"/>
</p>

The learned ARD lengthscales serve as feature importance indicators. Relative humidity (RH) receives an extremely large lengthscale for all outputs — the model effectively ignores it, consistent with its near-zero Pearson correlations. Temperature (T) has smaller lengthscales across all outputs, confirming its modest relevance. Sensor columns show moderate lengthscales, with variation across outputs reflecting each pollutant's particular sensor dependencies. The absolute values (in the tens of thousands) reflect the standardised input space; the informative quantity is the relative ordering across features within each output, not the magnitude.

---

## 3. Multi-Output GPs: ICM and LCM

### Full-Data Performance: A Negative Result

<p align="center">
  <img src="outputs/03a_model_comparison.png" width="850"/>
</p>

At full data (4,858 training points), both coregionalization models perform worse than the independent GP baseline across all four outputs by RMSE. This is the central and perhaps surprising finding of this experiment:

**RMSE:**

| Model | CO | C6H6 | NOx | NO2 |
|-------|-----|-------|-----|-----|
| Independent GP | **0.371** | **0.032** | **51.85** | **15.76** |
| ICM (Q=1) | 0.444 | 0.085 | 55.68 | 18.37 |
| LCM (Q=2) | 0.453 | 0.068 | 54.03 | 29.98 |

**NLPD:**

| Model | CO | C6H6 | NOx | NO2 |
|-------|-----|-------|-----|-----|
| Independent GP | **0.469** | **-0.508** | **5.308** | **4.258** |
| ICM (Q=1) | 3.381 | 2.554 | 8.759 | 7.363 |
| LCM (Q=2) | 0.628 | -0.072 | 5.365 | 4.818 |

The ICM results are particularly poor in calibration terms. NLPD of 3.38 for CO and 2.55 for C6H6 indicate that the model assigns near-zero probability density to the true values — the predictive variance is systematically miscalibrated. LCM is considerably better on NLPD but still substantially worse than the independent GP, and its NO2 RMSE (29.98 vs 15.76) represents a dramatic regression.

### Why Coregionalization Fails Here

The result is interpretable. With 4,858 training points per output, each independent GP can learn the input manifold thoroughly without needing borrowed strength from correlated outputs. At this data scale, the inter-output correlations represent no information that cannot be recovered from within-output data alone.

Meanwhile, coregionalization introduces genuine costs: more parameters (ICM adds a full Q×T² coregionalization matrix; LCM with ARD adds Q×D per-latent lengthscales), a harder non-convex optimization landscape, and — critically — the marginal likelihood landscape has many more local optima. At 4,858 points, GP hyperparameter optimization is already expensive, and the coregionalization parameters are harder to set accurately than per-output lengthscales. The result is that the optimizer finds poor local minima, particularly for ICM.

### The ICM Coregionalization Matrix

<p align="center">
  <img src="outputs/03b_icm_coregionalization.png" width="480"/>
</p>

The learned B matrix reveals the mechanism of failure. The NOx–NOx entry dominates (3604.06) and the NOx–NO2 cross-term is large (404.71), while CO and C6H6 entries are near zero across the board:

```
        CO      C6H6    NOx       NO2
CO      0.01    0.01     3.26      0.01
C6H6    0.01    0.01     0.01      0.01
NOx     3.26    0.01  3604.06   404.71
NO2     0.01    0.01   404.71    43.36
```

This is not a random pathology — it is a direct consequence of raw output units. NOx values span hundreds of ppb while CO is in single-digit mg/m³. The marginal likelihood weights outputs by their variance during optimization, so the shared latent process is allocated almost entirely to the highest-variance output (NOx). In effect, the ICM becomes a strong NOx predictor and a poor predictor of everything else. The NOx–NO2 coupling it does learn is physically sensible (NO2 is partly formed by NOx oxidation), but the total structure is dominated by this single scale imbalance. Pre-standardizing outputs before fitting would partially address this, but it would also change the marginal likelihood landscape and the physical interpretability of B.

### The LCM Mixing Matrix

<p align="center">
  <img src="outputs/03c_lcm_mixing_and_lengthscales.png" width="750"/>
</p>

The LCM mixing weights show the same pattern. Latent GP 1 carries large weights for NOx (6.42) and NO2 (2.14), with near-zero weights for CO and C6H6. Latent GP 2 has a similar structure (NOx: 3.84, NO2: 1.76), adding a second latent process that recapitulates the same NOx-dominated structure rather than capturing the CO–C6H6 co-variation. The two latents have specialised differently in their ARD lengthscales, suggesting they partition the input space, but neither latent provides meaningful coupling to CO or C6H6. This is why LCM improves only marginally over ICM on some outputs while degrading badly on NO2.

### Model Evidence

<p align="center">
  <img src="outputs/03e_nlml_comparison.png" width="480"/>
</p>

The negative log marginal likelihood (NLML) confirms the Occam's razor interpretation. The independent GP achieves the lowest NLML (1246.5), followed by ICM (1247.8) and LCM (1248.1). The coregionalization models have substantially more parameters, and the marginal likelihood penalises this complexity automatically. The RMSE degradation from coregionalization is therefore accompanied by worse model evidence — there is no trade-off here, just worse models.

<p align="center">
  <img src="outputs/03d_parity_comparison.png" width="850"/>
</p>

The parity plots confirm that ICM and LCM predictions are more scattered around the diagonal for CO and NO2 than the independent GP predictions. For NOx the scatter is comparable. The visual story matches the numbers.

---

## 4. Low-Data Regime

The motivation for MOGPs in sparse-data settings is clearer than at full data. Deploying a reference analyser for a year is expensive; what if you only have 40 measurements? The low-data experiments subsample the training set to n ∈ {20, 40, 80, 160} and evaluate over 5 random seeds to estimate variance.

### RMSE vs Training Size

<p align="center">
  <img src="outputs/04a_low_data_rmse.png" width="800"/>
</p>

The picture is output-dependent and does not cleanly favour any single model. For C6H6, the independent GP is best at all training sizes — its tight, well-correlated distribution rewards independent modelling even with very few points. For NOx, LCM shows a sharp spike at n=80 (RMSE more than doubles relative to n=40 before recovering at n=160). This is an optimization failure: the LCM with ARD has over 38 free parameters for Q=2 latents across D=10 features and T=4 outputs; trying to optimize these from 80 points leaves the optimizer in a poor local minimum. ICM is more robust to this pathology (no per-latent ARD) but still degrades badly on CO at n=160, where its coregionalization structure actively hurts that specific output. LCM is more stable than ICM overall across the range.

### Calibration (NLPD) vs Training Size

<p align="center">
  <img src="outputs/04b_low_data_nlpd.png" width="800"/>
</p>

The calibration picture is where MOGPs show their strongest advantage, and it is output-specific. For CO at n=20, the independent GP NLPD spikes to roughly 35–40 nats — the posterior variance collapses and the model assigns near-zero probability to true values. C6H6 shows a similar but less extreme pattern (20–25 nats at n=20). In both cases, ICM and LCM stay much lower (CO: ~5–10 nats; C6H6: ~1–3 nats), because cross-output coupling provides a regularising effect on the posterior variance when within-output data is scarce.

However, this calibration advantage is not universal. For NO2 at n=20, LCM is poorly calibrated (NLPD ~11), while ICM and the independent GP are more stable. For NOx, the independent GP is stably calibrated throughout while LCM spikes at n=80 as described above. By n=160 all three models converge to similar NLPD values.

The practical takeaway: if you must work with n=20 measurements and your priority is a calibrated uncertainty estimate for CO or C6H6, a MOGP is substantially better than an independent GP. For NOx and NO2, the advantage is absent or negative.

### Predictive Uncertainty at n=40

<p align="center">
  <img src="outputs/04c_uncertainty_n40.png" width="900"/>
</p>

At n=40, both models are trained on the same 40 points and evaluated on the full test set. Test samples are sorted by true target value, so the ±2σ band should track the true-value dots to indicate calibration. For C6H6, the independent GP is tighter and more accurate (RMSE 0.195 vs LCM's 0.264). For NOx and NO2 the uncertainty bands are wide for both models — 40 points is too few to characterise these outputs well — and the independent GP is more reliable because LCM's many ARD parameters fail to optimise from such a small sample.

---

## 5. Bayesian Optimization: Pareto Front Discovery

Beyond prediction, MOGPs serve as surrogates in active learning loops. The experiment here: given a pool of unlabelled time points (cheap sensor readings available, but reference analyser measurements not yet taken), can a GP surrogate guide which points to label so as to efficiently map the joint CO–NO2 extremes?

### The True Pareto Front

<p align="center">
  <img src="outputs/05d_true_pareto_front.png" width="750"/>
</p>

The true CO–NO2 Pareto front (maximising both) consists of 10 non-dominated points with total hypervolume 3,443.2. The trade-off curve reveals an interesting physical structure: the most extreme NO2 events (>300 µg/m³) occur at moderate CO levels, while peak CO events (~11–12 mg/m³) correspond to moderate NO2. This reflects different source processes — extreme NO2 involves secondary photochemical formation while peak CO is a direct combustion product — so the two extremes do not coincide. The vast majority of the ~6,900 pool points are dominated.

### Active Learning Comparison

Three strategies, each starting from 30 random initial observations and making 40 sequential selections (Thompson sampling for GP strategies):

- **Random search:** select uniformly at random from the pool
- **Independent GP + Thompson sampling:** two independent GPs, draw a posterior sample, find its Pareto front, select from it
- **LCM + Thompson sampling:** joint MOGP surrogate, same acquisition procedure

<p align="center">
  <img src="outputs/05a_hypervolume_trace.png" width="720"/>
</p>

Random search achieves roughly 39% of the true hypervolume after 70 total evaluations. Both GP strategies dramatically outperform it, reaching 90–93% of the true hypervolume. The two GP strategies are closely matched across the 5-run average; LCM shows faster early convergence in some seeds by exploiting the CO–NO2 posterior correlation in its surrogate, but both strategies are within error bands of each other by N=40 additional observations.

<p align="center">
  <img src="outputs/05b_normalised_hv.png" width="720"/>
</p>

The normalised gap closure (proportion of the gap between initial HV and true HV that is closed) confirms this: both GP strategies converge near 1.0 while random search plateaus far below. The key practical finding is that any GP-based Thompson sampling is far superior to random acquisition for Pareto front discovery — the specific GP variant (independent vs MOGP) matters much less than the choice to use GP uncertainty at all.

### Discovered Pareto Fronts

<p align="center">
  <img src="outputs/05c_pareto_fronts.png" width="900"/>
</p>

After 70 total evaluations, random search produces a sparse, poorly concentrated front. Both GP strategies identify points near the true front (red dashed line), with LCM achieving marginally better coverage of the extreme pollution events.

---

## 6. Comparison with Deep Ensemble MLP

A deep ensemble of K=15 MLPs (architecture selected by grid search: best is (128, 64) hidden layers, learning rate 1e-3) provides a non-GP baseline with approximate uncertainty estimates. Ensemble variance approximates epistemic uncertainty; residual variance from the training set approximates aleatoric uncertainty.

<p align="center">
  <img src="outputs/06a_nn_parity.png" width="800"/>
</p>

**Test-set results:**

| Output | RMSE | NLPD |
|--------|------|------|
| CO (mg/m³) | 0.343 | 0.358 |
| C6H6 (µg/m³) | 0.106 | -0.385 |
| NOx (ppb) | 46.35 | 5.176 |
| NO2 (µg/m³) | 15.08 | 4.108 |

### Full-Data Comparison

<p align="center">
  <img src="outputs/06b_model_comparison_bar.png" width="900"/>
</p>

**RMSE at full data (lower is better):**

| Model | CO | C6H6 | NOx | NO2 |
|-------|-----|-------|-----|-----|
| Independent GP | 0.371 | **0.032** | 51.85 | 15.76 |
| ICM (Q=1) | 0.444 | 0.085 | 55.68 | 18.37 |
| LCM (Q=2) | 0.453 | 0.068 | 54.03 | 29.98 |
| Deep Ensemble MLP | **0.343** | 0.106 | **46.35** | **15.08** |

**NLPD at full data (lower is better):**

| Model | CO | C6H6 | NOx | NO2 |
|-------|-----|-------|-----|-----|
| Independent GP | 0.469 | **-0.508** | 5.308 | 4.258 |
| ICM (Q=1) | 3.381 | 2.554 | 8.759 | 7.363 |
| LCM (Q=2) | 0.628 | -0.072 | 5.365 | 4.818 |
| Deep Ensemble MLP | **0.358** | -0.385 | **5.176** | **4.108** |

The MLP result is genuinely surprising: it achieves the lowest RMSE on CO, NOx, and NO2, outperforming even the independent GP at this data scale. Its failure on C6H6 (RMSE 0.106 vs 0.032 for the independent GP) is striking. The C6H6 distribution is highly concentrated near zero with a long right tail; the ensemble members appear to overfit to different aspects of this distribution and their mean predictions are systematically biased. A GP prior with an RBF kernel imposes smoothness assumptions that happen to suit this tight, clustered distribution better than an unconstrained neural network.

The NLPD picture reveals the MLP's calibration advantage at full data: it achieves the lowest NLPD on CO (0.358), NOx (5.176), and NO2 (4.108), slightly better than the independent GP. ICM is catastrophically miscalibrated (CO: 3.381, C6H6: 2.554). The GP posterior variance derives directly from the kernel and data; the ensemble variance approximation performs comparably here because the ensemble is large (K=15) and the full training set provides sufficient diversity across members.

### Low-Data Regime with MLP

<p align="center">
  <img src="outputs/06c_nn_low_data_rmse.png" width="800"/>
</p>

At n=20, the MLP is catastrophic on C6H6 (RMSE roughly 18x worse than the independent GP). For CO, it is marginally worse than the best GP variant. Interestingly, the MLP outperforms LCM on NOx and NO2 at n=20, because LCM's many ARD parameters are impossible to optimize from 20 points — the MLP's relatively fewer effective degrees of freedom turn out to be a practical advantage at this extreme. The MLP error bars are extremely wide at small n, reflecting high sensitivity to the specific 20 points selected. By n=160, GP models consistently outperform MLP across all outputs.

<p align="center">
  <img src="outputs/06d_nn_low_data_nlpd.png" width="800"/>
</p>

Calibration at small n reveals the key weakness of ensemble uncertainty. For CO at n=20, both the independent GP and the MLP spike to roughly 35–40 nats NLPD, assigning near-zero probability to true values. For C6H6, the independent GP reaches ~20–25 nats while the MLP stays elevated throughout (NLPD ~2 even at n=160 for C6H6, versus near-zero for the independent GP). ICM and LCM are significantly better calibrated for CO and C6H6 at n=20 (5–10 nats for CO), with the cross-output coupling providing regularisation that the ensemble cannot replicate. The fundamental issue with ensemble uncertainty at small n is that ensemble members have not seen enough data to disagree in informative ways, so variance estimates become unreliable. GP posterior variance is derived from the kernel and the data, which makes it more principled — but as shown, even this can fail (independent GP at n=20 for CO/C6H6) without the regularising effect of cross-output coupling.

---

## Summary

| | **RMSE** | **NLPD** | **Low-data calibration** | **BO surrogate** |
|---|---|---|---|---|
| Independent GP | Best across all outputs at full data | Best on C6H6; poor on CO/C6H6 at n=20 | Collapses at n=20 for CO/C6H6 | Moderate (90–93% HV) |
| ICM (Q=1) | Worse than Independent GP on all outputs | Catastrophically miscalibrated (CO: 3.38, C6H6: 2.55) | Better than Ind. GP for CO at n=20 | N/A |
| LCM (Q=2) | Worse than Independent GP on all outputs; NO2 regression severe | Better than ICM; reasonable except NO2/NOx at n=80 | Best for CO/C6H6 at n=20 | **Fastest early convergence** |
| Deep Ensemble MLP | Best on CO, NOx, NO2; fails on C6H6 | Competitive at full data; catastrophic at n=20 | Worst at n=20 | N/A |

**Key findings:**

1. **At full data, coregionalization does not help.** Despite high inter-pollutant correlations (r=0.60–0.93), both ICM and LCM are outperformed by independent GPs on all four outputs at 4,858 training points. With sufficient data, each GP learns the input space thoroughly without needing cross-output borrowing, and the additional optimization burden of coregionalization parameters is a net cost.

2. **Scale imbalance degrades coregionalization.** The ICM coregionalization matrix is dominated by NOx (entries: 3604 NOx–NOx, 405 NOx–NO2, ~0 for CO/C6H6), because marginal likelihood optimization is sensitive to raw output variance. The single shared latent process is allocated almost entirely to the highest-variance output, leaving CO and C6H6 poorly modelled. LCM partially mitigates this but exhibits the same pathology.

3. **The low-data regime is where MOGPs deliver genuine value, but only on calibration.** At n=20, the independent GP NLPD spikes to 35–40 nats for CO and 20–25 nats for C6H6 — the posterior variance collapses without enough data. ICM and LCM stay at 5–10 nats for CO by exploiting cross-output coupling. This calibration advantage is output-specific (absent for NOx, negative for LCM on NO2) and not reflected in RMSE, where the independent GP remains competitive.

4. **GP-based Bayesian optimization dramatically outperforms random search.** Both independent GP and LCM acquire ~90–93% of the true hypervolume in 70 evaluations, versus ~39% for random search. The choice of GP variant (independent vs MOGP) has a much smaller effect than the choice to use GP-guided acquisition at all.

5. **The MLP is competitive at full data but brittle.** Its RMSE on CO (0.343), NOx (46.35), and NO2 (15.08) outperforms all GP variants at full data — a clean result for neural networks at scale. But C6H6 (RMSE 0.106 vs 0.032 for independent GP) exposes sensitivity to the output's skewed distribution, and the ensemble uncertainty is unreliable at n≤40.

---

## How to Run

**Dependencies (GPyTorch-based, tested on Python 3.10):**

```bash
pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install gpytorch==1.12 linear_operator==0.5.0
pip install numpy pandas scikit-learn matplotlib seaborn scipy requests
```

For CPU-only (no GPU), substitute `torch==2.3.1` without the `+cu121` flag.

**Run scripts in order:**

```bash
python 01_eda.py
python 02_independent_gps.py
python 03_icm_vs_lcm.py
python 04_low_data_regime.py
python 05_pareto_optimization.py   # ~15–20 min (pool-based active learning, 5 seeds)
python 06_nn_comparison.py
```

All outputs (plots + JSON results) are saved to `outputs/`. The dataset downloads automatically from UCI on first run. For the full-data GP run, a GPU is strongly recommended: the A100 on CSF3 reduces training time from several hours (CPU) to minutes per model. The SLURM submission script is at `submit_mogp.slurm`.

---

## Dataset & References

- Dataset: [UCI Air Quality Data Set #360](https://archive.ics.uci.edu/dataset/360/air+quality) — De Vito et al. (2008)
- GP framework: [GPyTorch](https://gpytorch.ai/) (Gardner et al., 2018)
- ICM/LCM: Álvarez & Lawrence (2011), *Computationally efficient convolved multiple output Gaussian processes*
- Multi-objective BO: Emmerich et al. (2006), hypervolume indicator; Thompson (1933), sampling-based acquisition
