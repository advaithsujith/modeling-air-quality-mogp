# Modeling Air Quality with Multi-Output Gaussian Processes

This project investigates whether Multi-Output Gaussian Processes (MOGPs) can outperform independent models on urban air quality prediction, and whether they can guide efficient data collection through Bayesian optimization. The central question is practical: reference-grade air quality analysers are expensive to deploy, so can I get more information out of fewer measurements by exploiting the fact that pollutants are correlated?

I use the UCI Air Quality dataset (De Vito et al., 2008): hourly readings from an Italian city between March 2004 and February 2005, comprising 5 metal-oxide sensor channels, 3 meteorological variables, and 4 reference analyser outputs (CO, Benzene/C6H6, NOx, NO2). After cleaning, ~6000 usable rows remain. GP inference scales as O(n³), so GP models are trained on a random subsample of 500 rows (split roughly 350 train / 50 val / 100 test); the neural network and Bayesian optimisation experiments use the same splits for consistency.

I compare three GP model families: independent GPs, ICM (Intrinsic Coregionalization Model), and LCM (Linear Coregionalization Model, Q=2 latents), along with a deep ensemble of MLPs as a non-GP baseline.

---

## 1. Exploratory Data Analysis

### Missing Data

The first thing I address is missingness. The dataset uses -200 as a sentinel for missing values, which I replace with NaN before any modelling.

<p align="center">
  <img src="outputs/01a_missing_data.png" width="750"/>
</p>

NMHC(GT) is >90% missing and I drop it entirely; no imputation strategy can recover a column with that little signal. The four reference analyser outputs (CO, NOx, NO2, and to a lesser extent C6H6) each have 15-20% missingness, but since I drop any row where *any* target is missing, the effective dataset shrinks further. Sensor columns and meteorological variables have <10% missing and I median-impute them using training-set medians only (imputation is deferred to after the train/test split to avoid data leakage). This asymmetry matters: the reference analysers (expensive) are more often offline than the cheap metal-oxide sensors.

### Output Distributions

<p align="center">
  <img src="outputs/01b_output_distributions.png" width="800"/>
</p>

All four outputs are right-skewed with heavy tails, a pattern typical of pollutant concentrations. CO has a median of ~1.5 mg/m3 with occasional spikes above 8; C6H6 is tightly clustered near 6 ug/m3 with a long tail; NOx has a median around 166 ppb but extends past 1000 ppb during high-traffic events; NO2 is the most Gaussian-looking of the four with a median around 110 ug/m3. The scale differences across outputs (from single-digit CO to triple-digit NOx) matter when interpreting the coregionalization matrices later.

### Output Correlations

<p align="center">
  <img src="outputs/01c_output_correlation.png" width="420"/>
</p>

This is the key validation for using a MOGP. Everything, especially CO and C6H6, are pretty positively correlated (MOGPs perform better on positive correlations) (r=0.930). Both are direct combustion products that peak together during traffic rush hours. CO-NOx (r=0.786) and C6H6-NOx (r=0.718) are also strong. NO2 is the weakest correlate (r=0.674 with CO, 0.603 with C6H6), which makes sense: NO2 involves secondary photochemical reactions and has a different diurnal cycle than the primary combustion pollutants. All off-diagonal correlations are positive and substantial, which is exactly what I need for coregionalization to be beneficial.

<p align="center">
  <img src="outputs/01f_output_pairplot.png" width="650"/>
</p>

The pairplot confirms these are not just linear relationships; the joint densities show clear elongated structures, especially between CO, C6H6 and NOx. There is also visible heteroscedasticity: variance grows with the mean for most pairs, which motivates GP models that can capture this.

### Temporal Structure

<p align="center">
  <img src="outputs/01d_temporal_patterns.png" width="800"/>
</p>

The hourly profiles are informative. CO and NOx show classic double-peak patterns: morning rush (7-9am) and evening peak (6-8pm), with a midday dip as atmospheric mixing disperses pollutants. C6H6 follows CO very closely. NO2, by contrast, rises through the day and peaks in the evening, reflecting its dependence on photochemical reactions and secondary formation from NOx. Seasonally, CO and NOx are elevated in winter (Jan, Feb, Nov, Dec), consistent with heating emissions and reduced atmospheric boundary layer height. NO2 is more flat across months. These patterns motivated me to add cyclic hour features (sin/cos encoding of hour of day) rather than raw hour integers, which would create a 23 to 0 discontinuity.

### Feature Importance

<p align="center">
  <img src="outputs/01e_feature_output_correlation.png" width="500"/>
</p>

The sensor columns dominate. S1(CO) is strongly correlated with CO (r=0.88) and C6H6 (r=0.91); the metal-oxide CO sensor tracks both combustion pollutants well. S2(NMHC) shows similar behaviour. Meteorological variables (T, RH, AH) have weak correlations with all outputs, though temperature has a modest negative relationship with NOx and NO2, consistent with photochemistry being more active in warmer conditions. The cyclic hour features (sin/cos) show moderate correlations, confirming that time of day carries predictive signal beyond what the sensors capture.

---

## 2. Baseline: Independent GPs

Before introducing any cross-output structure, I fit one ARD-RBF GP independently to each output. This is a strong baseline: it can learn feature-specific lengthscales per output and model uncertainty, but treats the four pollutants as if they are completely unrelated.

<p align="center">
  <img src="outputs/02a_igp_predictions.png" width="800"/>
</p>

<p align="center">
  <img src="outputs/02b_igp_parity.png" width="800"/>
</p>

**Results:** C6H6 is predicted almost perfectly (RMSE=0.031, R²≈1.00); its tight distribution and strong sensor correlation make it an easy target. CO achieves RMSE=0.419. NOx has high absolute RMSE (79.6 ppb) because of its wide dynamic range (values span 0 to 1200 ppb) but the relative fit is reasonable (R²≈0.90). NO2 is the most challenging output (RMSE=23.8 ug/m3, R²≈0.70); its secondary photochemical nature means sensor readings are a noisier proxy.

### ARD Lengthscales

<p align="center">
  <img src="outputs/02c_ard_lengthscales.png" width="750"/>
</p>

The ARD lengthscales reveal which features each GP relies on. Relative humidity (RH) has an extremely large lengthscale for all outputs; the model essentially ignores it, treating the output as flat with respect to RH. This is consistent with RH having near-zero Pearson correlations with the pollutants. Temperature (T) has smaller lengthscales, confirming its modest relevance. The sensor columns generally have moderate lengthscales. The absolute values (tens of thousands) reflect the standardised input space; what matters is the relative ordering across features, not the magnitude.

---

## 3. Multi-Output GPs: ICM and LCM

### Model Comparison

<p align="center">
  <img src="outputs/03a_model_comparison.png" width="850"/>
</p>

ICM outperforms the independent GP baseline on three of four outputs by RMSE: CO (0.385 vs 0.419), NOx (69.5 vs 79.6), NO2 (20.9 vs 23.8). With ARD lengthscales enabled, LCM (Q=2) is strongly competitive across the board: CO (0.388), NOx (75.3 vs 79.6 for independent GP), and NO2 (21.2 vs 23.8), while only losing ground on C6H6 (0.157 vs 0.031). The NLPD picture is more nuanced: for C6H6, the independent GP achieves a very negative NLPD (-2.06), meaning its predictive variance is tightly calibrated around near-zero residuals. ICM's NLPD on C6H6 (0.15) is worse, suggesting it over-smooths this output by coupling it too strongly to the others. LCM's C6H6 NLPD (0.64) reflects the same trade-off; the two-latent structure improves NOx (5.60 vs 5.88) and CO (0.42 vs 0.55) calibration at the cost of slightly worse C6H6 calibration.

### The ICM Coregionalization Matrix

<p align="center">
  <img src="outputs/03b_icm_coregionalization.png" width="480"/>
</p>

The learned B matrix is dominated by the NOx-NOx diagonal entry (3604.06) and the NOx-NO2 cross-term (404.71). CO and C6H6 have near-zero entries. This is not a failure of the model; it is a direct consequence of the raw output units. NOx values range into the hundreds of ppb, while CO is in the single-digit mg/m3 range. The marginal likelihood naturally weights outputs by their variance, so the model allocates most of the shared latent process to the highest-variance output (NOx). In practice, this means the ICM is learning a strong NOx-NO2 coupling (physically sensible, as NO2 is formed partly from NOx oxidation) while largely treating CO and C6H6 independently, which explains why ICM improves mostly on NOx and NO2.

### The LCM Mixing Matrix

<p align="center">
  <img src="outputs/03c_lcm_mixing_and_lengthscales.png" width="750"/>
</p>

The LCM mixing matrix shows a similar scale-dominated pattern. Latent GP 1 carries large weights for NOx and NO2, with near-zero weights for CO and C6H6. Latent GP 2 has smaller weights across all outputs. The ARD lengthscale plot (first dimension of each latent's per-feature lengthscales) shows that the two latents have specialised differently, each learning its own feature relevance. This reflects how the two latent processes partition the input space to capture complementary aspects of the pollution dynamics.

### Parity Comparison and Model Evidence

<p align="center">
  <img src="outputs/03d_parity_comparison.png" width="850"/>
</p>

ICM achieves better R² than the independent GP on CO and NO2, the two outputs that benefit most from the NOx-NO2 coupling learned in B. With ARD enabled, LCM now matches or exceeds ICM on NOx and NO2 (RMSE 75.3 and 21.2 respectively), while ICM still holds the edge on raw NOx accuracy (RMSE 69.5). The two-latent structure redistributes predictive power differently across outputs: LCM is more balanced whereas ICM concentrates gains on the highest-variance outputs.

<p align="center">
  <img src="outputs/03e_nlml_comparison.png" width="480"/>
</p>

A notable result: the independent GP achieves the lowest NLML, followed by LCM and ICM. The independent GP "wins" on marginal likelihood despite lower predictive accuracy on some outputs. This is because the coregionalization models have substantially more parameters (LCM with ARD carries per-latent per-dimension lengthscales), and the marginal likelihood applies an automatic Occam's razor, penalising model complexity. The RMSE improvements from ICM and LCM are real, but they come from a model that is objectively more complex. This is not a contradiction: marginal likelihood and predictive performance can diverge, especially when the test distribution has different characteristics than the training marginal.

---

## 4. Low-Data Regime

The central motivation for MOGPs in settings like this is not just better accuracy at full data, but the ability to extrapolate more reliably when labelled data is scarce. Deploying a reference analyser for a year is expensive; what if you only have 40 measurements?

### RMSE vs Training Size

<p align="center">
  <img src="outputs/04a_low_data_rmse.png" width="800"/>
</p>

The picture in the low-data regime is mixed and output-dependent. For C6H6, the independent GP is consistently the best; its tight, well-correlated distribution rewards treating it independently, while ICM is notably poor at n=80 (RMSE over 2x the independent GP) and LCM shows an anomalous improvement at n=80 that does not persist. For NOx, LCM shows a clear optimisation failure at n=80 (RMSE spikes sharply before recovering at n=160), a consequence of fitting 38+ ARD parameters from only 80 points; the optimiser finds a poor local minimum. ICM is competitive for NOx at n=20-40 but degrades heavily for CO at n=160 (RMSE 0.87 vs 0.43 for the independent GP), suggesting its coregionalization structure can actively hurt calibration for specific outputs at certain data sizes. LCM is more stable than ICM overall in the low-data regime. All three models are evaluated at the same training sizes (n=20, 40, 80, 160), so the comparison is on equal footing throughout.

### Calibration (NLPD) vs Training Size

<p align="center">
  <img src="outputs/04b_low_data_nlpd.png" width="800"/>
</p>

The calibration picture is more nuanced than the RMSE story and splits clearly by output. For CO, the independent GP is the worst-calibrated model at n=20, with NLPD spiking to roughly 35-40 nats. C6H6 shows a similar but less extreme pattern, with the independent GP reaching around 20-25 nats at n=20. In both cases ICM and LCM stay much lower (CO: ~5-10 nats; C6H6: ~1-3 nats), meaning the cross-output coupling gives the MOGP posterior a more grounded variance estimate when data is scarce. For NOx, the independent GP is stably calibrated throughout while LCM shows the same n=80 spike seen in RMSE. For NO2, LCM is very poorly calibrated at n=20 (NLPD ~11), while ICM and the independent GP are more stable. By n=160, all three models converge to similar NLPD. The takeaway is that calibration performance at small n depends heavily on which output you care about: MOGPs help for CO/C6H6, hurt for NO2 (LCM), and are neutral for NOx. All three models are evaluated at the same training sizes (n=20, 40, 80, 160), so the comparison is on equal footing throughout.

### Predictive Uncertainty at n=40

<p align="center">
  <img src="outputs/04c_uncertainty_n40.png" width="900"/>
</p>

Both models are trained on the same 40 points and evaluated on the full test set. Test samples are sorted by their true target value (same layout as the full-data plots in section 2), so the shaded ±2σ band should track the black true-value dots to show a well-calibrated model. For C6H6 the independent GP is tighter and more accurate (RMSE 0.195 vs LCM's 0.264 at n=40). For NOx and NO2 the uncertainty bands are wide for both models, reflecting how hard these outputs are to characterise from only 40 points, but the independent GP is more reliable: LCM's NOx uncertainty balloons as its many ARD parameters fail to optimise from such a small sample.

---

## 5. Bayesian Optimization: Pareto Front Discovery

Beyond prediction, MOGPs can serve as surrogates in active learning loops. The question I pose here: given a pool of unlabelled time points (cheap sensor readings only), can a MOGP surrogate guide the selection of which points to label with the reference analyser, so as to map the CO-NO2 joint extremes as efficiently as possible?

### The True Pareto Front

<p align="center">
  <img src="outputs/05d_true_pareto_front.png" width="750"/>
</p>

The true CO-NO2 Pareto front (maximising both) spans the high-pollution region of the dataset and consists of 10 non-dominated points with total hypervolume 3443.2. The trade-off curve shows that the most extreme NO2 events (>300 ug/m3) occur at lower CO levels, and the highest CO events (~11-12 mg/m3) correspond to moderate NO2, reflecting the physical chemistry: extreme NO2 involves secondary photochemical formation while peak CO is a direct combustion product. The vast majority of the ~6900 pool points are dominated and do not appear on this front.

### Active Learning Comparison

I compare three strategies, each starting from 30 random initial observations and making 40 sequential selections:
- **Random search**: select uniformly at random from the pool
- **Independent GP + Thompson sampling**: fit two independent GPs, draw a posterior sample, find its Pareto front, select from it
- **LCM + Thompson sampling**: same procedure with a joint MOGP surrogate

<p align="center">
  <img src="outputs/05a_hypervolume_trace.png" width="720"/>
</p>

Random search achieves roughly 39% of the true hypervolume after 70 evaluations; it stumbles onto some non-dominated points by chance but never efficiently targets the high-pollution corner. Both GP strategies are dramatically better, each reaching around 90-93% of the true hypervolume. The two active learning strategies are closely matched; LCM shows faster initial convergence in some runs by exploiting CO-NO2 posterior correlations, but by N=40 the 5-run averages are within error bands of each other. The consistent finding is that any GP-based Thompson sampling is far superior to random search.

<p align="center">
  <img src="outputs/05b_normalised_hv.png" width="720"/>
</p>

The normalised HV gap closure confirms that both GP strategies converge near 1.0 while random search plateaus far below. LCM and independent GP+TS track closely across the budget, with LCM occasionally pulling ahead in early iterations due to richer posterior correlations. For settings where every evaluation is expensive (e.g., deploying a reference analyser for a day to measure a specific time window), either GP-based strategy provides a clear advantage over random acquisition.

### Discovered Pareto Fronts

<p align="center">
  <img src="outputs/05c_pareto_fronts.png" width="900"/>
</p>

The discovered fronts after 70 total evaluations confirm the quantitative story. Random search scatters evaluations across the full pool and produces a sparse, low-quality front. Both GP strategies identify points near the true front (red dashed line), with LCM achieving marginally better coverage of the extreme pollution events.

---

## 6. Comparison with Deep Ensemble MLP

I also compare against a deep ensemble of 15 MLPs (architecture selected by grid search on the validation set) as a non-GP baseline with uncertainty estimates. The ensemble variance approximates epistemic uncertainty; residual variance from the training set approximates aleatoric uncertainty.

<p align="center">
  <img src="outputs/06a_nn_parity.png" width="800"/>
</p>

The most striking failure is C6H6: the MLP achieves R2=0.990 visually but RMSE=0.777, far worse than the independent GP's 0.031. The C6H6 distribution is highly concentrated near zero with a long right tail; the MLP's predictions are systematically too high across the board, suggesting it cannot handle this distributional shape as well as a GP prior. For CO (R2=0.927) and NOx (R2=0.884), the MLP is respectable. NO2 (R2=0.722) is the weakest, similar to the GP story.

### Full-Data Comparison

<p align="center">
  <img src="outputs/06b_model_comparison_bar.png" width="900"/>
</p>

**RMSE at full data:**

| Model | CO | C6H6 | NOx | NO2 |
|---|---|---|---|---|
| Independent GP | 0.419 | **0.031** | 79.6 | 23.8 |
| ICM (Q=1) | **0.385** | 0.202 | **69.5** | **20.9** |
| LCM (Q=2) | 0.388 | 0.157 | 75.3 | 21.2 |
| Deep Ensemble MLP | 0.423 | 0.777 | 78.8 | 22.4 |

**NLPD at full data (lower = better calibration):**

| Model | CO | C6H6 | NOx | NO2 |
|---|---|---|---|---|
| Independent GP | 0.552 | **-2.061** | 5.878 | 4.597 |
| ICM (Q=1) | 0.434 | 0.153 | 5.901 | **4.464** |
| LCM (Q=2) | **0.421** | 0.643 | **5.604** | 4.554 |
| Deep Ensemble MLP | 0.532 | 1.256 | 5.790 | 4.563 |

After the methodology fixes (K=15 ensemble, ARD-consistent comparison), the MLP's NOx NLPD (5.79) is now comparable to the independent GP (5.88) and ICM (5.90), though still above LCM (5.60). The clearest calibration failures are on C6H6 (NLPD 1.26 vs independent GP's -2.06) and CO (0.53 vs LCM's 0.42), where the ensemble variance poorly captures the tight, skewed distribution. The GP predictive variance has a principled derivation from the kernel and the data; the ensemble variance is an approximation that is most reliable when ensemble members disagree in informative ways, which requires sufficient training data.

### Low-Data Regime with MLP

<p align="center">
  <img src="outputs/06c_nn_low_data_rmse.png" width="800"/>
</p>

At n=20, the MLP is catastrophic on C6H6 (RMSE ~3.6, roughly 13x worse than LCM's ~0.27). The independent GP outperforms all other models for C6H6 at all training sizes, but even it is an order of magnitude better than the MLP here. For CO, MLP is marginally worse than LCM at n=20. Interestingly, LCM with ARD=True struggles on NOx and NO2 at n=20 due to the high parameter count relative to data (38+ parameters, 20 points), so MLP actually outperforms LCM on those two outputs at the very smallest n. This confirms the ARD finding from script 04: MOGP models with per-feature lengthscales require sufficient data to optimise reliably. By n=160, GP models consistently outperform MLP across all outputs. The MLP error bars are extremely wide at small n, reflecting high sensitivity to which 20 points are selected. All four models are evaluated at the same training sizes (n=20, 40, 80, 160), so the comparison is on equal footing throughout.

<p align="center">
  <img src="outputs/06d_nn_low_data_nlpd.png" width="800"/>
</p>

The calibration picture at small n reveals an important split. For CO, both the independent GP and the MLP spike to roughly 35-40 nats NLPD at n=20, assigning near-zero probability to true values. For C6H6, the independent GP reaches around 20-25 nats while the MLP stays elevated as well. In both cases ICM and LCM are significantly better calibrated at n=20, benefiting from cross-output information sharing. For C6H6, the MLP NLPD stays high throughout (NLPD ~2 at n=160, versus near-zero for the independent GP), reflecting its inability to model the tight, skewed distribution. For NOx, the MLP is roughly flat across n while LCM shows the n=80 calibration spike. The core weakness of ensemble uncertainty remains: at small n the members have not seen enough data to disagree meaningfully, so the estimated uncertainty is unreliable. GP posterior variance is derived from the kernel and the data, which makes it more principled, but the independent GP variant still collapses at n=20 for CO and C6H6 without the regularising effect of cross-output coupling.

---

## Summary

| | **Accuracy (RMSE)** | **Calibration (NLPD)** | **Low-data stability** | **BO surrogate** |
|---|---|---|---|---|
| Independent GP | Good; **best on C6H6** | Poor on CO/C6H6 at small n | **Most stable overall** | Moderate |
| ICM (Q=1) | Best on NOx at small n; poor on CO at n=160 | Moderate | Unstable (CO degrades) | N/A |
| LCM (Q=2) | Competitive; NOx spike at n=80 | Best on CO/C6H6 at small n; unstable on NO2 | More stable than ICM | **Fastest BO convergence** |
| Deep Ensemble MLP | Fails on C6H6 | Worst across all | **Worst** | N/A |

**Key findings:**
1. All pollutant outputs are highly correlated (r=0.60-0.93), validating the MOGP premise.
2. ICM delivers the best raw predictive accuracy at full data (lowest NOx RMSE), exploiting the NOx-NO2 correlation. At small n it is more stable than LCM.
3. In the low-data regime the independent GP is the most consistent performer on RMSE: it is best on C6H6 at all training sizes and competitive on CO, while ICM degrades badly on CO at n=160 and LCM spikes on NOx at n=80. The main advantage of MOGPs at small n is calibration (NLPD): cross-output coupling prevents the posterior variance from collapsing, where the independent GP spikes to ~35-40 nats for CO and ~20-25 nats for C6H6 at n=20. These calibration gains are output-specific (LCM is poorly calibrated on NO2 at n=20 and crashes on NOx at n=80), so neither coregionalization model reliably dominates across all outputs in the low-data setting.
4. In the Bayesian optimisation experiment, both GP strategies dramatically outperform random search (~39% of true HV). Both reach ~90-93% of the true hypervolume; LCM shows faster early convergence in some seeds by exploiting CO-NO2 posterior correlations, but the two strategies are closely matched by N=40.
5. The deep ensemble MLP fails catastrophically on C6H6 at all data sizes and has poor calibration at small n. Its uncertainty estimates are unreliable at n=20-40 regardless of output, unlike GPs where the posterior variance is principled even with few points.

---

## How to Run

```bash
pip install gpy numpy pandas scikit-learn matplotlib seaborn requests
```

Run scripts in order:

```bash
python 01_eda.py
python 02_independent_gps.py
python 03_icm_vs_lcm.py
python 04_low_data_regime.py
python 05_pareto_optimization.py   # ~15-20 min
python 06_nn_comparison.py
```

All outputs (plots + JSON results) are saved to `outputs/`. The dataset downloads automatically from UCI on first run.

---

## Dependencies

- [GPy](https://github.com/SheffieldML/GPy) for GP models and coregionalization kernels
- NumPy, pandas, scikit-learn, matplotlib, seaborn
- Python 3.10+
