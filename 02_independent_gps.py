# %% [markdown]
# # 02 — Baseline: Independent Single-Output GPs
#
# One ARD-RBF GP per output, trained independently. No information sharing
# across outputs — this is what we're trying to beat with the MOGP models.

# %%
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_utils import (
    load_raw, get_Xy, split_and_scale, unscale_predictions,
    FEATURE_NAMES, SHORT_FEATURE_NAMES, OUTPUT_NAMES, SHORT_OUTPUT_NAMES,
    GP_SUBSAMPLE,
)
from gp_models import IndependentGP
from sklearn.metrics import r2_score
from evaluation import rmse, nlpd, summary_table

sns.set_theme(style="whitegrid", font_scale=1.1)
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(42)

def savefig(name):
    plt.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches="tight")
    plt.show()

# %%
print("Loading and splitting data...")
X, Y = get_Xy()
splits = split_and_scale(X, Y, test_size=0.2, val_size=0.1,
                         n_subsample=GP_SUBSAMPLE, random_state=42)

print(f"  Full cleaned dataset : {len(X)} rows")
print(f"  GP subsample         : {GP_SUBSAMPLE}")
print(f"  Train : {splits['n_train']}  Val : {splits['n_val']}  Test : {splits['n_test']}")

# %% [markdown]
# ## Fit Independent GPs

# %%
print("\nFitting Independent GPs (T=4 outputs, ~2 min)...")
igp = IndependentGP(ARD=True, n_restarts=3)
igp.fit(splits["X_train"], splits["Y_train"])
print("Done.")
print(f"  Combined NLML : {igp.nlml:.2f}")

# %% [markdown]
# ## Test-Set Evaluation

# %%
mu_test_scaled, var_test_scaled = igp.predict(splits["X_test"])
mu_test, var_test = unscale_predictions(mu_test_scaled, var_test_scaled, splits["scaler_Y"])
T = splits["Y_test"].shape[1]

results = {"Independent GP": {}}
for t in range(T):
    results["Independent GP"][f"Y{t+1} RMSE"] = rmse(splits["Y_test"][:, t], mu_test[:, t])
    results["Independent GP"][f"Y{t+1} NLPD"] = nlpd(splits["Y_test"][:, t], mu_test[:, t], var_test[:, t])

print("\n" + summary_table(results))

# %% [markdown]
# ## Prediction vs Ground Truth

# %%
output_colors = ["#4878CF", "#E87722", "#6ACC65", "#D62728"]

fig, axes = plt.subplots(1, T, figsize=(16, 4.5))
for i, (ax, col, oname) in enumerate(zip(axes, output_colors, OUTPUT_NAMES)):
    y_true = splits["Y_test"][:, i]
    y_pred = mu_test[:, i]
    y_std  = np.sqrt(var_test[:, i])

    order  = np.argsort(y_true)
    x_plot = np.arange(len(y_true))

    ax.fill_between(
        x_plot,
        y_pred[order] - 2 * y_std[order],
        y_pred[order] + 2 * y_std[order],
        alpha=0.2, color=col, label="±2σ"
    )
    ax.plot(x_plot, y_pred[order], color=col, lw=1.5, label="Predicted mean")
    ax.plot(x_plot, y_true[order], "k.", ms=3, alpha=0.5, label="True")
    ax.set_xlabel("Test sample (sorted by true value)")
    ax.set_ylabel(oname, fontsize=8)
    ax.set_title(f"Y{i+1}: {SHORT_OUTPUT_NAMES[i]}\nRMSE={results['Independent GP'][f'Y{i+1} RMSE']:.3f}",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, frameon=False)

fig.suptitle("Independent GP: Predictions on Test Set", fontsize=12, fontweight="bold")
plt.tight_layout()
savefig("02a_igp_predictions.png")

# %% [markdown]
# ## Parity Plots (Predicted vs True)

# %%
fig, axes = plt.subplots(1, T, figsize=(14, 4))
for i, (ax, col, oname) in enumerate(zip(axes, output_colors, OUTPUT_NAMES)):
    y_true = splits["Y_test"][:, i]
    y_pred = mu_test[:, i]
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, color=col, edgecolors="none")
    lims = [min(y_true.min(), y_pred.min()) - 0.5,
            max(y_true.max(), y_pred.max()) + 0.5]
    ax.plot(lims, lims, "k--", lw=1.2, label="Perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)
    r2 = r2_score(y_true, y_pred)
    ax.set_xlabel(f"True {SHORT_OUTPUT_NAMES[i]}", fontsize=9)
    ax.set_ylabel(f"Pred {SHORT_OUTPUT_NAMES[i]}", fontsize=9)
    ax.set_title(f"Y{i+1}: RMSE={results['Independent GP'][f'Y{i+1} RMSE']:.3f}  R²={r2:.3f}",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, frameon=False)

fig.suptitle("Independent GP: Parity Plots", fontsize=12, fontweight="bold")
plt.tight_layout()
savefig("02b_igp_parity.png")

# %% [markdown]
# ## ARD Lengthscales — Feature Importance

# %%
ls_all = igp.lengthscales()   # list of T arrays, each length D

x_pos = np.arange(len(SHORT_FEATURE_NAMES))
width = 0.8 / T

fig, ax = plt.subplots(figsize=(13, 5))
for i, (ls, col, sname) in enumerate(zip(ls_all, output_colors, SHORT_OUTPUT_NAMES)):
    offset = (i - T / 2 + 0.5) * width
    ax.bar(x_pos + offset, ls, width, label=f"Y{i+1}: {sname}", color=col, alpha=0.8)

ax.set_xticks(x_pos)
ax.set_xticklabels(SHORT_FEATURE_NAMES, rotation=25, ha="right", fontsize=9)
ax.set_ylabel("ARD Lengthscale  (smaller = more important)")
ax.set_title("ARD Lengthscales per Output — Independent GP", fontweight="bold")
ax.legend(frameon=False)
plt.tight_layout()
savefig("02c_ard_lengthscales.png")

# %% [markdown]
# ## Residual Distributions

# %%
fig, axes = plt.subplots(1, T, figsize=(14, 4))
for i, (ax, col, oname) in enumerate(zip(axes, output_colors, OUTPUT_NAMES)):
    resid = splits["Y_test"][:, i] - mu_test[:, i]
    ax.hist(resid, bins=25, color=col, edgecolor="white", alpha=0.85)
    ax.axvline(0, color="k", lw=1.5, linestyle="--")
    ax.set_xlabel("Residual (True − Predicted)")
    ax.set_title(f"Y{i+1}: {SHORT_OUTPUT_NAMES[i]}", fontsize=9, fontweight="bold")

fig.suptitle("Independent GP: Residual Histograms", fontweight="bold")
plt.tight_layout()
savefig("02d_igp_residuals.png")

# %%
print("\n=== Baseline Results (to beat in notebooks 03–05) ===")
print(summary_table(results))

import json
with open(os.path.join(OUT_DIR, "baseline_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to outputs/baseline_results.json")
