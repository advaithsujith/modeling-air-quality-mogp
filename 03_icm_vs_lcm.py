# %% [markdown]
# # 03 — ICM and LCM Multi-Output GPs
#
# Fitting the two coregionalization models and comparing against the independent
# GP baseline from script 02. ICM uses a single shared kernel across outputs;
# LCM uses Q=2 latent GPs with independent lengthscales.

# %%
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_utils import (
    get_Xy, split_and_scale,
    OUTPUT_NAMES, SHORT_OUTPUT_NAMES, SHORT_FEATURE_NAMES, GP_SUBSAMPLE,
)
from gp_models import ICM, LCM, IndependentGP
from sklearn.metrics import r2_score
from evaluation import rmse, nlpd, summary_table

sns.set_theme(style="whitegrid", font_scale=1.1)
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(42)

def savefig(name):
    plt.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches="tight")
    plt.show()

output_colors = ["#4878CF", "#E87722", "#6ACC65", "#D62728"]

# %%
print("Loading data...")
X, Y = get_Xy()
splits = split_and_scale(X, Y, test_size=0.2, val_size=0.1,
                         n_subsample=GP_SUBSAMPLE, random_state=42)
T = splits["Y_test"].shape[1]
print(f"  Train/Val/Test = {splits['n_train']}/{splits['n_val']}/{splits['n_test']}")

try:
    with open(os.path.join(OUT_DIR, "baseline_results.json")) as f:
        all_results = json.load(f)
    print("  Loaded baseline results from script 02.")
except FileNotFoundError:
    print("  WARNING: run 02_independent_gps.py first.")
    all_results = {}

# %% [markdown]
# ## Fit ICM  (Q = 1)

# %%
print("\nFitting ICM (Q=1) ...")
icm = ICM(W_rank=1, ARD=True, n_restarts=2)
icm.fit(splits["X_train"], splits["Y_train"])
print(f"  ICM NLML : {icm.nlml:.2f}")

mu_icm, var_icm = icm.predict(splits["X_test"])
all_results["ICM (Q=1)"] = {}
for t in range(T):
    all_results["ICM (Q=1)"][f"Y{t+1} RMSE"] = rmse(splits["Y_test"][:, t], mu_icm[:, t])
    all_results["ICM (Q=1)"][f"Y{t+1} NLPD"] = nlpd(splits["Y_test"][:, t], mu_icm[:, t], var_icm[:, t])
print(summary_table({"ICM (Q=1)": all_results["ICM (Q=1)"]}))

# %% [markdown]
# ## Fit LCM  (Q = 2)

# %%
print("\nFitting LCM (Q=2) ...")
lcm = LCM(num_latents=2, W_rank=1, ARD=True, n_restarts=2)
lcm.fit(splits["X_train"], splits["Y_train"])
print(f"  LCM (Q=2) NLML : {lcm.nlml:.2f}")

mu_lcm, var_lcm = lcm.predict(splits["X_test"])
all_results["LCM (Q=2)"] = {}
for t in range(T):
    all_results["LCM (Q=2)"][f"Y{t+1} RMSE"] = rmse(splits["Y_test"][:, t], mu_lcm[:, t])
    all_results["LCM (Q=2)"][f"Y{t+1} NLPD"] = nlpd(splits["Y_test"][:, t], mu_lcm[:, t], var_lcm[:, t])

print("\n=== Full Comparison Table ===")
print(summary_table(all_results))

# %% [markdown]
# ## Performance Comparison Bar Chart

# %%
models_list = list(all_results.keys())
model_palette = {"Independent GP": "#888888", "ICM (Q=1)": "#E87722", "LCM (Q=2)": "#4878CF"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, metric in zip(axes, ["RMSE", "NLPD"]):
    x = np.arange(T)
    width = 0.25
    for j, (model_name, color) in enumerate(model_palette.items()):
        if model_name not in all_results:
            continue
        vals = [all_results[model_name].get(f"Y{t+1} {metric}", float("nan"))
                for t in range(T)]
        offset = (j - 1) * width
        bars = ax.bar(x + offset, vals, width, label=model_name,
                      color=color, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.002 * max(vals),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7, color="dimgray")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Y{t+1}\n{SHORT_OUTPUT_NAMES[t]}" for t in range(T)], fontsize=9)
    ax.set_ylabel(metric)
    ax.set_title(f"Test {metric} by Model and Output", fontweight="bold")
    ax.legend(frameon=False, fontsize=9)

fig.suptitle("Independent GP vs. ICM vs. LCM — Test Performance", fontsize=12, fontweight="bold")
plt.tight_layout()
savefig("03a_model_comparison.png")

# %% [markdown]
# ## ICM Coregionalization Matrix  B  (4×4)

# %%
def get_B_matrix(model):
    for part in model.model.kern.parts:
        for sub in getattr(part, "parts", [part]):
            if hasattr(sub, "B"):
                return np.array(sub.B)
    return None

B = get_B_matrix(icm)
if B is not None:
    print("ICM coregionalization matrix B (4×4):")
    print(np.round(B, 3))

    labels = [f"Y{t+1}\n{SHORT_OUTPUT_NAMES[t]}" for t in range(T)]
    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = np.abs(B).max()
    im = ax.imshow(B, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(T)); ax.set_yticks(range(T))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(T):
        for j in range(T):
            ax.text(j, i, f"{B[i, j]:.2f}", ha="center", va="center",
                    fontsize=9,
                    color="white" if abs(B[i, j]) > vmax * 0.5 else "black")
    ax.set_title("ICM Coregionalization Matrix  B = WWᵀ + diag(κ)", fontweight="bold")
    plt.tight_layout()
    savefig("03b_icm_coregionalization.png")

# %% [markdown]
# ## LCM Mixing Matrix  W  (4×2)

# %%
W = lcm.mixing_matrix()   # (T, Q)
if W is not None:
    print("LCM (Q=2) Mixing Matrix W  (rows=outputs, cols=latent GPs):")
    for t in range(T):
        print(f"  Y{t+1} {SHORT_OUTPUT_NAMES[t]:<8}: {W[t]}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: heatmap of W
    ax = axes[0]
    vmax_w = np.abs(W).max()
    im = ax.imshow(W, cmap="RdBu_r", vmin=-vmax_w, vmax=vmax_w, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(W.shape[1]))
    ax.set_xticklabels([f"Latent GP {q+1}\n(u_{q+1})" for q in range(W.shape[1])], fontsize=9)
    ax.set_yticks(range(T))
    ax.set_yticklabels([f"Y{t+1}: {SHORT_OUTPUT_NAMES[t]}" for t in range(T)], fontsize=9)
    for i in range(T):
        for j in range(W.shape[1]):
            ax.text(j, i, f"{W[i, j]:.2f}", ha="center", va="center",
                    fontsize=10,
                    color="white" if abs(W[i, j]) > vmax_w * 0.5 else "black")
    ax.set_title("LCM Mixing Matrix  W\n(T × Q loading matrix)", fontweight="bold")

    # Right: per-latent ARD lengthscales (first dimension shown; LCM uses ARD=True)
    ax = axes[1]
    ls = lcm.latent_lengthscales()   # (Q, 1) or (Q, D) depending on ARD
    if ls is not None:
        ls_vals = ls[:, 0] if ls.ndim > 1 else ls
        colors_latent = ["#E87722", "#9B59B6"]
        ax.bar(range(len(ls_vals)), ls_vals,
               color=colors_latent[:len(ls_vals)], edgecolor="white", alpha=0.85)
        ax.set_xticks(range(len(ls_vals)))
        ax.set_xticklabels([f"Latent GP {q+1}" for q in range(len(ls_vals))], fontsize=10)
        ax.set_ylabel("Lengthscale (first dimension)")
        ax.set_title("LCM: Latent GP Lengthscales (ARD)\n(each latent GP learns its own feature relevance)",
                     fontweight="bold")

    plt.tight_layout()
    savefig("03c_lcm_mixing_and_lengthscales.png")

# %% [markdown]
# ## Parity Plots: ICM vs LCM side-by-side

# %%
fig, axes = plt.subplots(2, T, figsize=(16, 9))
for col_idx in range(T):
    y_true = splits["Y_test"][:, col_idx]
    col = output_colors[col_idx]
    oname = SHORT_OUTPUT_NAMES[col_idx]

    for row_idx, (model_name, mu_p, _) in enumerate([
        ("ICM (Q=1)",  mu_icm, var_icm),
        ("LCM (Q=2)",  mu_lcm, var_lcm),
    ]):
        ax = axes[row_idx, col_idx]
        ax.scatter(y_true, mu_p[:, col_idx], alpha=0.35, s=12,
                   color=col, edgecolors="none")
        lims = [min(y_true.min(), mu_p[:, col_idx].min()) - 0.3,
                max(y_true.max(), mu_p[:, col_idx].max()) + 0.3]
        ax.plot(lims, lims, "k--", lw=1.2)
        ax.set_xlim(lims); ax.set_ylim(lims)
        r2 = r2_score(y_true, mu_p[:, col_idx])
        rmse_val = all_results[model_name][f"Y{col_idx+1} RMSE"]
        ax.set_title(f"{model_name}  Y{col_idx+1}: {oname}\nRMSE={rmse_val:.3f}  R²={r2:.3f}",
                     fontsize=8, fontweight="bold")
        ax.set_xlabel(f"True {oname}", fontsize=8)
        ax.set_ylabel(f"Pred {oname}", fontsize=8)

fig.suptitle("Parity Plots: ICM vs LCM (Q=2)", fontsize=12, fontweight="bold")
plt.tight_layout()
savefig("03d_parity_comparison.png")

# %% [markdown]
# ## NLML Comparison  (Model Evidence)

# %%
igp_reload = IndependentGP(ARD=True, n_restarts=3)
igp_reload.fit(splits["X_train"], splits["Y_train"])

nlml_vals = {
    "Independent GP": igp_reload.nlml,
    "ICM (Q=1)":      icm.nlml,
    "LCM (Q=2)":      lcm.nlml,
}
print("\nNLML (lower = better model evidence):")
for name, val in nlml_vals.items():
    print(f"  {name:<20} NLML = {val:.2f}")

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(nlml_vals.keys(), nlml_vals.values(),
              color=["#888888", "#E87722", "#4878CF"],
              alpha=0.85, edgecolor="white")
ax.set_ylabel("NLML (lower = better)")
ax.set_title("Model Evidence Comparison (NLML)", fontweight="bold")
for bar, val in zip(bars, nlml_vals.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}", ha="center", fontsize=10, color="dimgray")
plt.tight_layout()
savefig("03e_nlml_comparison.png")

# %%
print("\n=== Final Results Summary ===")
print(summary_table(all_results))

with open(os.path.join(OUT_DIR, "all_model_results.json"), "w") as f:
    json.dump(all_results, f, indent=2)
print("Saved to outputs/all_model_results.json")
