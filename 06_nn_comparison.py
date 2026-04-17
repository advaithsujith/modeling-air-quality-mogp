# %% [markdown]
# # 06 — Neural Network Comparison
#
# Deep ensemble of MLPs (K=15 networks) as a non-GP uncertainty baseline.
# Compares accuracy (RMSE) and calibration (NLPD) against all three GP variants,
# and repeats the low-data ablation to see how quickly the NN degrades.

# %%
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import r2_score
from collections import defaultdict

warnings.filterwarnings("ignore")

from data_utils import (
    get_Xy, split_and_scale, subsample_train, unscale_predictions,
    OUTPUT_NAMES, SHORT_OUTPUT_NAMES, GP_SUBSAMPLE,
)
from gp_models import IndependentGP, ICM, LCM
from evaluation import rmse, nlpd, summary_table

sns.set_theme(style="whitegrid", font_scale=1.1)
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(42)
output_colors = ["#4878CF", "#E87722", "#6ACC65", "#D62728"]

def savefig(name):
    plt.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches="tight")
    plt.show()

# %%
print("Loading and splitting data...")
X, Y = get_Xy()
splits = split_and_scale(X, Y, test_size=0.2, val_size=0.1,
                         n_subsample=GP_SUBSAMPLE, random_state=42)
T = splits["Y_test"].shape[1]
print(f"  Train : {splits['n_train']}  Val : {splits['n_val']}  Test : {splits['n_test']}")

# %% [markdown]
# ## Architecture Search on Validation Set

# %%
PARAM_GRID = list(ParameterGrid({
    "hidden_layer_sizes": [(64, 64), (128, 64), (128, 128)],
    "alpha":              [1e-4, 1e-3],
}))

print("Running architecture search on validation set...")
best_val_rmse = np.inf
best_params   = None

for params in PARAM_GRID:
    m = MLPRegressor(**params, activation="relu", max_iter=2000,
                     random_state=0, learning_rate_init=1e-3)
    m.fit(splits["X_train"], splits["Y_train"])
    mu_val = m.predict(splits["X_val"])
    val_r  = np.mean([rmse(splits["Y_val"][:, t], mu_val[:, t]) for t in range(T)])
    if val_r < best_val_rmse:
        best_val_rmse = val_r
        best_params   = params

print(f"  Best params   : {best_params}")
print(f"  Val RMSE avg  : {best_val_rmse:.4f}")

# %% [markdown]
# ## Deep Ensemble (K=15)

# %%
K_ENSEMBLE = 15

def fit_ensemble(X_tr, Y_tr, X_val, Y_val, params, k=K_ENSEMBLE):
    models = []
    for seed in range(k):
        m = MLPRegressor(**params, activation="relu", max_iter=2000,
                         random_state=seed, learning_rate_init=1e-3)
        m.fit(X_tr, Y_tr)
        models.append(m)
    preds_val = np.stack([m.predict(X_val) for m in models], axis=0)  # (K, n_val, T)
    noise_var = np.mean((Y_val - preds_val.mean(axis=0)) ** 2, axis=0)  # (T,)
    return models, noise_var


def ensemble_predict(models, noise_var, X):
    preds     = np.stack([m.predict(X) for m in models], axis=0)     # (K, n, T)
    mu        = preds.mean(axis=0)
    epistemic = preds.var(axis=0)
    var       = epistemic + noise_var[np.newaxis, :]
    return mu, var


print(f"\nFitting deep ensemble (K={K_ENSEMBLE}) on full training set...")
ensemble_models, noise_var = fit_ensemble(splits["X_train"], splits["Y_train"],
                                          splits["X_val"], splits["Y_val"], best_params)
print("Done.")

# %%
mu_test_s, var_test_s = ensemble_predict(ensemble_models, noise_var, splits["X_test"])
mu_test, var_test = unscale_predictions(mu_test_s, var_test_s, splits["scaler_Y"])

nn_results = {}
for t in range(T):
    nn_results[f"Y{t+1} RMSE"] = rmse(splits["Y_test"][:, t], mu_test[:, t])
    nn_results[f"Y{t+1} NLPD"] = nlpd(splits["Y_test"][:, t], mu_test[:, t], var_test[:, t])

print("\nDeep Ensemble MLP results:")
for k, v in nn_results.items():
    print(f"  {k}: {v:.4f}")

# %% [markdown]
# ## Full-Data Comparison Against All GP Models

# %%
try:
    with open(os.path.join(OUT_DIR, "all_model_results.json")) as f:
        gp_results = json.load(f)
except FileNotFoundError:
    print("WARNING: run 02 and 03 first to generate all_model_results.json")
    gp_results = {}

all_results = dict(gp_results)
all_results["Deep Ensemble MLP"] = nn_results

print("\n" + summary_table(all_results))

with open(os.path.join(OUT_DIR, "all_model_results_with_nn.json"), "w") as f:
    json.dump(all_results, f, indent=2)

# %% [markdown]
# ## Parity Plots — NN Predictions

# %%
fig, axes = plt.subplots(1, T, figsize=(14, 4))
nn_colors = ["#9B59B6", "#D62728", "#1ABC9C", "#F39C12"]

for i, (ax, col, oname) in enumerate(zip(axes, nn_colors, OUTPUT_NAMES)):
    y_true = splits["Y_test"][:, i]
    y_pred = mu_test[:, i]
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, color=col, edgecolors="none")
    lims = [min(y_true.min(), y_pred.min()) - 0.3,
            max(y_true.max(), y_pred.max()) + 0.3]
    ax.plot(lims, lims, "k--", lw=1.2, label="Perfect")
    ax.set_xlim(lims); ax.set_ylim(lims)
    r2 = r2_score(y_true, y_pred)
    ax.set_xlabel(f"True {SHORT_OUTPUT_NAMES[i]}", fontsize=9)
    ax.set_ylabel(f"Pred {SHORT_OUTPUT_NAMES[i]}", fontsize=9)
    ax.set_title(f"Y{i+1}: RMSE={nn_results[f'Y{i+1} RMSE']:.3f}  R²={r2:.3f}",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, frameon=False)

fig.suptitle("Deep Ensemble MLP: Parity Plots", fontsize=12, fontweight="bold")
plt.tight_layout()
savefig("06a_nn_parity.png")

# %% [markdown]
# ## Side-by-Side Model Comparison Bar Chart

# %%
model_order  = ["Independent GP", "ICM (Q=1)", "LCM (Q=2)", "Deep Ensemble MLP"]
model_colors = {"Independent GP": "#888888", "ICM (Q=1)": "#E87722",
                "LCM (Q=2)": "#4878CF", "Deep Ensemble MLP": "#D62728"}

fig, axes = plt.subplots(2, T, figsize=(16, 9))

for row, metric in enumerate(["RMSE", "NLPD"]):
    for col in range(T):
        ax = axes[row, col]
        vals  = [all_results.get(m, {}).get(f"Y{col+1} {metric}", float("nan"))
                 for m in model_order]
        bars  = ax.bar(range(len(model_order)), vals,
                       color=[model_colors[m] for m in model_order],
                       edgecolor="white", width=0.6)
        ax.set_xticks(range(len(model_order)))
        ax.set_xticklabels([m.replace(" ", "\n") for m in model_order], fontsize=7)
        ax.set_ylabel(metric)
        ax.set_title(f"Y{col+1}: {SHORT_OUTPUT_NAMES[col]} — {metric}",
                     fontweight="bold", fontsize=9)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + 0.005 * max(v for v in vals if not np.isnan(v)),
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)

fig.suptitle(
    "Model Comparison: GP Variants vs Deep Ensemble MLP\n(Full training data)",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
savefig("06b_model_comparison_bar.png")

# %% [markdown]
# ## Low-Data Regime: NN vs GP Models

# %%
TRAIN_SIZES = [20, 40, 80, 160]
N_SEEDS     = 5

MODEL_CONFIGS = {
    "Independent GP": lambda: IndependentGP(ARD=True,  n_restarts=2),
    "ICM (Q=1)":      lambda: ICM(W_rank=1,  ARD=True,  n_restarts=2),
    "LCM (Q=2)":      lambda: LCM(num_latents=2, W_rank=1, ARD=True, n_restarts=2),
}

records = defaultdict(lambda: defaultdict(dict))

print("\nRunning low-data ablation (GP models)...")
for n in TRAIN_SIZES:
    print(f"\n--- n={n} ---")
    for model_name, model_factory in MODEL_CONFIGS.items():
        seed_metrics = defaultdict(list)
        for seed in range(N_SEEDS):
            sub = subsample_train(splits, n, random_state=seed)
            try:
                m = model_factory()
                m.fit(sub["X_train"], sub["Y_train"])
                mu_s, var_s = m.predict(splits["X_test"])
                mu, var = unscale_predictions(mu_s, var_s, splits["scaler_Y"])
                for t in range(T):
                    seed_metrics[f"Y{t+1} RMSE"].append(rmse(splits["Y_test"][:, t], mu[:, t]))
                    seed_metrics[f"Y{t+1} NLPD"].append(nlpd(splits["Y_test"][:, t], mu[:, t], var[:, t]))
            except Exception as e:
                print(f"    [{model_name}, seed={seed}] ERROR: {e}")
        if seed_metrics:
            for key, vals in seed_metrics.items():
                records[n][model_name][f"{key} mean"] = np.mean(vals)
                records[n][model_name][f"{key} std"]  = np.std(vals)
            print(f"  {model_name:<20}  "
                  + "  ".join(f"Y{t+1}={records[n][model_name].get(f'Y{t+1} RMSE mean', float('nan')):.3f}"
                               for t in range(T)))

# %%
print("\nRunning low-data ablation (Deep Ensemble MLP)...")
for n in TRAIN_SIZES:
    print(f"  n={n}", end="  ", flush=True)
    seed_metrics = defaultdict(list)
    for seed in range(N_SEEDS):
        sub = subsample_train(splits, n, random_state=seed)
        K = K_ENSEMBLE
        models_s = [
            MLPRegressor(**best_params, activation="relu", max_iter=1000,
                         random_state=seed * 100 + k, learning_rate_init=1e-3).fit(
                sub["X_train"], sub["Y_train"])
            for k in range(K)
        ]
        preds_val_s = np.stack([m.predict(splits["X_val"]) for m in models_s], axis=0)
        nv = np.mean((splits["Y_val"] - preds_val_s.mean(axis=0)) ** 2, axis=0)
        mu_s, var_s = ensemble_predict(models_s, nv, splits["X_test"])
        mu, var = unscale_predictions(mu_s, var_s, splits["scaler_Y"])
        for t in range(T):
            seed_metrics[f"Y{t+1} RMSE"].append(rmse(splits["Y_test"][:, t], mu[:, t]))
            seed_metrics[f"Y{t+1} NLPD"].append(nlpd(splits["Y_test"][:, t], mu[:, t], var[:, t]))

    for key, vals in seed_metrics.items():
        records[n]["Deep Ensemble MLP"][f"{key} mean"] = np.mean(vals)
        records[n]["Deep Ensemble MLP"][f"{key} std"]  = np.std(vals)
    print("  ".join(f"Y{t+1}={records[n]['Deep Ensemble MLP'].get(f'Y{t+1} RMSE mean', float('nan')):.3f}"
                    for t in range(T)))

# %% [markdown]
# ## RMSE vs Training Size (all models, all outputs)

# %%
model_styles = {
    "Independent GP":    dict(color="#888888", marker="o", linestyle="--"),
    "ICM (Q=1)":         dict(color="#E87722", marker="s", linestyle="-"),
    "LCM (Q=2)":         dict(color="#4878CF", marker="^", linestyle="-"),
    "Deep Ensemble MLP": dict(color="#D62728", marker="D", linestyle=":"),
}

for metric_key, metric_label, fig_name in [
    ("RMSE", "RMSE", "06c_nn_low_data_rmse.png"),
    ("NLPD", "NLPD (nats, lower = better)", "06d_nn_low_data_nlpd.png"),
]:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes_flat = axes.flatten()

    for t in range(T):
        ax = axes_flat[t]
        for model_name, style in model_styles.items():
            ns, means, stds = [], [], []
            for n in TRAIN_SIZES:
                key_mean = f"Y{t+1} {metric_key} mean"
                if model_name in records[n] and key_mean in records[n][model_name]:
                    ns.append(n)
                    means.append(records[n][model_name][key_mean])
                    stds.append(records[n][model_name][f"Y{t+1} {metric_key} std"])
            if not ns:
                continue
            ns, means, stds = np.array(ns), np.array(means), np.array(stds)
            ax.plot(ns, means, label=model_name, **style, lw=2, ms=7)
            ax.fill_between(ns, means - stds, means + stds,
                            alpha=0.12, color=style["color"])

        ax.set_xscale("log")
        ax.set_xlabel("Training set size  n  (log scale)")
        ax.set_ylabel(metric_label)
        ax.set_title(f"Y{t+1}: {OUTPUT_NAMES[t]}", fontweight="bold", fontsize=9)
        ax.legend(frameon=False, fontsize=8)
        ax.set_xticks(TRAIN_SIZES); ax.set_xticklabels(TRAIN_SIZES, fontsize=8)

    fig.suptitle(
        f"Low-Data Regime: GP Variants vs Deep Ensemble MLP — {metric_key}\n"
        "(shaded band = ±1 std over 5 random seeds)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    savefig(fig_name)

# %%
print("\n=== Final Summary ===")
print(summary_table(all_results))
print("\nKey takeaway:")
print("  GP models provide principled uncertainty (calibrated NLPD) with far fewer")
print("  parameters.  MOGPs (ICM/LCM) additionally exploit inter-pollutant correlation,")
print("  which is especially valuable when reference-analyser evaluations are scarce.")
