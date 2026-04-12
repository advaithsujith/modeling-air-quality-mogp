# %% [markdown]
# # 05 — Multi-Objective Active Learning with MOGP Surrogate
#
# Pool-based active learning to discover the CO–NO2 Pareto front using as
# few labelled evaluations as possible. We compare three acquisition strategies:
# random search, independent GP with Thompson sampling, and LCM with Thompson
# sampling (TSEMO-style). The goal is to find worst-case co-pollution events.

# %%
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler

from data_utils import get_Xy, OUTPUT_NAMES, SHORT_OUTPUT_NAMES
from gp_models import IndependentGP, LCM
from evaluation import pareto_mask, pareto_front, hypervolume_2d, default_reference_point

sns.set_theme(style="whitegrid", font_scale=1.1)
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(0)

def savefig(name):
    plt.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## Setup: Pool-Based Active Learning

# %%
# Use the full cleaned dataset as pool (no GP_SUBSAMPLE cap here)
print("Loading full dataset as candidate pool...")
X_all, Y_all_4 = get_Xy()

# BO objectives: CO(GT) = col 0, NO2(GT) = col 3
BO_OUTPUT_INDICES = [0, 3]
BO_OUTPUT_NAMES   = [SHORT_OUTPUT_NAMES[i] for i in BO_OUTPUT_INDICES]
Y_all = Y_all_4[:, BO_OUTPUT_INDICES]   # (n, 2)

scaler = StandardScaler()
X_all_s = scaler.fit_transform(X_all)

# Negate for maximisation → minimisation framework
Y_neg = -Y_all

ref_point = default_reference_point(Y_neg, slack=0.1)
true_pf   = pareto_front(Y_neg)
true_hv   = hypervolume_2d(true_pf, ref_point)

print(f"  Pool size  : {len(X_all)}")
print(f"  CO  range  : [{Y_all[:,0].min():.2f}, {Y_all[:,0].max():.2f}]")
print(f"  NO2 range  : [{Y_all[:,1].min():.2f}, {Y_all[:,1].max():.2f}]")
print(f"  True Pareto front : {len(true_pf)} points")
print(f"  True hypervolume  : {true_hv:.3f}")

# %% [markdown]
# ## BO Loop

# %%
N_INIT   = 30
N_ITER   = 40
N_REPEAT = 5


def thompson_sample(model, X_candidates, rng):
    """
    Marginal-output posterior sample: mu + std * z per output.

    Note: this samples each output independently (marginal approximation).
    True TSEMO-style sampling would draw from the joint T-output Gaussian at
    each candidate, exploiting posterior cross-output correlations. The marginal
    approximation is standard practice when full posterior covariance extraction
    is computationally prohibitive.
    """
    mu, var = model.predict(X_candidates)
    std = np.sqrt(np.clip(var, 1e-9, None))
    return mu + std * rng.randn(*mu.shape)


def select_next(sample_Y, rng):
    """Pick a uniformly random point from the sample Pareto front."""
    mask = pareto_mask(sample_Y)
    pf_idx = np.where(mask)[0]
    return pf_idx[rng.randint(len(pf_idx))] if len(pf_idx) > 0 else rng.randint(len(sample_Y))


def run_bo(strategy, seed):
    rng = np.random.RandomState(seed)
    pool_idx  = np.arange(len(X_all))
    init_idx  = rng.choice(pool_idx, size=N_INIT, replace=False)
    remaining = np.setdiff1d(pool_idx, init_idx)

    obs_X = X_all_s[init_idx].copy()
    obs_Y = Y_neg[init_idx].copy()   # minimisation frame

    def current_hv():
        return hypervolume_2d(pareto_front(obs_Y), ref_point)

    hv_trace = [current_hv()]

    for it in range(N_ITER):
        X_cand = X_all_s[remaining]

        if strategy == "random":
            chosen_local = rng.randint(len(remaining))

        elif strategy == "independent_gp":
            m = IndependentGP(ARD=True, n_restarts=1)
            m.fit(obs_X, obs_Y)
            sample = thompson_sample(m, X_cand, rng)
            chosen_local = select_next(sample, rng)

        elif strategy == "lcm":
            m = LCM(num_latents=2, W_rank=1, ARD=True, n_restarts=1)
            m.fit(obs_X, obs_Y)
            sample = thompson_sample(m, X_cand, rng)
            chosen_local = select_next(sample, rng)

        chosen_global = remaining[chosen_local]
        obs_X  = np.vstack([obs_X,  X_all_s[chosen_global]])
        obs_Y  = np.vstack([obs_Y,  Y_neg[chosen_global]])
        remaining = np.delete(remaining, chosen_local)

        hv_trace.append(current_hv())

        if (it + 1) % 10 == 0:
            print(f"    iter {it+1:2d}/{N_ITER}  HV={hv_trace[-1]:.4f}  ({strategy}, seed={seed})")

    return hv_trace, obs_Y


results_bo = {s: [] for s in ["random", "independent_gp", "lcm"]}
for strategy in results_bo:
    print(f"\n=== Strategy: {strategy} ===")
    for seed in range(N_REPEAT):
        print(f"  Run {seed+1}/{N_REPEAT}")
        hv_trace, _ = run_bo(strategy, seed=seed * 100)
        results_bo[strategy].append(hv_trace)

# %% [markdown]
# ## Hypervolume vs Evaluations

# %%
strategy_styles = {
    "random":         dict(color="#888888", label="Random Search",       linestyle="--"),
    "independent_gp": dict(color="#E87722", label="Independent GP + TS", linestyle="-"),
    "lcm":            dict(color="#4878CF", label="LCM (Q=2) + TS",      linestyle="-"),
}

iters = np.arange(N_ITER + 1)

fig, ax = plt.subplots(figsize=(9, 5))
for strat, style in strategy_styles.items():
    traces   = np.array(results_bo[strat])
    mean_hv  = traces.mean(axis=0)
    std_hv   = traces.std(axis=0)
    ax.plot(iters, mean_hv, lw=2.5, **style)
    ax.fill_between(iters, mean_hv - std_hv, mean_hv + std_hv,
                    alpha=0.15, color=style["color"])

ax.axhline(true_hv, color="crimson", lw=1.5, linestyle=":",
           label=f"True HV = {true_hv:.3f}")
ax.set_xlabel(f"Number of Reference Evaluations (after N₀={N_INIT})")
ax.set_ylabel("Hypervolume of Discovered CO–NO2 Pareto Front")
ax.set_title(
    "Multi-Objective Active Learning: Hypervolume vs Evaluation Budget\n"
    f"(mean ± 1 std over {N_REPEAT} runs;  objectives: CO and NO2 concentration)",
    fontweight="bold"
)
ax.legend(frameon=False, fontsize=9)
plt.tight_layout()
savefig("05a_hypervolume_trace.png")

# %% [markdown]
# ## Normalised Hypervolume Gap

# %%
fig, ax = plt.subplots(figsize=(9, 5))
for strat, style in strategy_styles.items():
    traces = np.array(results_bo[strat])
    hv0    = traces[:, 0:1]
    normed = (traces - hv0) / (true_hv - hv0 + 1e-10)
    ax.plot(iters, normed.mean(axis=0), lw=2.5, **style)
    ax.fill_between(iters,
                    normed.mean(axis=0) - normed.std(axis=0),
                    normed.mean(axis=0) + normed.std(axis=0),
                    alpha=0.15, color=style["color"])

ax.axhline(1.0, color="crimson", lw=1.5, linestyle=":",
           label="True Pareto Front (gap = 0)")
ax.set_xlabel(f"Number of Reference Evaluations (after N₀={N_INIT})")
ax.set_ylabel("Normalised Hypervolume  (1.0 = optimal)")
ax.set_title("Normalised HV Gap Closure by Strategy", fontweight="bold")
ax.legend(frameon=False, fontsize=9)
plt.tight_layout()
savefig("05b_normalised_hv.png")

# %% [markdown]
# ## Discovered Pareto Fronts (CO vs NO2)

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

for ax, strat in zip(axes, ["random", "independent_gp", "lcm"]):
    _, obs_Y_neg = run_bo(strat, seed=42)
    obs_Y_orig   = -obs_Y_neg   # back to original scale
    true_Y_orig  = Y_all

    disc_pf_neg  = pareto_front(obs_Y_neg)
    disc_pf_orig = -disc_pf_neg
    true_pf_orig = -pareto_front(Y_neg)

    # All pool points (background)
    ax.scatter(true_Y_orig[:, 0], true_Y_orig[:, 1],
               s=6, alpha=0.15, color="#cccccc", edgecolors="none", label="Full pool")
    # Discovered Pareto front
    ax.scatter(disc_pf_orig[:, 0], disc_pf_orig[:, 1],
               s=40, color=strategy_styles[strat]["color"],
               zorder=4, label=f"Discovered PF ({len(disc_pf_orig)} pts)")
    # True Pareto front
    tpf_s = true_pf_orig[np.argsort(true_pf_orig[:, 0])]
    ax.step(
        np.append(tpf_s[:, 0], tpf_s[-1, 0]),
        np.append(tpf_s[0, 1], tpf_s[:, 1]),
        where="post", color="crimson", lw=1.8, linestyle="--", label="True Pareto front"
    )
    ax.set_xlabel(f"CO(GT) [mg/m³]")
    if ax == axes[0]:
        ax.set_ylabel(f"NO2(GT) [µg/m³]")
    hv_disc = hypervolume_2d(pareto_front(obs_Y_neg), ref_point)
    ax.set_title(
        f"{strategy_styles[strat]['label']}\nHV={hv_disc:.3f}  (true={true_hv:.3f})",
        fontweight="bold", fontsize=10
    )
    ax.legend(fontsize=7, frameon=False)

fig.suptitle(
    f"Discovered CO–NO2 Pareto Fronts after {N_INIT + N_ITER} Total Evaluations\n"
    "(upper-right = high CO and high NO2 — worst-case pollution events)",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
savefig("05c_pareto_fronts.png")

# %% [markdown]
# ## True Pareto Front Visualisation

# %%
true_pf_orig   = -pareto_front(Y_neg)
pf_mask_orig   = pareto_mask(Y_neg)
true_pf_sorted = true_pf_orig[np.argsort(true_pf_orig[:, 0])]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.scatter(Y_all[~pf_mask_orig, 0], Y_all[~pf_mask_orig, 1],
           s=8, alpha=0.2, color="#cccccc", edgecolors="none", label="Dominated")
ax.scatter(Y_all[pf_mask_orig, 0], Y_all[pf_mask_orig, 1],
           s=40, color="crimson", zorder=5,
           label=f"Pareto front ({pf_mask_orig.sum()} pts)")
ax.step(
    np.append(true_pf_sorted[:, 0], true_pf_sorted[-1, 0]),
    np.append(true_pf_sorted[0, 1], true_pf_sorted[:, 1]),
    where="post", color="crimson", lw=2
)
ax.set_xlabel("CO(GT) [mg/m³]")
ax.set_ylabel("NO2(GT) [µg/m³]")
ax.set_title("True Pareto Front of CO vs NO2\n(upper-right = worst-case co-pollution)",
             fontweight="bold")
ax.legend(fontsize=9, frameon=False)

ax = axes[1]
ax.plot(true_pf_sorted[:, 0], true_pf_sorted[:, 1],
        "o-", color="crimson", ms=5, lw=2)
ax.set_xlabel("CO(GT) [mg/m³]")
ax.set_ylabel("NO2(GT) [µg/m³]")
ax.set_title("Pareto Trade-off Curve\n(upper-right corner = simultaneously dangerous)",
             fontweight="bold")

plt.tight_layout()
savefig("05d_true_pareto_front.png")

# %%
print("\n=== BO Results Summary ===")
print(f"{'Strategy':<25} {'Final HV (mean)':>16} {'Final HV (std)':>15} {'% of True HV':>14}")
print("-" * 72)
for strat, style in strategy_styles.items():
    traces    = np.array(results_bo[strat])
    final_hvs = traces[:, -1]
    print(f"{style['label']:<25}{final_hvs.mean():>16.4f}{final_hvs.std():>15.4f}"
          f"{100 * final_hvs.mean() / true_hv:>13.1f}%")
print(f"\n  True hypervolume (full pool): {true_hv:.4f}")
print(f"  Evaluation budget: {N_INIT} initial + {N_ITER} BO iterations = {N_INIT + N_ITER} total")
