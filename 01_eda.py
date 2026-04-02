# %% [markdown]
# # 01 — Exploratory Data Analysis: UCI Air Quality Dataset
#
# The dataset is hourly air quality readings from an Italian city (Mar 2004 –
# Feb 2005), with 4 reference pollutant measurements (CO, Benzene, NOx, NO2)
# and 5 metal-oxide sensor readings + meteorological variables as features.
# Missing values are encoded as −200 sentinel values.

# %%
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from data_utils import (
    load_raw, get_Xy, missing_report,
    FEATURE_NAMES, OUTPUT_NAMES, SHORT_OUTPUT_NAMES, SHORT_FEATURE_NAMES,
    GP_SUBSAMPLE,
)

sns.set_theme(style="whitegrid", font_scale=1.1)
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def savefig(name):
    plt.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches="tight")
    plt.show()

# %%
print("Loading raw data...")
df_raw = load_raw()
print(f"  Raw shape : {df_raw.shape}")
print(f"  Date range: {df_raw['datetime'].min()} → {df_raw['datetime'].max()}")

# %% [markdown]
# ## 1 — Missing Data Analysis
#
# The −200 sentinel has already been replaced with NaN by `load_raw()`.
# Here we quantify how much data is missing per column before any imputation.

# %%
miss = missing_report(df_raw)
print("\nMissing-data report (columns with any missingness):")
print(miss[miss["n_missing"] > 0].to_string())

# Bar chart of missingness
fig, ax = plt.subplots(figsize=(12, 4.5))
cols_with_miss = miss[miss["pct_missing"] > 0].index.tolist()
pcts = miss.loc[cols_with_miss, "pct_missing"]

colors_miss = ["#D62728" if p > 50 else "#E87722" if p > 10 else "#4878CF"
               for p in pcts]
ax.bar(range(len(cols_with_miss)), pcts, color=colors_miss, edgecolor="white")
ax.set_xticks(range(len(cols_with_miss)))
ax.set_xticklabels(cols_with_miss, rotation=40, ha="right", fontsize=9)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylabel("% Missing")
ax.set_title("Missingness per Column (−200 sentinel → NaN)", fontweight="bold")

from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color="#D62728", label=">50% missing (dropped)"),
    Patch(color="#E87722", label="10–50% missing"),
    Patch(color="#4878CF", label="<10% missing"),
], frameon=False, fontsize=9)

plt.tight_layout()
savefig("01a_missing_data.png")

# %% [markdown]
# ## 2 — Output Distributions (after cleaning)

# %%
X, Y = get_Xy(df_raw)
print(f"\nCleaned dataset: {len(X)} rows × {X.shape[1]} features, {Y.shape[1]} outputs")
print(f"GP experiments will use a subsample of {GP_SUBSAMPLE} training rows.")

output_colors = ["#4878CF", "#E87722", "#6ACC65", "#D62728"]

fig, axes = plt.subplots(1, 4, figsize=(15, 4))
for i, (ax, col, name) in enumerate(zip(axes, output_colors, OUTPUT_NAMES)):
    ax.hist(Y[:, i], bins=50, color=col, edgecolor="white", alpha=0.85)
    ax.axvline(np.median(Y[:, i]), color="k", lw=1.5, linestyle="--",
               label=f"Median={np.median(Y[:,i]):.1f}")
    ax.set_xlabel(name, fontsize=9)
    ax.set_ylabel("Count" if i == 0 else "")
    ax.set_title(f"Y{i+1}: {SHORT_OUTPUT_NAMES[i]}", fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

fig.suptitle("Output Distributions (UCI Air Quality — after cleaning)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
savefig("01b_output_distributions.png")

# %% [markdown]
# ## 3 — Output Correlation Matrix

# %%
Y_df = pd.DataFrame(Y, columns=SHORT_OUTPUT_NAMES)
corr = Y_df.corr()

print("\nOutput correlation matrix:")
print(corr.round(3).to_string())

fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(
    corr, annot=True, fmt=".3f", cmap="RdBu_r",
    vmin=-1, vmax=1, center=0, square=True,
    linewidths=0.5, ax=ax,
    annot_kws={"size": 11},
)
ax.set_title("Pairwise Output Correlation", fontweight="bold")
plt.tight_layout()
savefig("01c_output_correlation.png")

# %% [markdown]
# ## 4 — Temporal Patterns

# %%
output_raw_names = ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
df_clean = load_raw().dropna(subset=output_raw_names).reset_index(drop=True)
df_clean["Hour"]  = df_clean["datetime"].dt.hour
df_clean["Month"] = df_clean["datetime"].dt.month

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i, (col, name, color) in enumerate(zip(output_raw_names, OUTPUT_NAMES, output_colors)):
    ax = axes[0, i]
    hourly = df_clean.groupby("Hour")[col].median()
    ax.plot(hourly.index, hourly.values, color=color, lw=2.5, marker="o", ms=4)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(name if i == 0 else "")
    ax.set_title(f"{SHORT_OUTPUT_NAMES[i]} — Hourly Median", fontweight="bold", fontsize=9)
    ax.set_xticks([0, 6, 12, 18, 23])

    ax = axes[1, i]
    monthly = df_clean.groupby("Month")[col].median()
    ax.bar(monthly.index, monthly.values, color=color, edgecolor="white", alpha=0.85)
    ax.set_xlabel("Month")
    ax.set_ylabel(name if i == 0 else "")
    ax.set_title(f"{SHORT_OUTPUT_NAMES[i]} — Monthly Median", fontweight="bold", fontsize=9)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"], fontsize=8)

fig.suptitle("Temporal Structure — Hourly and Monthly Median Profiles",
             fontsize=12, fontweight="bold")
plt.tight_layout()
savefig("01d_temporal_patterns.png")

# %% [markdown]
# ## 5 — Feature–Output Correlation Heatmap

# %%
X_df  = pd.DataFrame(X, columns=SHORT_FEATURE_NAMES)
Y_df2 = pd.DataFrame(Y, columns=SHORT_OUTPUT_NAMES)
feat_out_corr = pd.concat([X_df, Y_df2], axis=1).corr().loc[
    SHORT_FEATURE_NAMES, SHORT_OUTPUT_NAMES
]

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    feat_out_corr, annot=True, fmt=".2f", cmap="RdBu_r",
    vmin=-1, vmax=1, center=0,
    linewidths=0.4, ax=ax, annot_kws={"size": 9},
)
ax.set_title("Feature–Output Correlation  (Pearson r)", fontweight="bold")
ax.set_xlabel("Output")
ax.set_ylabel("Feature")
plt.tight_layout()
savefig("01e_feature_output_correlation.png")

# %% [markdown]
# ## 6 — Pairplot of Outputs (sample n=1000)

# %%
np.random.seed(42)
sample_idx = np.random.choice(len(Y), size=min(1000, len(Y)), replace=False)
pair_df = pd.DataFrame(Y[sample_idx], columns=SHORT_OUTPUT_NAMES)

g = sns.PairGrid(pair_df, diag_sharey=False)
g.map_upper(sns.scatterplot, s=8, alpha=0.35, color="#4878CF", edgecolors="none")
g.map_lower(sns.kdeplot, fill=True, cmap="Blues", thresh=0.05)
g.map_diag(sns.histplot, bins=30, color="#4878CF", edgecolor="white")
g.figure.suptitle("Output Pairplot (sample n=1 000)", y=1.01, fontweight="bold")
plt.tight_layout()
savefig("01f_output_pairplot.png")

# %%
print("\n=== Summary Statistics ===")
print(pd.DataFrame(Y, columns=SHORT_OUTPUT_NAMES).describe().round(2).to_string())
print(f"\nFeature matrix : {X.shape}")
print(f"Target  matrix : {Y.shape}")
print(f"\n{len(Y)} usable rows after removing rows with missing targets.")
print(f"GP experiments subsample to {GP_SUBSAMPLE} training points for tractable inference.")
