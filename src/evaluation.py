"""
Evaluation metrics for GP regression and multi-objective optimisation.

Metrics
-------
rmse          : Root Mean Squared Error per output
nlpd          : Negative Log Predictive Density (lower = better calibrated)
nlml          : Negative Log Marginal Likelihood (used for model comparison)
pareto_mask   : Boolean mask of Pareto-efficient points (minimisation)
hypervolume_2d: Hypervolume indicator for 2-objective minimisation
"""

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Pointwise regression metrics
# ---------------------------------------------------------------------------

def rmse(y_true, y_pred):
    """Root Mean Squared Error. Arrays of shape (n,) or (n,1)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def nlpd(y_true, mu_pred, var_pred):
    """
    Negative Log Predictive Density under a Gaussian predictive distribution.

    Parameters
    ----------
    y_true   : (n,)  ground-truth targets
    mu_pred  : (n,)  predictive mean
    var_pred : (n,)  predictive variance (NOT std)

    Returns
    -------
    scalar NLPD (nats); lower is better.
    """
    y_true = np.asarray(y_true).ravel()
    mu_pred = np.asarray(mu_pred).ravel()
    var_pred = np.asarray(var_pred).ravel().clip(1e-9)
    log_p = norm.logpdf(y_true, loc=mu_pred, scale=np.sqrt(var_pred))
    return -np.mean(log_p)


def summary_table(results: dict) -> str:
    """
    Format a results dict as a plain-text table.

    results = {
        "Model Name": {"Y1 RMSE": ..., "Y2 RMSE": ..., "Y1 NLPD": ..., ...},
        ...
    }

    Columns are inferred automatically from the first model's keys, grouped so
    all RMSE columns appear before all NLPD columns (both sorted alphabetically).
    Works for any number of outputs T.
    """
    if not results:
        return "(no results)"
    first = next(iter(results.values()))
    rmse_keys = sorted(k for k in first if "RMSE" in k)
    nlpd_keys = sorted(k for k in first if "NLPD" in k)
    metric_keys = rmse_keys + nlpd_keys

    col_w = 11
    header = f"{'Model':<30}" + "".join(f" {k:>{col_w}}" for k in metric_keys)
    lines = [header, "-" * len(header)]
    for name, m in results.items():
        row = f"{name:<30}" + "".join(
            f" {m.get(k, float('nan')):>{col_w}.3f}" for k in metric_keys
        )
        lines.append(row)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pareto utilities  (minimisation)
# ---------------------------------------------------------------------------

def pareto_mask(costs: np.ndarray) -> np.ndarray:
    """
    Boolean mask of Pareto-efficient rows under minimisation.
    A point is efficient if no other point is <= on all objectives and < on at least one.
    """
    costs = np.asarray(costs)
    n = costs.shape[0]
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        # Mark all points dominated BY i (i is at least as good and strictly better on one)
        dominated = (
            np.all(costs[i] <= costs, axis=1)
            & np.any(costs[i] < costs, axis=1)
        )
        dominated[i] = False  # do not remove i itself
        is_efficient[dominated] = False
    return is_efficient


def pareto_front(costs: np.ndarray):
    """Return the (k, m) array of Pareto-efficient points, sorted by costs[:,0]."""
    mask = pareto_mask(costs)
    front = costs[mask]
    return front[np.argsort(front[:, 0])]


# ---------------------------------------------------------------------------
# Hypervolume indicator  (2-objective minimisation)
# ---------------------------------------------------------------------------

def hypervolume_2d(pareto_pts: np.ndarray, ref_point: np.ndarray) -> float:
    """
    Exact hypervolume indicator for 2-objective minimisation.
    ref_point should be worse than (dominated by) all points in pareto_pts.
    """
    pts = np.asarray(pareto_pts, dtype=float)
    ref = np.asarray(ref_point, dtype=float)

    # Keep only dominated Pareto points (safety check)
    valid = np.all(pts < ref, axis=1)
    pts = pts[valid]
    if len(pts) == 0:
        return 0.0

    # Sort by f1 ascending  →  f2 must be descending for a true Pareto front
    pts = pts[np.argsort(pts[:, 0])]

    hv = 0.0
    prev_y = ref[1]
    for f1, f2 in pts:
        hv += (ref[0] - f1) * (prev_y - f2)
        prev_y = f2
    return hv


def default_reference_point(Y: np.ndarray, slack: float = 0.1) -> np.ndarray:
    """
    Compute a reference point strictly worse than the worst observed value
    on each objective. Works correctly for both positive and negative Y values.
    """
    worst = Y.max(axis=0)
    return worst + np.maximum(np.abs(worst), 1.0) * slack
