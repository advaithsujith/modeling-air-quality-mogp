"""
Data utilities for the UCI Air Quality dataset (UCI ML Repository #360).

~9358 hourly readings from an Italian city (Mar 2004 – Feb 2005).
Features: 5 metal-oxide sensors + 3 meteorological variables + cyclic time.
Outputs: CO, Benzene (C6H6), NOx, NO2 reference analyser measurements.
Missing values are encoded as -200 in the original CSV.

GP_SUBSAMPLE caps training at 500 rows to keep inference tractable.
"""

import io
import os
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths and download URLs
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(__file__)
DATA_DIR  = os.path.join(_HERE, "..", "data")
ZIP_PATH  = os.path.join(DATA_DIR, "AirQualityUCI.zip")
CSV_PATH  = os.path.join(DATA_DIR, "AirQualityUCI.csv")

# UCI changed its URL scheme; try both
DATA_URLS = [
    "https://archive.ics.uci.edu/static/public/360/air+quality.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00244/AirQualityUCI.zip",
]

# ---------------------------------------------------------------------------
# Column / feature names
# ---------------------------------------------------------------------------

# Raw sensor inputs (5 metal-oxide sensors + 3 meteorological)
_SENSOR_COLS = [
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
]
_MET_COLS = ["T", "RH", "AH"]

# Cyclic time features derived from the timestamp
_TIME_COLS = ["HourSin", "HourCos"]

FEATURE_NAMES = _SENSOR_COLS + _MET_COLS + _TIME_COLS   # length 10
SHORT_FEATURE_NAMES = [
    "S1(CO)", "S2(NMHC)", "S3(NOx)", "S4(NO2)", "S5(O3)",
    "T", "RH", "AH", "sin(h)", "cos(h)",
]

OUTPUT_NAMES = [
    "CO(GT) [mg/m³]",
    "C6H6(GT) [µg/m³]",
    "NOx(GT) [ppb]",
    "NO2(GT) [µg/m³]",
]
SHORT_OUTPUT_NAMES = ["CO", "C6H6", "NOx", "NO2"]

# Subsampling cap for GP training (kept low due to O(n³) cost with T=4 outputs)
GP_SUBSAMPLE = 500


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_data(force: bool = False) -> None:
    """Download and extract AirQualityUCI.csv from UCI if not already present."""
    import requests

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(CSV_PATH) and not force:
        print(f"Data already at {CSV_PATH}")
        return

    last_err = None
    for url in DATA_URLS:
        try:
            print(f"Downloading from {url} ...")
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with open(ZIP_PATH, "wb") as f:
                f.write(r.content)
            print("Download complete.")
            break
        except Exception as e:
            last_err = e
            print(f"  Failed ({e}), trying next URL...")
    else:
        raise RuntimeError(
            f"Could not download the dataset. Last error: {last_err}\n"
            "Please download AirQualityUCI.zip manually from:\n"
            "  https://archive.ics.uci.edu/ml/machine-learning-databases/00244/AirQualityUCI.zip\n"
            f"and place it at: {ZIP_PATH}"
        )

    # Extract CSV from zip
    with zipfile.ZipFile(ZIP_PATH) as z:
        names = z.namelist()
        csv_members = [n for n in names if n.endswith(".csv")]
        if not csv_members:
            raise RuntimeError(f"No CSV found in zip. Contents: {names}")
        member = csv_members[0]
        print(f"Extracting {member} ...")
        with z.open(member) as src, open(CSV_PATH, "wb") as dst:
            dst.write(src.read())
    print(f"Saved to {CSV_PATH}")


# ---------------------------------------------------------------------------
# Loading & cleaning
# ---------------------------------------------------------------------------

def load_raw() -> pd.DataFrame:
    """
    Load the raw CSV, replace -200 sentinels with NaN, and add cyclic hour
    features. NMHC(GT) is kept here but dropped in get_Xy().
    """
    if not os.path.exists(CSV_PATH):
        download_data()

    # Semicolon-separated, comma decimal points; trailing semicolons create phantom columns.
    df = pd.read_csv(CSV_PATH, sep=";", decimal=",", low_memory=False)
    df = df.dropna(axis=1, how="all")   # remove all-NaN phantom columns
    df = df.dropna(axis=0, how="all")   # remove all-NaN trailing rows

    # Parse datetime (format: "10/03/2004", "18.00.00")
    try:
        df["datetime"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            format="%d/%m/%Y %H.%M.%S",
        )
    except Exception:
        df["datetime"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            dayfirst=True,
            errors="coerce",
        )

    # Replace -200 sentinel with NaN across all numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df.loc[df[col] < -100, col] = np.nan

    # Cyclic hour encoding (avoids the 23→0 discontinuity)
    hour = df["datetime"].dt.hour
    df["HourSin"] = np.sin(2 * np.pi * hour / 24)
    df["HourCos"] = np.cos(2 * np.pi * hour / 24)

    return df


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame summarising missingness per column.

    Columns: n_missing, pct_missing (%).
    Useful for EDA to confirm which columns to drop.
    """
    target_cols = [c for c in df.columns if c not in ("Date", "Time", "datetime")]
    miss = df[target_cols].isnull().sum()
    pct  = 100 * miss / len(df)
    return pd.DataFrame({"n_missing": miss, "pct_missing": pct.round(1)}).sort_values(
        "pct_missing", ascending=False
    )


# ---------------------------------------------------------------------------
# Feature / target extraction
# ---------------------------------------------------------------------------

def get_Xy(df: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Return cleaned X (n, 10) and Y (n, 4). Drops NMHC(GT) (>90% missing),
    drops rows with any NaN target, and median-imputes feature NaNs.
    """
    if df is None:
        df = load_raw()

    output_raw_names = ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]

    # Step 1: check all required columns exist
    missing_cols = [c for c in output_raw_names + _SENSOR_COLS + _MET_COLS
                    if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

    # Step 2: drop rows where any target is NaN
    df = df.dropna(subset=output_raw_names).reset_index(drop=True)

    # Note: feature NaN imputation is intentionally deferred to split_and_scale()
    # so that medians are computed only from training data (no data leakage).

    # Ensure time features are present
    if "HourSin" not in df.columns:
        hour = df["datetime"].dt.hour
        df["HourSin"] = np.sin(2 * np.pi * hour / 24)
        df["HourCos"] = np.cos(2 * np.pi * hour / 24)

    X = df[FEATURE_NAMES].values.astype(float)
    Y = df[output_raw_names].values.astype(float)
    return X, Y


# ---------------------------------------------------------------------------
# Train / val / test splitting
# ---------------------------------------------------------------------------

def split_and_scale(
    X: np.ndarray,
    Y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    n_subsample: int | None = None,
    random_state: int = 42,
) -> dict:
    """
    Optional subsampling → train/val/test split → StandardScaler on X.
    Returns a dict with X_train/val/test, Y_train/val/test, scaler_X, and counts.
    Pass n_subsample=GP_SUBSAMPLE for GP experiments; None for the full pool.
    """
    rng = np.random.RandomState(random_state)

    if n_subsample is not None and n_subsample < len(X):
        idx = rng.choice(len(X), size=n_subsample, replace=False)
        X, Y = X[idx], Y[idx]

    X_tv, X_test, Y_tv, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    val_frac = val_size / (1 - test_size)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_tv, Y_tv, test_size=val_frac, random_state=random_state
    )

    # Median-impute feature NaNs using training-set medians only (no leakage)
    n_features = X_train.shape[1]
    train_medians = np.nanmedian(X_train, axis=0)
    for fi in range(n_features):
        nan_mask_tr  = np.isnan(X_train[:, fi])
        nan_mask_val = np.isnan(X_val[:, fi])
        nan_mask_te  = np.isnan(X_test[:, fi])
        if nan_mask_tr.any():
            X_train[nan_mask_tr, fi] = train_medians[fi]
        if nan_mask_val.any():
            X_val[nan_mask_val, fi] = train_medians[fi]
        if nan_mask_te.any():
            X_test[nan_mask_te, fi] = train_medians[fi]

    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val   = scaler_X.transform(X_val)
    X_test  = scaler_X.transform(X_test)

    return dict(
        X_train=X_train, X_val=X_val, X_test=X_test,
        Y_train=Y_train, Y_val=Y_val, Y_test=Y_test,
        scaler_X=scaler_X,
        n_train=len(X_train), n_val=len(X_val), n_test=len(X_test),
    )


def subsample_train(splits: dict, n: int, random_state: int = 0) -> dict:
    """
    Return a copy of splits with training data subsampled to n points.
    Used for the low-data regime experiments.
    """
    rng = np.random.RandomState(random_state)
    idx = rng.choice(splits["n_train"], size=min(n, splits["n_train"]), replace=False)
    new = dict(splits)
    new["X_train"] = splits["X_train"][idx]
    new["Y_train"] = splits["Y_train"][idx]
    new["n_train"] = len(idx)
    return new
