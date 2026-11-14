"""Step 3: baselines + evaluation harness for the M5 subset (self-contained)."""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

# ---------- settings (all defined here) ----------
ROOT = Path(__file__).resolve().parent.parent
PANEL_PATH = ROOT / "data" / "processed" / "panel_subset.parquet"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_DAYS = 1941
HORIZON = 28
# split name -> forecast origin (last day whose sales are known to the forecaster)
SPLITS = {"val": 1885, "test": 1913}

# ---------- load panel and build a series x day matrix ----------
panel = pd.read_parquet(PANEL_PATH, columns=["id", "cat_id", "store_id", "d", "sales"])
panel["day_num"] = panel["d"].str.slice(2).astype(int)

ids = np.sort(panel["id"].unique())
id_to_row = {s: i for i, s in enumerate(ids)}
rows = panel["id"].map(id_to_row).to_numpy()
cols = panel["day_num"].to_numpy() - 1

# NaN = day not in panel (pre-launch days were dropped in Step 2)
M = np.full((len(ids), N_DAYS), np.nan, dtype=np.float32)
M[rows, cols] = panel["sales"].to_numpy(dtype=np.float32)

meta = panel.drop_duplicates("id").set_index("id").loc[ids, ["cat_id", "store_id"]]
cat_arr = meta["cat_id"].to_numpy()
print(f"Matrix: {M.shape[0]} series x {M.shape[1]} days, observed cells: {np.isfinite(M).sum():,}")


# ---------- forecasting methods (direct, 28-day horizon from a fixed origin) ----------
def forecast(method: str, origin: int) -> np.ndarray:
    """Return (n_series, HORIZON) forecasts for days origin+1 .. origin+HORIZON.
    Only uses days <= origin."""
    out = np.zeros((M.shape[0], HORIZON), dtype=np.float64)
    for h in range(1, HORIZON + 1):
        t = origin + h
        k = int(np.ceil(h / 7))  # whole weeks back so that the source day is <= origin
        if method == "zero":
            out[:, h - 1] = 0.0
        elif method == "seasonal_naive":
            # same weekday, most recent occurrence on or before the origin
            src = t - 7 * k
            out[:, h - 1] = np.nan_to_num(M[:, src - 1], nan=0.0)
        elif method == "seasonal_mean4":
            # mean of the last 4 same-weekday values before the origin
            srcs = [t - 7 * (k + j) - 1 for j in range(4)]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                vals = np.nanmean(M[:, srcs], axis=1)
            out[:, h - 1] = np.nan_to_num(vals, nan=0.0)
        else:
            raise ValueError(method)
    return out


def score(pred: np.ndarray, origin: int):
    """MAE / RMSE over observed (non-NaN) actual cells in the horizon."""
    actual = M[:, origin: origin + HORIZON].astype(np.float64)  # days origin+1..origin+HORIZON
    mask = np.isfinite(actual)
    err = (pred - actual)[mask]
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2))), int(mask.sum()), mask, actual


records = []
for split, origin in SPLITS.items():
    for method in ["zero", "seasonal_naive", "seasonal_mean4"]:
        pred = forecast(method, origin)
        mae, rmse, n, mask, actual = score(pred, origin)
        records.append({"split": split, "method": method, "level": "all", "group": "all",
                        "MAE": mae, "RMSE": rmse, "n_cells": n})
        for cat in np.unique(cat_arr):
            sel = (cat_arr == cat)[:, None] & mask
            e = (pred - actual)[sel]
            records.append({"split": split, "method": method, "level": "category", "group": cat,
                            "MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
                            "n_cells": int(sel.sum())})

res = pd.DataFrame(records)
res.to_csv(RESULTS_DIR / "baseline_metrics.csv", index=False)

print("\n=== Overall (all 1000 series) ===")
print(res[res["level"] == "all"][["split", "method", "MAE", "RMSE", "n_cells"]]
      .round(4).to_string(index=False))
print("\n=== By category ===")
print(res[res["level"] == "category"][["split", "method", "group", "MAE", "RMSE"]]
      .round(4).to_string(index=False))
print(f"\nSaved: {RESULTS_DIR / 'baseline_metrics.csv'}")