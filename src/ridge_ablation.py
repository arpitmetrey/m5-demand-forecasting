"""Step 4: feature engineering + Ridge + feature-group ablation (self-contained)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# ---------- settings ----------
ROOT = Path(__file__).resolve().parent.parent
PANEL_PATH = ROOT / "data" / "processed" / "panel_subset.parquet"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_DAYS = 1941
MIN_DAY = 57                       # first day with all lags available (lag_56)
VAL_RANGE = (1886, 1913)           # validation window
TEST_RANGE = (1914, 1941)          # test window
TRAIN_END_FOR_VAL = 1885           # fit on days <= 1885 to evaluate on validation
TRAIN_END_FOR_TEST = 1913          # refit on days <= 1913 (train + val) to evaluate on test
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]

# ---------- load panel ----------
cols_needed = ["id", "store_id", "cat_id", "d", "sales", "sell_price", "wday", "month",
               "event_name_1", "event_type_1", "event_name_2", "snap"]
panel = pd.read_parquet(PANEL_PATH, columns=cols_needed)
panel["day_num"] = panel["d"].str.slice(2).astype(int)

ids = np.sort(panel["id"].unique())
id_to_row = {s: i for i, s in enumerate(ids)}
r = panel["id"].map(id_to_row).to_numpy()
c = panel["day_num"].to_numpy() - 1
day = panel["day_num"].to_numpy()
y = panel["sales"].to_numpy(dtype=np.float32)

# series x day matrices (NaN -> 0 so pre-launch days count as zero history)
H = np.zeros((len(ids), N_DAYS), dtype=np.float32)
H[r, c] = y
Pm = np.zeros((len(ids), N_DAYS), dtype=np.float32)
Pm[r, c] = panel["sell_price"].to_numpy(dtype=np.float32)


def shift(A, k):
    """out[:, t] = A[:, t-k]; zeros for t < k."""
    out = np.zeros_like(A)
    out[:, k:] = A[:, :-k]
    return out


# ---------- features ----------
# All history features look back at least 28 days, so one model produces a
# 28-day forecast with only information available at the forecast origin.
feat = {}
for k in (28, 35, 42, 56):
    feat[f"lag_{k}"] = shift(H, k)[r, c]

H28 = shift(H, 28)
S = np.cumsum(H28, axis=1)


def roll_mean(w):
    return (S - shift(S, w)) / w


feat["roll7_l28"] = roll_mean(7)[r, c]      # mean of days t-34 .. t-28
feat["roll28_l28"] = roll_mean(28)[r, c]    # mean of days t-55 .. t-28

# price features (planned prices are known in advance)
price = panel["sell_price"].to_numpy(dtype=np.float32)
price_l28 = shift(Pm, 28)[r, c]
feat["sell_price"] = price
feat["price_ratio_l28"] = np.where(price_l28 > 0, price / np.maximum(price_l28, 1e-6), 1.0).astype(np.float32)

# calendar features
wday = panel["wday"].to_numpy()
month = panel["month"].to_numpy()
for g in range(1, 8):
    feat[f"wd_{g}"] = (wday == g).astype(np.float32)
for m in range(1, 13):
    feat[f"mo_{m}"] = (month == m).astype(np.float32)
feat["snap"] = panel["snap"].to_numpy().astype(np.float32)
feat["is_event1"] = panel["event_name_1"].notna().to_numpy().astype(np.float32)
feat["is_event2"] = panel["event_name_2"].notna().to_numpy().astype(np.float32)
et = panel["event_type_1"].to_numpy()
for t in ["Sporting", "Cultural", "National", "Religious"]:
    feat[f"ev_{t}"] = (et == t).astype(np.float32)

# static identity features (included in every model)
store = panel["store_id"].to_numpy()
cat = panel["cat_id"].to_numpy()
for s in np.unique(store):
    feat[f"store_{s}"] = (store == s).astype(np.float32)
for ct in np.unique(cat):
    feat[f"cat_{ct}"] = (cat == ct).astype(np.float32)

X = pd.DataFrame(feat)
names = list(X.columns)
G_HIST = [n for n in names if n.startswith(("lag_", "roll"))]
G_PRICE = ["sell_price", "price_ratio_l28"]
G_CAL = [n for n in names if n.startswith(("wd_", "mo_", "ev_")) or n in ("snap", "is_event1", "is_event2")]
G_STATIC = [n for n in names if n.startswith(("store_", "cat_"))]
Xa = X.to_numpy(dtype=np.float32)
col_idx = {n: i for i, n in enumerate(names)}
print(f"Rows: {len(X):,}  features: {len(names)}  "
      f"(history {len(G_HIST)}, calendar {len(G_CAL)}, price {len(G_PRICE)}, static {len(G_STATIC)})")

CONFIGS = {
    "static_only": G_STATIC,
    "+history": G_STATIC + G_HIST,
    "+calendar": G_STATIC + G_CAL,
    "+price": G_STATIC + G_PRICE,
    "+history+calendar": G_STATIC + G_HIST + G_CAL,
    "+history+price": G_STATIC + G_HIST + G_PRICE,
    "all_features": G_STATIC + G_HIST + G_CAL + G_PRICE,
}


def split_masks(train_end, eval_range):
    tr = (day >= MIN_DAY) & (day <= train_end)
    ev = (day >= eval_range[0]) & (day <= eval_range[1])
    return tr, ev


def metrics(pred, actual):
    e = pred - actual
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e ** 2)))


def fit_eval(cols, train_end, eval_range, alphas):
    idx = [col_idx[n] for n in cols]
    tr, ev = split_masks(train_end, eval_range)
    scaler = StandardScaler().fit(Xa[tr][:, idx])          # fit on training rows only
    Xtr = scaler.transform(Xa[tr][:, idx])
    Xev = scaler.transform(Xa[ev][:, idx])
    out = {}
    for a in alphas:
        model = Ridge(alpha=a).fit(Xtr, y[tr])
        pred = np.clip(model.predict(Xev), 0, None)         # sales cannot be negative
        out[a] = metrics(pred, y[ev]) + (int(ev.sum()),)
    return out


rows_search, rows_final = [], []
for name, cols in CONFIGS.items():
    val_res = fit_eval(cols, TRAIN_END_FOR_VAL, VAL_RANGE, ALPHAS)
    for a, (mae, rmse, n) in val_res.items():
        rows_search.append({"config": name, "alpha": a, "val_MAE": mae, "val_RMSE": rmse, "n_val": n})
    best_a = min(val_res, key=lambda a: val_res[a][1])      # pick alpha by validation RMSE
    test_res = fit_eval(cols, TRAIN_END_FOR_TEST, TEST_RANGE, [best_a])[best_a]
    rows_final.append({"config": name, "n_features": len(cols), "best_alpha": best_a,
                       "val_MAE": val_res[best_a][0], "val_RMSE": val_res[best_a][1],
                       "test_MAE": test_res[0], "test_RMSE": test_res[1], "n_test": test_res[2]})
    print(f"{name:20s} alpha={best_a:<7} val MAE {val_res[best_a][0]:.4f} RMSE {val_res[best_a][1]:.4f} | "
          f"test MAE {test_res[0]:.4f} RMSE {test_res[1]:.4f}")

pd.DataFrame(rows_search).to_csv(RESULTS_DIR / "ridge_alpha_search.csv", index=False)
final = pd.DataFrame(rows_final)
final.to_csv(RESULTS_DIR / "ridge_ablation.csv", index=False)

print("\n=== Ridge feature-group ablation (alpha tuned on validation) ===")
print(final.round(4).to_string(index=False))

ref = RESULTS_DIR / "baseline_metrics.csv"
if ref.exists():
    b = pd.read_csv(ref)
    b = b[b["level"] == "all"][["split", "method", "MAE", "RMSE"]]
    print("\n=== Baselines for reference ===")
    print(b.round(4).to_string(index=False))

print(f"\nSaved: {RESULTS_DIR / 'ridge_ablation.csv'} and ridge_alpha_search.csv")