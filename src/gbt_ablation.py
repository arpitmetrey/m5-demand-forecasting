"""Step 5: gradient-boosted trees + feature-group ablation (self-contained)."""
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# ---------- settings ----------
ROOT = Path(__file__).resolve().parent.parent
PANEL_PATH = ROOT / "data" / "processed" / "panel_subset.parquet"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_DAYS = 1941
MIN_DAY = 57
VAL_RANGE = (1886, 1913)
TEST_RANGE = (1914, 1941)
TRAIN_END_FOR_VAL = 1885
TRAIN_END_FOR_TEST = 1913
LOSSES = ["squared_error", "poisson"]      # tuned on validation RMSE
GBT_PARAMS = dict(learning_rate=0.1, max_iter=120, max_leaf_nodes=31,
                  min_samples_leaf=200, l2_regularization=1.0,
                  early_stopping=False, random_state=42)

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

H = np.zeros((len(ids), N_DAYS), dtype=np.float32)
H[r, c] = y
Pm = np.zeros((len(ids), N_DAYS), dtype=np.float32)
Pm[r, c] = panel["sell_price"].to_numpy(dtype=np.float32)


def shift(A, k):
    """out[:, t] = A[:, t-k]; zeros for t < k."""
    out = np.zeros_like(A)
    out[:, k:] = A[:, :-k]
    return out


# ---------- features (all history features look back >= 28 days) ----------
feat = {}
for k in (28, 35, 42, 56):
    feat[f"lag_{k}"] = shift(H, k)[r, c]

H28 = shift(H, 28)
S = np.cumsum(H28, axis=1)


def roll_mean(w):
    return (S - shift(S, w)) / w


feat["roll7_l28"] = roll_mean(7)[r, c]
feat["roll28_l28"] = roll_mean(28)[r, c]

price = panel["sell_price"].to_numpy(dtype=np.float32)
price_l28 = shift(Pm, 28)[r, c]
feat["sell_price"] = price
feat["price_ratio_l28"] = np.where(price_l28 > 0, price / np.maximum(price_l28, 1e-6), 1.0).astype(np.float32)

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
print(f"Rows: {len(X):,}  features: {len(names)}")

CONFIGS = {
    "static_only": G_STATIC,
    "+history": G_STATIC + G_HIST,
    "+calendar": G_STATIC + G_CAL,
    "+price": G_STATIC + G_PRICE,
    "+history+calendar": G_STATIC + G_HIST + G_CAL,
    "+history+price": G_STATIC + G_HIST + G_PRICE,
    "all_features": G_STATIC + G_HIST + G_CAL + G_PRICE,
}


def masks(train_end, eval_range):
    tr = (day >= MIN_DAY) & (day <= train_end)
    ev = (day >= eval_range[0]) & (day <= eval_range[1])
    return tr, ev


def metrics(pred, actual):
    e = pred - actual
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e ** 2)))


def fit_eval(cols, loss, train_end, eval_range):
    idx = [col_idx[n] for n in cols]
    tr, ev = masks(train_end, eval_range)
    model = HistGradientBoostingRegressor(loss=loss, **GBT_PARAMS)
    model.fit(Xa[tr][:, idx], y[tr])
    pred = np.clip(model.predict(Xa[ev][:, idx]), 0, None)
    return metrics(pred, y[ev]) + (int(ev.sum()),)


rows_search, rows_final = [], []
t0 = time.time()
for name, cols in CONFIGS.items():
    val_res = {}
    for loss in LOSSES:
        mae, rmse, n = fit_eval(cols, loss, TRAIN_END_FOR_VAL, VAL_RANGE)
        val_res[loss] = (mae, rmse)
        rows_search.append({"config": name, "loss": loss, "val_MAE": mae, "val_RMSE": rmse, "n_val": n})
    best_loss = min(val_res, key=lambda l: val_res[l][1])
    tmae, trmse, tn = fit_eval(cols, best_loss, TRAIN_END_FOR_TEST, TEST_RANGE)
    rows_final.append({"config": name, "n_features": len(cols), "best_loss": best_loss,
                       "val_MAE": val_res[best_loss][0], "val_RMSE": val_res[best_loss][1],
                       "test_MAE": tmae, "test_RMSE": trmse, "n_test": tn})
    print(f"{name:20s} loss={best_loss:<14s} val MAE {val_res[best_loss][0]:.4f} "
          f"RMSE {val_res[best_loss][1]:.4f} | test MAE {tmae:.4f} RMSE {trmse:.4f} "
          f"[{(time.time() - t0) / 60:.1f} min]")

pd.DataFrame(rows_search).to_csv(RESULTS_DIR / "gbt_loss_search.csv", index=False)
final = pd.DataFrame(rows_final)
final.to_csv(RESULTS_DIR / "gbt_ablation.csv", index=False)

print("\n=== GBT feature-group ablation (loss tuned on validation) ===")
print(final.round(4).to_string(index=False))

for fname, label in [("baseline_metrics.csv", "Baselines"), ("ridge_ablation.csv", "Ridge ablation")]:
    f = RESULTS_DIR / fname
    if f.exists():
        d = pd.read_csv(f)
        if fname == "baseline_metrics.csv":
            d = d[d["level"] == "all"][["split", "method", "MAE", "RMSE"]]
        print(f"\n=== {label} for reference ===")
        print(d.round(4).to_string(index=False))

print(f"\nSaved: {RESULTS_DIR / 'gbt_ablation.csv'} and gbt_loss_search.csv")