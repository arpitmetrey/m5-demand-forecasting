"""Step 6: error analysis, paired bootstrap CIs, figures (self-contained)."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# ---------- settings ----------
ROOT = Path(__file__).resolve().parent.parent
PANEL_PATH = ROOT / "data" / "processed" / "panel_subset.parquet"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_DAYS = 1941
MIN_DAY = 57
ORIGIN = 1913                      # last known day for the test forecast
TEST_RANGE = (1914, 1941)
RIDGE_ALPHA = 0.1                  # best alpha from the validation search
GBT_PARAMS = dict(loss="squared_error", learning_rate=0.1, max_iter=120, max_leaf_nodes=31,
                  min_samples_leaf=200, l2_regularization=1.0, early_stopping=False,
                  random_state=42)
N_BOOT = 2000
SEED = 42
METHODS = ["mean4", "ridge", "gbt"]
LABELS = {"mean4": "4-wk weekday mean", "ridge": "Ridge (all features)", "gbt": "GBT (all features)"}

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
    out = np.zeros_like(A)
    out[:, k:] = A[:, :-k]
    return out


# ---------- features ----------
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
Xa = X.to_numpy(dtype=np.float32)
col_idx = {n: i for i, n in enumerate(names)}
idx_all = [col_idx[n] for n in names]
print(f"Rows: {len(X):,}  features: {len(names)}")

# ---------- train final models on days <= 1913, predict the test window ----------
tr = (day >= MIN_DAY) & (day <= ORIGIN)
te = (day >= TEST_RANGE[0]) & (day <= TEST_RANGE[1])

scaler = StandardScaler().fit(Xa[tr])
ridge = Ridge(alpha=RIDGE_ALPHA).fit(scaler.transform(Xa[tr]), y[tr])
p_ridge = np.clip(ridge.predict(scaler.transform(Xa[te])), 0, None)
print("Ridge trained.")

gbt = HistGradientBoostingRegressor(**GBT_PARAMS).fit(Xa[tr], y[tr])
p_gbt = np.clip(gbt.predict(Xa[te]), 0, None)
print("GBT trained.")

# 4-week same-weekday mean baseline from the origin
rt, ct_ = r[te], c[te]
h = day[te] - ORIGIN
k = np.ceil(h / 7).astype(int)
p_m4 = np.mean([H[rt, ct_ - 7 * (k + j)] for j in range(4)], axis=0)

# ---------- test frame ----------
vol_mean = H[:, ORIGIN - 91: ORIGIN].mean(axis=1)           # mean daily sales, last 91 train days
T = pd.DataFrame({
    "row": rt, "day": day[te], "h": h, "y": y[te].astype(np.float64),
    "mean4": p_m4, "ridge": p_ridge, "gbt": p_gbt,
    "store": store[te], "cat": cat[te],
    "snap": X["snap"].to_numpy()[te].astype(int),
    "event": X["is_event1"].to_numpy()[te].astype(int),
    "vol_mean": vol_mean[rt],
})
T["hblock"] = pd.cut(T["h"], [0, 7, 14, 21, 28], labels=["1-7", "8-14", "15-21", "22-28"])
T["vol_bucket"] = pd.cut(T["vol_mean"], [-0.01, 0.25, 1, 3, np.inf],
                         labels=["<0.25", "0.25-1", "1-3", ">=3"])
T["zero"] = np.where(T["y"] == 0, "y=0", "y>0")
T["snap_lbl"] = np.where(T["snap"] == 1, "SNAP day", "non-SNAP")
T["event_lbl"] = np.where(T["event"] == 1, "event day", "no event")
T.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)


def summarize(g):
    out = {"n_cells": len(g)}
    for m in METHODS:
        e = g[m].to_numpy() - g["y"].to_numpy()
        out[f"MAE_{m}"] = np.abs(e).mean()
        out[f"RMSE_{m}"] = np.sqrt((e ** 2).mean())
    return out


def table(keycol):
    rows = []
    for key in T[keycol].dropna().unique():
        d = summarize(T[T[keycol] == key])
        d[keycol] = key
        rows.append(d)
    out = pd.DataFrame(rows)
    return out[[keycol] + [c_ for c_ in out.columns if c_ != keycol]]


# ---------- 1. overall ----------
overall = pd.DataFrame([{"method": m,
                         "MAE": np.abs(T[m] - T["y"]).mean(),
                         "RMSE": np.sqrt(((T[m] - T["y"]) ** 2).mean()),
                         "bias (pred - actual)": (T[m] - T["y"]).mean()} for m in METHODS])
print("\n=== Overall test (28,000 cells) ===")
print(overall.round(4).to_string(index=False))
print(f"(sanity check: 4-wk mean MAE should be ~1.1567, got {overall.loc[0, 'MAE']:.4f})")
overall.to_csv(RESULTS_DIR / "error_overall.csv", index=False)

# ---------- 2-6. slices ----------
for key, title, fname in [("hblock", "By horizon block (days ahead)", "error_by_horizon_block.csv"),
                          ("cat", "By category", "error_by_category.csv"),
                          ("store", "By store", "error_by_store.csv"),
                          ("zero", "Zero vs non-zero actual cells", "error_by_zero.csv"),
                          ("event_lbl", "Event vs non-event days", "error_by_event.csv"),
                          ("snap_lbl", "SNAP vs non-SNAP days", "error_by_snap.csv")]:
    tb = table(key)
    print(f"\n=== {title} ===")
    print(tb.round(4).to_string(index=False))
    tb.to_csv(RESULTS_DIR / fname, index=False)

# volume buckets, with series counts and share of total sales
vb = table("vol_bucket")
ser = pd.DataFrame({"row": np.arange(len(ids)), "vol_mean": vol_mean})
ser["vol_bucket"] = pd.cut(ser["vol_mean"], [-0.01, 0.25, 1, 3, np.inf], labels=["<0.25", "0.25-1", "1-3", ">=3"])
vb["n_series"] = vb["vol_bucket"].map(ser["vol_bucket"].value_counts()).astype(int)
tot = T["y"].sum()
vb["share_of_test_sales"] = vb["vol_bucket"].map(T.groupby("vol_bucket", observed=True)["y"].sum() / tot)
vb["order"] = vb["vol_bucket"].map({"<0.25": 0, "0.25-1": 1, "1-3": 2, ">=3": 3})
vb = vb.sort_values("order").drop(columns="order")
print("\n=== By sales volume (mean daily sales over last 91 training days) ===")
print(vb.round(4).to_string(index=False))
vb.to_csv(RESULTS_DIR / "error_by_volume.csv", index=False)

# per-horizon MAE for the figure
per_h = T.groupby("h").apply(lambda g: pd.Series({m: np.abs(g[m] - g["y"]).mean() for m in METHODS}),
                             include_groups=False)

# ---------- 7. paired bootstrap over series ----------
nS = len(ids)
cnt = np.bincount(T["row"], minlength=nS).astype(np.float64)
abs_sum = {m: np.bincount(T["row"], weights=np.abs(T[m] - T["y"]), minlength=nS) for m in METHODS}
sq_sum = {m: np.bincount(T["row"], weights=(T[m] - T["y"]) ** 2, minlength=nS) for m in METHODS}
rng = np.random.default_rng(SEED)
samp = rng.integers(0, nS, size=(N_BOOT, nS))
den = cnt[samp].sum(axis=1)
boot_mae = {m: abs_sum[m][samp].sum(axis=1) / den for m in METHODS}
boot_rmse = {m: np.sqrt(sq_sum[m][samp].sum(axis=1) / den) for m in METHODS}
full_mae = {m: abs_sum[m].sum() / cnt.sum() for m in METHODS}
full_rmse = {m: np.sqrt(sq_sum[m].sum() / cnt.sum()) for m in METHODS}

rows = []
for a, b in [("ridge", "mean4"), ("gbt", "mean4"), ("gbt", "ridge")]:
    for metric, boot, full in [("MAE", boot_mae, full_mae), ("RMSE", boot_rmse, full_rmse)]:
        diff = boot[a] - boot[b]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        rows.append({"comparison": f"{a} - {b}", "metric": metric,
                     "difference": full[a] - full[b], "CI95_low": lo, "CI95_high": hi,
                     "CI_excludes_0": bool(lo > 0 or hi < 0)})
boot_tbl = pd.DataFrame(rows)
print(f"\n=== Paired bootstrap over series ({N_BOOT} resamples; negative = first model better) ===")
print(boot_tbl.round(4).to_string(index=False))
boot_tbl.to_csv(RESULTS_DIR / "bootstrap_differences.csv", index=False)

# ---------- figures ----------
plt.rcParams.update({"font.size": 9})
colors = {"mean4": "#888888", "ridge": "#1f77b4", "gbt": "#d62728"}

# Fig 1: ablation (test MAE) for Ridge and GBT vs the 4-wk mean baseline
rf, gf = RESULTS_DIR / "ridge_ablation.csv", RESULTS_DIR / "gbt_ablation.csv"
if rf.exists() and gf.exists():
    ra, ga = pd.read_csv(rf), pd.read_csv(gf)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    xs = np.arange(len(ra))
    ax.bar(xs - 0.2, ra["test_MAE"], 0.4, label="Ridge", color=colors["ridge"])
    ax.bar(xs + 0.2, ga["test_MAE"], 0.4, label="GBT", color=colors["gbt"])
    ax.axhline(full_mae["mean4"], color="k", ls="--", lw=1, label="4-wk weekday mean")
    ax.set_xticks(xs)
    ax.set_xticklabels(ra["config"], rotation=30, ha="right")
    ax.set_ylabel("Test MAE")
    ax.set_title("Feature-group ablation")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_ablation.png", dpi=200)
    plt.close(fig)

# Fig 2: MAE by forecast horizon
fig, ax = plt.subplots(figsize=(6.4, 2.8))
for m in METHODS:
    ax.plot(per_h.index, per_h[m], marker="o", ms=3, label=LABELS[m], color=colors[m])
ax.set_xlabel("Days ahead")
ax.set_ylabel("Test MAE")
ax.set_title("Error by forecast horizon")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_horizon.png", dpi=200)
plt.close(fig)

# Fig 3: MAE by sales-volume bucket
fig, ax = plt.subplots(figsize=(6.4, 2.8))
xs = np.arange(len(vb))
for i, m in enumerate(METHODS):
    ax.bar(xs + (i - 1) * 0.27, vb[f"MAE_{m}"], 0.27, label=LABELS[m], color=colors[m])
ax.set_xticks(xs)
ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(vb["vol_bucket"], vb["n_series"])])
ax.set_xlabel("Mean daily sales (last 91 training days)")
ax.set_ylabel("Test MAE")
ax.set_title("Error by sales volume")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_volume.png", dpi=200)
plt.close(fig)

print(f"\nSaved tables and test_predictions.csv in {RESULTS_DIR}")
print(f"Saved figures in {FIG_DIR}")