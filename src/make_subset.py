"""Step 2: choose a reproducible subset of series and build a long-format panel.

The subset is chosen from metadata only (store and category), never from sales
values, so the selection cannot leak information about the test period.

Run from the project root (project_02) with:
    python -m src.make_subset
"""
import numpy as np
import pandas as pd

from src.config import (CALENDAR_CSV, SALES_EVAL_CSV, PRICES_CSV, PANEL_PARQUET,
                        SUBSET_IDS_CSV, PROCESSED_DIR, RESULTS_DIR, SEED, N_PER_STORE)

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def load_sales(path):
    """Read a wide sales file; daily counts as int16 and id columns as category."""
    header = pd.read_csv(path, nrows=0).columns
    dtypes = {c: "int16" for c in header if c.startswith("d_")}
    dtypes.update({c: "category" for c in ID_COLS})
    return pd.read_csv(path, dtype=dtypes)


def select_subset(meta, n_per_store, seed):
    """Sample ~n_per_store series per store, split across categories in proportion
    to that store's category mix. Uses only id/metadata columns."""
    parts = []
    for store in sorted(meta["store_id"].unique()):
        g = meta[meta["store_id"] == store]
        shares = g["cat_id"].value_counts(normalize=True)
        for cat, share in shares.items():
            k = int(round(share * n_per_store))
            pool = g[g["cat_id"] == cat]
            if k > 0:
                parts.append(pool.sample(n=min(k, len(pool)), random_state=seed))
    return pd.concat(parts)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. load sales and choose the subset from metadata only ----
    sales = load_sales(SALES_EVAL_CSV)
    d_cols = [c for c in sales.columns if c.startswith("d_")]
    for c in ID_COLS:
        sales[c] = sales[c].astype(str)

    chosen = select_subset(sales[ID_COLS], N_PER_STORE, SEED)
    chosen = chosen.sort_values(["store_id", "cat_id", "id"])
    chosen.to_csv(SUBSET_IDS_CSV, index=False)
    print("selected series:", len(chosen), "| seed:", SEED, "| per store target:", N_PER_STORE)
    print(pd.crosstab(chosen["store_id"], chosen["cat_id"]).to_string())

    sub = sales.loc[chosen.index].copy()
    del sales

    # ---- 2. wide -> long (one row per item-store-day) ----
    long = sub.melt(id_vars=ID_COLS, value_vars=d_cols, var_name="d", value_name="sales")
    long["sales"] = long["sales"].astype("int16")
    long["day"] = long["d"].str.slice(2).astype("int16")
    del sub

    # ---- 3. merge calendar features ----
    cal = pd.read_csv(CALENDAR_CSV, parse_dates=["date"])
    cal_cols = ["d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
                "event_name_1", "event_type_1", "event_name_2", "event_type_2",
                "snap_CA", "snap_TX", "snap_WI"]
    long = long.merge(cal[cal_cols], on="d", how="left", validate="many_to_one")

    # SNAP indicator that applies to this series' own state
    long["snap"] = np.select(
        [long["state_id"] == "CA", long["state_id"] == "TX", long["state_id"] == "WI"],
        [long["snap_CA"], long["snap_TX"], long["snap_WI"]],
    ).astype("int8")
    long = long.drop(columns=["snap_CA", "snap_TX", "snap_WI"])
    for c in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        long[c] = long[c].fillna("none")

    # ---- 4. prices: keep only rows for the selected store-item pairs ----
    prices = pd.read_csv(PRICES_CSV, dtype={"store_id": "category", "item_id": "category"})
    keep = (prices["store_id"].isin(chosen["store_id"].unique())
            & prices["item_id"].isin(chosen["item_id"].unique()))
    prices = prices[keep].copy()
    prices["store_id"] = prices["store_id"].astype(str)
    prices["item_id"] = prices["item_id"].astype(str)
    pairs = chosen[["store_id", "item_id"]].drop_duplicates()
    prices = prices.merge(pairs, on=["store_id", "item_id"], how="inner")

    # ---- 5. drop pre-launch days (before the item's first price record) ----
    first_week = (prices.groupby(["store_id", "item_id"])["wm_yr_wk"].min()
                  .rename("first_price_week").reset_index())
    n_before = len(long)
    zero_before = float((long["sales"] == 0).mean())
    long = long.merge(first_week, on=["store_id", "item_id"], how="left", validate="many_to_one")
    long = long[long["wm_yr_wk"] >= long["first_price_week"]].drop(columns="first_price_week")
    n_after = len(long)
    zero_after = float((long["sales"] == 0).mean())

    # ---- 6. merge weekly selling price (left join keeps every remaining day) ----
    n_rows = len(long)
    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left",
                      validate="many_to_one")
    assert len(long) == n_rows, "price merge changed the number of rows"
    long["sell_price"] = long["sell_price"].astype("float32")

    # ---- 7. tidy, sort, save ----
    long = long.sort_values(["id", "day"]).reset_index(drop=True)
    for c in ID_COLS:
        long[c] = long[c].astype("category")
    long.to_parquet(PANEL_PARQUET, index=False)

    # ---- 8. summary for the paper ----
    print("\n" + "=" * 70)
    print("PANEL SUMMARY")
    print("=" * 70)
    print("rows before dropping pre-launch days:", n_before)
    print("rows after dropping pre-launch days :", n_after,
          f"({100 * (n_before - n_after) / n_before:.1f}% dropped)")
    print("zero-sales fraction before / after dropping:",
          round(zero_before, 3), "/", round(zero_after, 3))
    print("remaining rows with missing sell_price:", int(long["sell_price"].isna().sum()))
    print("series:", long["id"].nunique(),
          "| day range:", int(long["day"].min()), "to", int(long["day"].max()))
    print("mean daily sales:", round(float(long["sales"].mean()), 3),
          "| max:", int(long["sales"].max()))
    lens = long.groupby("id", observed=True).size()
    print("days of history per series (min / median / max):",
          int(lens.min()), int(lens.median()), int(lens.max()))
    print("series with fewer than 400 days of history:", int((lens < 400).sum()))
    print("panel memory (MB):", round(long.memory_usage(deep=True).sum() / 1e6, 1))
    print("columns:", list(long.columns))

    check = pd.read_parquet(PANEL_PARQUET)
    print("\nsaved:", PANEL_PARQUET)
    print("reloaded shape:", check.shape)
    print("selected ids saved to:", SUBSET_IDS_CSV)
    print("\nStep 2 finished.")


if __name__ == "__main__":
    main()