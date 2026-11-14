"""Step 1: load every M5 table and run basic data-quality checks.

Run from the project root (project_02) with:
    python -m src.check_data
"""
import numpy as np
import pandas as pd

from src.config import CALENDAR_CSV, SALES_EVAL_CSV, SALES_VAL_CSV, PRICES_CSV

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def load_sales(path):
    """Read a wide sales file; daily counts as int16 and id columns as category."""
    header = pd.read_csv(path, nrows=0).columns
    dtypes = {c: "int16" for c in header if c.startswith("d_")}
    dtypes.update({c: "category" for c in ID_COLS})
    return pd.read_csv(path, dtype=dtypes)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    # ---- 0. make sure every file is where we expect it ----
    for p in [CALENDAR_CSV, SALES_EVAL_CSV, SALES_VAL_CSV, PRICES_CSV]:
        status = "OK     " if p.exists() else "MISSING"
        print(f"[{status}] {p}")
        if not p.exists():
            raise SystemExit("Fix the missing file path above and rerun.")

    # ---- 1. calendar ----
    section("CALENDAR")
    cal = pd.read_csv(CALENDAR_CSV, parse_dates=["date"])
    print("shape:", cal.shape)
    print("columns:", list(cal.columns))
    print("date range:", cal["date"].min().date(), "to", cal["date"].max().date())
    print("missing values per column:")
    print(cal.isna().sum().to_string())
    print("(event_* columns are mostly empty by design: most days have no event)")
    print("event_type_1 counts:")
    print(cal["event_type_1"].value_counts(dropna=False).to_string())
    print("SNAP days per state (of", len(cal), "days):",
          {s: int(cal[f"snap_{s}"].sum()) for s in ["CA", "TX", "WI"]})

    # ---- 2. sales (evaluation file = validation file + 28 extra days) ----
    section("SALES (sales_train_evaluation)")
    sales = load_sales(SALES_EVAL_CSV)
    d_cols = [c for c in sales.columns if c.startswith("d_")]
    print("shape:", sales.shape)
    print("memory (MB):", round(sales.memory_usage(deep=True).sum() / 1e6, 1))
    print("number of item-store series:", sales["id"].nunique())
    print("unique items:", sales["item_id"].nunique(),
          "| stores:", sales["store_id"].nunique(),
          "| departments:", sales["dept_id"].nunique(),
          "| categories:", sales["cat_id"].nunique())
    print("number of days:", len(d_cols), "| first:", d_cols[0], "| last:", d_cols[-1])
    print("stores:", sorted(sales["store_id"].unique()))

    values = sales[d_cols].to_numpy()
    print("overall fraction of zero-sales observations:", round(float((values == 0).mean()), 3))
    print("max daily units for one series:", int(values.max()),
          "| min:", int(values.min()))
    print("number of series with at least one negative value:",
          int((values < 0).any(axis=1).sum()))

    # ---- 3. check validation file is a prefix of the evaluation file ----
    section("VALIDATION vs EVALUATION FILE")
    val = load_sales(SALES_VAL_CSV)
    val_d = [c for c in val.columns if c.startswith("d_")]
    print("validation days:", len(val_d), "| evaluation days:", len(d_cols))
    same_rows = bool((val["id"].astype(str).to_numpy() == sales["id"].astype(str).to_numpy()).all())
    same_vals = bool(np.array_equal(val[val_d].to_numpy(), sales[val_d].to_numpy()))
    print("same series order:", same_rows, "| identical sales on shared days:", same_vals)
    del val

    # ---- 4. prices ----
    section("SELL PRICES")
    prices = pd.read_csv(PRICES_CSV, dtype={"store_id": "category", "item_id": "category"})
    print("shape:", prices.shape)
    print("columns:", list(prices.columns))
    print("price min / median / max:",
          prices["sell_price"].min(), prices["sell_price"].median(), prices["sell_price"].max())
    print("missing values:", int(prices.isna().sum().sum()))
    print("store-item pairs with a price record:",
          prices.groupby(["store_id", "item_id"], observed=True).ngroups)
    print("wm_yr_wk range:", prices["wm_yr_wk"].min(), "to", prices["wm_yr_wk"].max())

    # ---- 5. how much of the item-store-day grid has a price? ----
    section("PRICE COVERAGE ON THE SALES GRID")
    week_of_day = cal.set_index("d")["wm_yr_wk"]
    first_week = prices.groupby(["store_id", "item_id"], observed=True)["wm_yr_wk"].min()
    series_keys = list(zip(sales["store_id"].astype(str), sales["item_id"].astype(str)))
    first_week_lookup = {(s, i): w for (s, i), w in first_week.items()}
    day_weeks = week_of_day.loc[d_cols].to_numpy()
    first_w = np.array([first_week_lookup.get(k, 10**9) for k in series_keys])
    has_price = day_weeks[None, :] >= first_w[:, None]
    print("fraction of item-store-days on or after the item's first price record:",
          round(float(has_price.mean()), 3))
    print("series whose sales are zero on every day BEFORE their first price:",
          int(((values == 0) | has_price).all(axis=1).sum()), "of", len(sales))
    print("(zeros before the first price usually mean the item was not yet on sale)")

    print("\nStep 1 verification finished.")


if __name__ == "__main__":
    main()