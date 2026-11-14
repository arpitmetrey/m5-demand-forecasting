# M5 demand forecasting: what do history, calendar and price add?

ES 678 course project, IIT Gandhinagar. Author: Arpit Metrey (roll number 26210175).

## Question

How much do historical sales, calendar information and selling price improve 28-day-ahead daily retail demand forecasts over simple baselines? Data: M5 Walmart forecasting data (Kaggle), a stratified subset of 1,000 item-store series (100 per store, 1,525,319 active item-store-days after dropping pre-launch days).

## Main result (test period d1914-d1941, 28,000 item-store-day cells)

| Method | MAE | RMSE |
|---|---|---|
| Zero forecast | 1.556 | 3.539 |
| Seasonal naive (same weekday last week) | 1.292 | 2.466 |
| 4-week weekday mean (best baseline) | 1.157 | 2.130 |
| Ridge (all features) | 1.115 | 1.966 |
| Gradient-boosted trees (all features) | 1.114 | 1.948 |

- Both models beat the best baseline by about 3.6% in MAE and 8% in RMSE; a paired bootstrap over series gives confidence intervals that exclude zero. Ridge and boosted trees are statistically tied.
- Lagged sales history gives almost all of the gain. Calendar features add about 0.027 MAE on top of history; price adds about 0.002.
- Models are worse than the baseline on zero-sales days and slow-moving items (squared-loss models predict means, MAE favours medians).

## Repository layout

- `src/` scripts (see run order below)
- `results/` CSV outputs (metrics, ablations, error analysis, bootstrap) and `results/figures/`
- `paper/` final 4-page paper (PDF) and its LaTeX source
- `docs/` static interactive website (open `docs/index.html`, or use the GitHub Pages link below)
- `requirements.txt` minimal packages; `requirements-lock.txt` full pip freeze of the author's environment

Interactive website (GitHub Pages): https://arpitmetrey.github.io/m5-demand-forecasting/

## Reproduce

Environment: Python 3.12, scikit-learn 1.9.1, pandas 3.1.0, NumPy 2.5.4, matplotlib 3.11.2, pyarrow 26.0.0, fixed seed 42.

1. Clone the repository and create an environment (Windows shown; on Linux or macOS use `source .venv/bin/activate`):

```
git clone https://github.com/arpitmetrey/m5-demand-forecasting.git
cd m5-demand-forecasting
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Download the M5 Forecasting - Accuracy data from Kaggle and put these files in a folder named `data/` in the project root (raw data is not included in the repository): `calendar.csv`, `sell_prices.csv`, `sales_train_validation.csv`, `sales_train_evaluation.csv`.

3. Run from the project root, in this order:

```
python src/check_data.py          # optional data sanity checks
python src/make_subset.py         # builds the 1,000-series panel (data/processed/panel_subset.parquet)
python src/baselines.py           # zero, seasonal naive and 4-week mean baselines (results/baseline_metrics.csv)
python src/final_experiments.py   # features, Ridge and boosted trees, ablations, error analysis, bootstrap, figures (about 10-15 minutes)
```

All outputs are written to `results/`. The scripts `ridge_ablation.py`, `gbt_ablation.py` and `error_analysis.py` are earlier versions kept for reference; `final_experiments.py` supersedes them and produced the numbers in the paper.

## Notes

- Chronological split: train d <= 1885, validation d1886-d1913 (tuning), test d1914-d1941 (evaluated once after refitting on d <= 1913).
- All history features use lags of at least 28 days, so there is no leakage across the forecast horizon.
