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
- `results/` derived outputs (metrics, ablations, error analysis, bootstrap, test-period predictions) and `results/figures/`; the raw M5 data is not included
- `paper/` final 4-page paper (PDF) and its LaTeX source
- `docs/` static interactive website (open `docs/index.html`, or use the GitHub Pages link below)
- `requirements.txt` minimal packages; `requirements-lock.txt` full pip freeze of the author's environment

Interactive website (GitHub Pages): https://arpitmetrey.github.io/m5-demand-forecasting/

## Data access

The raw data is not redistributed here. It comes from the Kaggle competition "M5 Forecasting - Accuracy": https://www.kaggle.com/competitions/m5-forecasting-accuracy . A free Kaggle account is needed, and the competition rules must be accepted before the files can be downloaded. Either download the files from the competition's Data tab, or use the Kaggle command line tool:

```
kaggle competitions download -c m5-forecasting-accuracy
```

Unzip the files into a folder named `data/` in the project root. The scripts use `calendar.csv`, `sell_prices.csv`, `sales_train_validation.csv` and `sales_train_evaluation.csv`.

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

2. Put the M5 files in `data/` as described under Data access.

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
- The repository contains no private data, credentials or API keys.

## AI-use disclosure

The topic, research question and project proposal are my own. I used Claude (Anthropic) throughout the project as the only AI tool, for: design suggestions, writing and debugging the Python scripts, flagging problems in outputs, drafting the text of the paper and of this README, and building the interactive website in `docs/`. I ran every script locally. No other AI tools were used.

## Citations and external material

Data
- S. Makridakis, E. Spiliotis and V. Assimakopoulos. M5 accuracy competition: Results, findings, and conclusions. International Journal of Forecasting, 38(4):1346-1364, 2022. Data: Kaggle, "M5 Forecasting - Accuracy" (link above).

Methods
- R. J. Hyndman and G. Athanasopoulos. Forecasting: Principles and Practice, 3rd ed. OTexts, 2021 (seasonal naive baseline).
- A. E. Hoerl and R. W. Kennard. Ridge regression: Biased estimation for nonorthogonal problems. Technometrics, 12(1):55-67, 1970.
- J. H. Friedman. Greedy function approximation: A gradient boosting machine. Annals of Statistics, 29(5):1189-1232, 2001.
- B. Efron. Bootstrap methods: Another look at the jackknife. Annals of Statistics, 7(1):1-26, 1979.

Software and libraries
- F. Pedregosa et al. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825-2830, 2011 (Ridge, HistGradientBoostingRegressor, StandardScaler).
- NumPy, pandas, matplotlib and pyarrow (versions above).
- Chart.js (https://www.chartjs.org/, MIT licence), loaded from the jsDelivr CDN by `docs/index.html` for the website charts.
- The paper uses the course-supplied ES 678 LaTeX template and style file (`paper/latex_source/neurips_2026.sty`).
