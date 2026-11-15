# M5 Retail Demand Forecasting: interactive results dashboard

This is the static companion website for the ES 678 course project "Feature-Based Inventory Demand Forecasting on M5 Walmart Data: What History, Calendar, and Price Add" (Arpit Metrey, roll number 26210175, IIT Gandhinagar).

Live site (GitHub Pages): https://arpitmetrey.github.io/m5-demand-forecasting/

## What the site shows

Every number on the site is copied from the project's measured results (the paper tables and the CSV files in `../results/`). No model is run in the browser and nothing on the site is simulated.

1. **Research Overview**: question, headline numbers and findings.
2. **Dataset Explorer**: the M5 subset (1,000 item-store series, 1,525,319 rows after dropping pre-launch days) and quality checks.
3. **Exploratory Data Analysis**: zero-inflation and the four sales-volume tiers.
4. **Feature Engineering Laboratory**: the direct 28-day feature groups (47 features) and the no-leakage rule. The code box is a simplified illustration; the real code is `../src/final_experiments.py`.
5. **Results Explorer**: tick feature groups to see the measured test MAE and RMSE of Ridge and boosted trees for that configuration (Table 2 of the paper), and pick a volume tier to see the measured MAE of the 4-week weekday mean, Ridge and boosted trees in that tier.
6. **Evaluation and Interpretation**: benchmark table, model comparison, volume-tier and forecast-week error charts, and the feature-group ablation table.

Models compared in the project: zero forecast, seasonal naive, 4-week weekday mean, Ridge regression, and gradient-boosted trees (HistGradientBoostingRegressor). Evaluation uses one chronological split: train d <= 1885, validation d1886-d1913, test d1914-d1941.

## Run locally

From the repository root:

```
cd docs
python -m http.server 8000
```

Then open http://localhost:8000 in a browser. The page loads Chart.js from the jsDelivr CDN, so an internet connection is needed for the charts.

## Credits

- Charts: Chart.js (https://www.chartjs.org/, MIT licence), loaded from https://cdn.jsdelivr.net.
- The site was built with the help of Claude (Anthropic); see the AI-use disclosure in the main README.
