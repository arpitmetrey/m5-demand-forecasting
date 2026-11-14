# M5 Retail Demand Forecasting: Interactive Research Dashboard

This project is an interactive, standalone web interface for **Feature-Based Inventory Demand Forecasting on M5 Walmart Data** (ES 678 Course Project at IIT Gandhinagar by Arpit Metrey, Roll No: 26210175).

## Quick Start (How to Run Locally)

1. Extract the downloaded ZIP archive.
2. Open the extracted folder `m5_demand_forecasting_website` in **VS Code**.
3. Open the VS Code terminal (`Ctrl+\`` or `Cmd+\``).
4. Run the python HTTP server:
   ```bash
   python -m http.server 8000
   ```
5. Open [http://localhost:8000](http://localhost:8000) in your web browser.

---

## 6-Section Architecture

1. **01 — Research Overview**: Problem motivation, target definition, and headline findings.
2. **02 — Dataset Explorer**: M5 Walmart structure, 1,000 stratified series, 1.5M daily observations, and quality checks.
3. **03 — Exploratory Data Analysis**: Zero-inflation (58.8% zeros), sales volume distribution, store-level breakdowns, and SNAP/event effects.
4. **04 — Feature Engineering Laboratory**: Interactive calculation of 28-day direct lags, rolling averages, calendar encodings, price ratios, and leakage prevention rules.
5. **05 — Forecasting Laboratory**: Real-time 28-day What-If forecast simulator comparing Baselines, Ridge, Price-aware Ridge, Random Forest, and GBT.
6. **06 — Evaluation & Interpretation**: Comprehensive benchmark tables, multi-window evaluation across 3 validation horizons, feature ablation matrices, and paired bootstrap significance tests.
