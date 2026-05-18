# Financial Machine Learning for Stock Crash Prediction

A publication-oriented financial machine learning pipeline for predicting future stock crash risk using technical indicators, rolling-window validation, non-overlapping sampling, SHAP explainability, and ensemble tree models.

---

# Project Overview

This project builds a complete end-to-end framework for:

* Stock crash risk labeling
* Technical feature engineering
* Fundamental feature integration
* Non-overlapping sample construction
* Purged rolling-window backtesting
* Multi-model machine learning training
* SHAP-based interpretability analysis
* Strategy backtesting and visualization

The pipeline is designed to avoid common pitfalls in financial ML such as:

* Label leakage
* Overlapping-window bias
* Temporal leakage
* Imbalanced classification distortion
* Unrealistic evaluation methodology

The project supports both:

1. Binary crash prediction
2. Multi-class crash severity prediction

---

# Dataset

## Universe

* 165 US stocks
* Daily frequency
* Period: 2015–2025

## Core Files

| File                     | Description                       |
| ------------------------ | --------------------------------- |
| `Prices_fixed.csv`       | Daily stock prices                |
| `stock_sectors.csv`      | Sector classification             |
| `day3_dataset_mdd.csv`   | Crash-labeled dataset             |
| `day4_features.csv`      | Technical feature dataset         |
| `v2_final_results.csv`   | Final model evaluation results    |
| `v2_rolling_results.csv` | Rolling-window validation results |

---

# Project Structure

```text
.
├── day0_expand_stocks.ipynb
├── day3_drawdown_labeling.ipynb
├── day4_feature_engineering.ipynb
├── day5_fundamental_scraper.ipynb
├── day6_fundamental_features.ipynb
├── day7_merge_features.ipynb
├── day8_model_training.ipynb
├── day10_model_evaluation.ipynb
│
├── analyze_thresholds.py
├── final_analysis.py
├── main_pipeline.py
│
├── Prices_fixed.csv
├── stock_sectors.csv
├── day3_dataset_mdd.csv
├── day4_features.csv
├── v2_feature_list.txt
├── v2_final_results.csv
├── v2_rolling_results.csv
│
├── models/
├── figures/
└── README.md
```

---

# Methodology

# 1. Crash Label Construction

The project defines future crash risk using future maximum drawdown (MDD).

For each stock and date:

```math
future\_mdd_{20} = \frac{\min(P_{t+1:t+20})}{P_t} - 1
```

Binary crash label:

```python
crash = 1 if future_mdd_20 <= -0.10 else 0
```

Multi-class labels:

| Label | Definition                 |
| ----- | -------------------------- |
| 0     | No crash                   |
| 1     | Mild drawdown (-5% ~ -10%) |
| 2     | Crash (-10% ~ -15%)        |
| 3     | Severe crash (< -15%)      |

The project also implements:

* Multi-threshold experiments
* Multi-horizon prediction
* NCSKEW-based academic crash definitions

Reference implementation available in:

* `analyze_thresholds.py`
* `main_pipeline.py`

---

# 2. Technical Feature Engineering

The project constructs a large set of technical indicators including:

## Momentum Features

* `ret_5d`
* `ret_10d`
* `ret_20d`
* `ret_60d`
* `ret_120d`

## Volatility Features

* `vol_5d`
* `vol_10d`
* `vol_20d`
* `vol_60d`
* `atr_14`
* `atr_pct`

## Trend Features

* Moving averages
* MA gaps
* MA trend ratios
* MACD indicators

## Statistical Features

* `skew_20d
