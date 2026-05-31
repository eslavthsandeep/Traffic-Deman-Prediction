# 🚦 Traffic Demand Prediction — Flipkart GRID Hackathon

> **Final Score: 97.60 / 100** (R² = 0.9760)

A complete end-to-end machine learning pipeline for predicting traffic demand across geospatial locations and time intervals, built for the Flipkart GRID hackathon challenge.

---

## 📋 Problem Statement

Given historical traffic data with features like geohash (location), timestamp, road type, weather, temperature, and lane information, predict the **demand** (continuous value between 0 and 1) for each location-time combination.

- **Evaluation Metric**: `score = max(0, 100 × R²)`
- **Train**: 77,299 rows × 11 columns (days 48–49)
- **Test**: 41,778 rows × 10 columns (day 49)
- **96 timestamps/day** (15-minute intervals), **1,249 unique geohashes**

---

## 🏗️ Pipeline Architecture

The project uses a clean **two-stage pipeline**:

```
Raw Data ──► extract_features.py ──► Feature CSVs ──► train_models.py ──► Submission
```

| Stage | Script | Input | Output |
|-------|--------|-------|--------|
| **1. Feature Extraction** | `extract_features.py` | `dataset/train.csv`, `dataset/test.csv` | `dataset/train_features.csv`, `dataset/test_features.csv` |
| **2. Model Training & Ensemble** | `train_models.py` | Feature CSVs | `output/submission.csv`, `output/results_summary.json` |

---

## 🧠 Feature Engineering (75 Features from 10 Raw Columns)

| Category | Count | Key Features |
|----------|-------|-------------|
| **Time** | 13 | `hour_sin/cos`, `minutes_since_midnight`, `is_rush_hour`, `time_slot` |
| **Day** | 5 | `day_of_week`, `is_weekend`, `dow_sin/cos` |
| **Geohash** | 13 | `geo_frequency`, `geohash_encoded`, prefix encodings, `geo_hour_demand_mean` (target-encoded) |
| **Road** | 10 | `road_capacity`, `road_complexity`, `landmark_lane_interaction` |
| **Temperature** | 5 | `temp_bin`, `temp_squared`, `temp_normalized` |
| **Weather** | 9 | One-hot, `weather_severity`, `is_bad_weather`, target-encoded stats |
| **Interactions** | 10 | `geo × road`, `geo × weather`, `hour × weather` (label + target encoded) |
| **Aggregates** | 10 | `hour_demand_mean/std/median`, `timeslot_demand_mean`, `lane_demand_mean` |

> The **#1 feature** is `geo_hour_demand_mean` — geohash × hour target encoding captures the core pattern that traffic demand is driven by **where** (location) and **when** (time of day).

---

## 🤖 Models & Results

All models trained with **5-Fold Cross-Validation**:

| Model | OOF R² | Score |
|-------|--------|-------|
| **CatBoost** | 0.9754 | 97.54 |
| **LightGBM** | 0.9752 | 97.52 |
| XGBoost | 0.9734 | 97.34 |
| ExtraTrees | 0.9687 | 96.87 |

### Ensemble Strategy

Optimal blend found via grid search over OOF predictions:

```
50% LightGBM + 50% CatBoost → R² = 0.9760 → Score = 97.60
```

---

## 🔍 Missing Value Handling

| Column | Missing % | Strategy |
|--------|-----------|----------|
| **Temperature** | 3.23% | Hierarchical median: `(geohash, hour, weather)` → `(hour, weather)` → `(weather)` → global |
| **Weather** | 1.03% | Mode (Sunny) |
| **RoadType** | 0.78% | Mode (Residential) |

---

## 📊 Visualizations

The pipeline generates EDA plots and feature importance charts in `output/`:

- `eda_demand_distribution.png` — Demand histogram, log-transform, and box plot by weather
- `eda_demand_by_features.png` — Mean demand by road type, lanes, vehicles, landmarks
- `eda_correlation.png` — Correlation matrix of numeric features
- `lgb_feature_importance.png` — Top 25 LightGBM feature importances

---

## 🚀 How to Run

### Prerequisites

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost matplotlib seaborn
```

### Step 1: Extract Features

```bash
python extract_features.py
```

This reads `dataset/train.csv` and `dataset/test.csv`, engineers 75 features, and saves:
- `dataset/train_features.csv`
- `dataset/test_features.csv`

### Step 2: Train Models & Generate Submission

```bash
python train_models.py
```

This loads the feature CSVs, trains 4 models with 5-fold CV, finds the optimal ensemble blend, and saves:
- `output/submission.csv`
- `output/results_summary.json`
- `output/lgb_feature_importance.png`

---

## 📁 Project Structure

```
Traffic-Demand-Prediction/
├── dataset/
│   ├── train.csv                  # Raw training data
│   ├── test.csv                   # Raw test data
│   ├── sample_submission.csv      # Submission format
│   ├── train_features.csv         # Generated features (train)
│   └── test_features.csv          # Generated features (test)
├── output/
│   ├── submission.csv             # Final predictions (97.60 score)
│   ├── results_summary.json       # Model performance metrics
│   ├── eda_correlation.png        # Correlation heatmap
│   ├── eda_demand_by_features.png # Demand by categorical features
│   ├── eda_demand_distribution.png# Distribution analysis
│   └── lgb_feature_importance.png # Feature importance chart
├── extract_features.py            # Stage 1: Feature engineering
├── train_models.py                # Stage 2: Model training & ensemble
├── approach.txt                   # Approach documentation
├── .gitignore
└── README.md
```

---

## 🛠️ Tools & Libraries

- **Python 3.10**
- pandas, numpy, scikit-learn
- LightGBM, XGBoost, CatBoost
- matplotlib, seaborn

---

## 💡 Key Insights

1. Traffic demand is primarily driven by **location** (geohash) and **time** (hour of day)
2. Target encoding with proper train-only computation prevents data leakage
3. CatBoost and LightGBM have complementary error patterns → strong 50/50 ensemble
4. Hierarchical temperature imputation preserves location-specific climate patterns
