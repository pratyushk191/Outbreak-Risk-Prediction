# 🦠 Disease Outbreak Prediction — CODECURE AI Hackathon (Track C)

Early detection of infectious disease outbreaks can save lives. This system uses machine learning to predict **which countries are at risk of a case surge in the coming week**, combining epidemiological time-series data with demographics, vaccination rates, hospitalization strain, and excess mortality signals — giving public health teams an early warning before outbreaks accelerate.

## 📋 Deliverables Checklist (Track C)

| Deliverable | Status | Details |
|---|---|---|
| GitHub Repository | ✅ | This repo — full code, data, outputs |
| Outbreak prediction model | ✅ | RF + GradientBoosting ensemble, temporal split, oversampling, recall-constrained threshold tuning |
| Interactive epidemic dashboard | ✅ | 6-tab Streamlit app |
| Risk map of disease spread | ✅ | Static choropleth + animated time-lapse spread map |
| Feature importance analysis | ✅ | Bar chart + category breakdown + epidemiological interpretation |

---

## 📁 Project Structure

```
.
├── disease_spread_model.py                # Training script — RF + GBM ensemble, temporal split, PR-AUC threshold
├── predict_outbreak_risk.py               # Inference — generates predictions CSV
├── data_fusion.py                         # Multi-dataset fusion + Rt estimator + causal rolling MA
├── demographics.py                        # Embedded country-level demographic data (115+ countries)
├── api_app.py                             # FastAPI REST server (8 endpoints, Pydantic validation)
├── dashboard.py                           # Streamlit dashboard (6 tabs)
├── Dockerfile                             # Container for API / dashboard
├── docker-compose.yml                     # Spin up API + dashboard together
├── requirements.txt                       # Pinned Python dependencies
│
├── tests/
│   └── test_pipeline.py                   # 19 unit tests — data, Rt, labels, split, model, regression guards
│
├── full_data.csv                          # Primary OWID COVID dataset
├── vaccinations.csv                       # Vaccination time-series (OWID)
├── covid-hospitalizations.csv             # Hospitalization indicators
├── excess_mortality.csv                   # Excess mortality estimates
├── weekly_cases.csv                       # Weekly case counts
├── weekly_deaths.csv                      # Weekly death counts
├── total_deaths.csv                       # Cumulative deaths
├── time_series_covid19_confirmed_global.csv  # JHU global confirmed cases
│
└── outputs/
    ├── optimal_threshold.txt              # Tuned prediction threshold (0.846)
    ├── metrics.txt                        # PR-AUC, F1, ROC-AUC, classification report
    └── feature_importances.csv            # Top 30 features (excl. country dummies)
```

---

## ⚙️ Setup

> Requires **Python 3.10+**

```bash
# Windows
py -3 -m pip install -r requirements.txt

# macOS / Linux
python3 -m pip install -r requirements.txt
```

> All dependencies are **version-pinned** for reproducibility.

---

## 🚀 Quick Start (Full Pipeline)

```bash
# 1. Train the model
# Windows
py -3 disease_spread_model.py --data-dir . --output-dir outputs
# macOS / Linux
python3 disease_spread_model.py --data-dir . --output-dir outputs

# 2. Generate predictions
# Windows
py -3 predict_outbreak_risk.py
# macOS / Linux
python3 predict_outbreak_risk.py

# 3. Open the dashboard
# Windows
py -3 -m streamlit run dashboard.py
# macOS / Linux
python3 -m streamlit run dashboard.py
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Covers: data shape validation, Rt estimator correctness, label generation logic, temporal split integrity.

---

## 📊 Dashboard — 6 Tabs

```
http://localhost:8501
```

| Tab | Description |
|---|---|
| **1 · Risk Table** | Filterable table by risk probability. Color-coded risk bands. CSV download. |
| **2 · Hotspot Map** | Global choropleth + Top 10 hotspot sidebar |
| **3 · Spread Animation** | Animated time-lapse choropleth — monthly risk evolution + top spreading countries |
| **4 · Country Trends & Rt** | Per-country cases, deaths, vaccination + live Rt estimator chart + country comparison |
| **5 · 4-Week Forecast** | Iterative 4-week ahead outbreak risk forecast with uncertainty bands |
| **6 · Feature Importance** | Top 30 features, category pie chart, epidemiological interpretation |

**Risk bands:**

| Level | Probability |
|---|---|
| 🟢 Low | < 25% |
| 🟡 Medium | 25–49% |
| 🔴 High | 50–74% |
| 🚨 Very High | ≥ 75% |

---

## 🧠 How It Works

### 1. Data Fusion (`data_fusion.py`)

Merges all datasets into a single feature-rich DataFrame:

- **Base:** `full_data.csv` (OWID — cases, deaths, weekly/biweekly aggregates)
- **Vaccination:** Forward-filled within each country to fix time-series sparsity (NaN reduced from 83% → 11%)
- **Hospitalizations:** Pivoted from long to wide format by indicator type
- **Excess mortality:** P-scores and excess deaths per million as mortality quality signal
- **JHU confirmed cases:** Cross-validated secondary case count source
- **Demographics:** 115+ countries — population density, median age, urban %, hospital beds, GDP per capita, `healthcare_index`, `vulnerability_index`. Missing countries imputed with regional median then global median fallback.

### 2. Feature Engineering

`build_temporal_features()` engineers time-series features:

- **Cyclical signals:** Month and week-of-year
- **Lag features:** 1-week and 2-week lags for every numeric column
- **Moving averages:** 3-week rolling mean for every numeric column
- **Rt estimator:** Effective reproduction number computed per country per week
- **Target label:** `outbreak_risk = 1` if next week's cases grow ≥ 20% vs current week **AND** current week has ≥ 100 cases (floor removes noise from near-zero baselines)

### 3. Rt (Effective Reproduction Number) Estimator

The Rt estimator is a key epidemiological feature and dashboard metric:

```
Rt = (new_cases_t / new_cases_{t-7}) ^ (serial_interval / 7)
```

Using COVID-19 serial interval of **5 days**. A value **> 1** means the epidemic is growing exponentially; **< 1** means it is contracting. This is the most widely used real-time epidemic metric in public health.

### 4. Model Training

- **Algorithm:** Soft-vote ensemble (1:2) — `RandomForestClassifier` (150 trees, depth 10, `class_weight="balanced_subsample"`) + `GradientBoostingClassifier` (200 trees, depth 4, lr=0.05, `sample_weight` = imbalance ratio capped at 60x)
- **Preprocessing:** Median imputation for numerics, mode + one-hot encoding for location; `float64→float32` memory optimisation
- **Split:** Three-way temporal split — train-fit / validation / test (most recent 20% of dates) — threshold chosen on validation only, test set never touched
- **Threshold tuning:** PR curve operating point: `precision ≥ 0.20 AND recall ≥ 0.35`, maximising F1 — prevents precision collapse seen in v1

**Results (v1 → v5):**

| Metric | v1 | v2 | v5 (current) |
|---|---|---|---|
| Label noise | High (1→2 cases = outbreak) | Fixed (100-case floor) | Same |
| Split method | Random (leakage) | Temporal (no leakage) | Same + validation fold |
| Rolling MA | Leaky (roll-then-shift) | Leaky | **Fixed (shift-then-roll, fully causal)** |
| Oversampling | None | Random duplicate | **Removed — GBM sample_weight instead** |
| Model | RF only | RF + GBM (1:1) | **RF (balanced_subsample) + GBM (sample_weight, 1:2 vote)** |
| Threshold strategy | Max F1, no floor | Recall ≥ 0.50 floor | **PR curve: precision ≥ 0.20 AND recall ≥ 0.35** |
| Spurious feature | Present | Present | **Removed (`death_reporting_sparse` excluded)** |
| F1 (outbreak class) | 0.18 | 0.17 | **0.34** |
| Precision (outbreak) | 0.10 | 0.11 | **0.23** |
| Recall (outbreak) | 0.98 | 0.50 | **0.66** |
| PR-AUC | not reported | not reported | **0.276** |
| ROC-AUC | not reported | 0.968 | **0.991** |
| Primary metric | Accuracy (misleading) | Accuracy (misleading) | **PR-AUC (correct for imbalanced data)** |
| Smoke test | No | No | **Yes (load → predict_proba → assert range)** |
| Unit tests | None | None | **19 tests across 5 test classes** |
| Docker | No | No | **Yes (API + dashboard, docker-compose)** |

### Why the F1 was 0.17

The outbreak class is only **0.65% of rows** (152:1 imbalance). Three compounding problems caused the weak F1:

1. **Random oversampling** just copies minority rows — the model memorises them without generalising
2. **GBM has no class weight** — treats all errors equally, learns to always predict "no outbreak"
3. **Threshold 0.66** was too conservative — precision=0.11 means 9 false alarms per real alert

### v3 fixes

- **SMOTE** creates synthetic interpolated minority samples — model generalises to unseen outbreak patterns
- **XGBoost scale_pos_weight=152** — each outbreak misclassification costs 152x a normal week
- **XGBoost eval_metric=aucpr** — optimises PR-AUC directly during training, not log-loss
- **Ensemble weights 1:2** — XGBoost gets double vote since it handles imbalance better
- **min_recall relaxed to 0.40** — threshold can find a better precision/recall trade-off

> **Note on class imbalance:** Actual outbreak weeks are rare events (~1.7% of all weeks after label fix). The ensemble uses `class_weight='balanced_subsample'`, train-set oversampling, and a recall-constrained threshold to balance sensitivity against false-alarm rate — a requirement for any real public health early-warning system.

### 5. New Engineered Features

Two new features added in `data_fusion.py`:

- **`case_acceleration`** — week-over-week change in the case growth ratio. Captures whether spread is speeding up or slowing down, independent of absolute case count. Analogous to the second derivative of the epidemic curve.
- **`death_lag_ratio`** — deaths from 2 weeks ago divided by current cases. Because COVID-19 deaths lag infections by ~2 weeks, a rising ratio signals a surge already in progress even before deaths peak. Epidemiologically grounded in the infection-fatality timeline.

### 6. Feature Importance & Biological Insights

Top features by mean decrease in impurity (country one-hot dummies excluded):

- **3-week moving average of new cases** — strongest predictor, consistent with COVID-19's ~14-day incubation period plus ~1-week reporting lag. Sustained trends are more reliable than single-week spikes.
- **3-week moving average of new deaths** — mortality rises 1–2 weeks after a case surge, confirming the model captures the biological delay between infection and severe outcome.
- **`case_acceleration`** — detects whether a trend is accelerating before absolute numbers become alarming.
- **Rt estimate** — direct mechanistic transmission signal independent of population size.
- **Vaccination coverage** — dampens outbreak risk by reducing the susceptible pool.
- **Vulnerability index** — countries with low hospital capacity and high density have less headroom to absorb surges.

### 7. 4-Week Forecast

An iterative forecast is generated by feeding each week's predicted risk probability back as lag features for the next week, using the trained Random Forest's `predict_proba`. Uncertainty increases with each forecast horizon — weeks 3–4 should be treated as directional guidance.

---

## 🗺️ Datasets Used

| Dataset | Source | Used For |
|---|---|---|
| COVID-19 full data | Our World in Data | Base epidemiological signals |
| JHU confirmed cases | Johns Hopkins CSSE | Secondary case count cross-validation |
| Vaccinations | OWID | Vaccination coverage features |
| Hospitalizations | OWID | Healthcare strain signals |
| Excess mortality | OWID | Mortality signal beyond reported deaths |
| Demographics | World Bank / UN / WHO | Structural vulnerability features |

---

## 🌐 REST API (`api_app.py`)

```bash
# Windows
py -3 -m uvicorn api_app:app --reload --host 127.0.0.1 --port 8000
# macOS / Linux
python3 -m uvicorn api_app:app --reload --host 127.0.0.1 --port 8000
```

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check + high-risk alert count |
| `POST` | `/train` | Train the model |
| `POST` | `/predict` | Generate predictions CSV |
| `GET` | `/predictions` | Paginated predictions |
| `GET` | `/hotspots` | Top-N highest risk countries |
| `GET` | `/alerts` | Countries above risk threshold |
| `GET` | `/trend/{location}` | Historical trend + 8-week risk proxy |
| `GET` | `/locations` | List/search all countries |

Interactive docs: http://127.0.0.1:8000/docs

---

## 🔁 Workflow

```
[Raw CSVs] → data_fusion.py (fuse + ffill vax + impute demographics + causal rolling MA)
                                      ↓
                    build_temporal_features() + compute_rt()
                                      ↓
               [Lag features + Rt + case_acceleration + death_lag_ratio + outbreak_risk label]
                                      ↓
         disease_spread_model.py (train-fit / val / test temporal split)
                    ↓                              ↓
            RF (balanced_subsample)     GBM (sample_weight = imbalance ratio)
                    ↓                              ↓
                    └──── soft-vote ensemble (1:2) ────┘
                                      ↓
                    threshold on validation PR curve
                                      ↓
         outputs/outbreak_risk_model.joblib
         outputs/optimal_threshold.txt  (0.846)
         outputs/feature_importances.csv
         outputs/metrics.txt  (PR-AUC=0.276, ROC-AUC=0.991, F1=0.337)
                                      ↓
                     predict_outbreak_risk.py
                                      ↓
         outputs/latest_outbreak_risk_predictions.csv
                                      ↓
              dashboard.py (6 tabs) / api_app.py (REST + Pydantic validation)
```

---

## ⚠️ Known Limitations

### Precision ceiling at 152:1 class imbalance
The outbreak class represents only ~1.7% of all country-weeks. At this imbalance ratio, achieving precision ≥ 0.50 (1 false alarm per real alert) is extremely difficult without finer-grained features such as mobility data, genomic surveillance, or hospital admission trends at sub-national level. The current model achieves **precision = 0.32** (roughly 2 false alarms per real alert), which is a realistic operating point for an early-warning system where **missing a real outbreak (false negative) is more costly than an unnecessary alert (false positive)**. This is why recall (0.95) is prioritised over precision in the threshold selection strategy.

> This trade-off is intentional and epidemiologically justified. Public health agencies running outbreak surveillance systems typically accept lower precision in exchange for high sensitivity — the cost of a missed outbreak vastly exceeds the cost of an unnecessary investigation.

### Data recency and sparse death reporting (Aug 2024)
The most recent rows in the dataset (around Aug 2024) show `weekly_deaths = 0.0` for the majority of countries. This is a **data coverage artefact**, not a model failure — COVID-19 death reporting became increasingly sparse and inconsistent globally after 2023 as countries wound down mandatory surveillance. Because death-related features (`new_deaths_ma_3`, `death_lag_ratio`) are among the top predictors, this sparsity suppresses predicted risk probabilities for recent rows. As a result, `latest_outbreak_risk_predictions.csv` shows no country exceeding the outbreak threshold — this reflects data absence, not an absence of risk.

**Practical implication:** For real-time deployment, this system should be retrained with a data source that maintains active mortality surveillance (e.g. excess mortality estimates, hospital admissions).

### Imbalance handling approach
The v3 model uses `class_weight='balanced_subsample'` (RF) and `sample_weight` (GBM) rather than synthetic oversampling (SMOTE) or XGBoost. These sklearn-native approaches were chosen for reproducibility and to avoid optional-dependency failure modes in evaluation environments. If `xgboost` and `imbalanced-learn` are available, re-enabling the SMOTE + XGBoost path (see commented flags in the training script) may improve PR-AUC further.

---

## 🐛 Troubleshooting

| Problem | Fix |
|---|---|
| `full_data.csv not found` | Ensure all CSVs are in the project root |
| Dashboard shows "No predictions found" | Run `predict_outbreak_risk.py` first |
| `outbreak_risk_model.joblib not found` | Run the training script first |
| Feature importances tab is empty | Re-run training script (auto-generates the CSV) |
| 4-week forecast button does nothing | Model must be loaded — check outputs/ folder |
| Port 8000 in use | Add `--port 8001` |
| Port 8501 in use | Add `--server.port 8502` |
| All predicted risks are 0 / probabilities low | Expected for Aug 2024 data — death reporting is sparse post-2023. See Known Limitations. |

---

## v4 — Gaps fixed

| Gap | Fix |
|---|---|
| Forecast lag-shift fragile | Replaced string suffix matching with explicit suffix constants + set-based lookup |
| No API input validation | Added Pydantic `BaseModel` schemas with `Field` constraints on all query endpoints |
| No Docker / deployment config | Added `Dockerfile` + `docker-compose.yml` (API + dashboard, shared outputs volume) |
| Lag window capped at t-2 | Added `lag_4` (monthly) and `ma_6` (6-week) features to `build_temporal_features()` |
| Death data sparsity post-2023 | Forward-fill (limit=4 weeks) for zero deaths per country + `death_reporting_sparse` flag |

### Docker quick start

```bash
# Build and run both API + dashboard
docker compose up --build

# API: http://localhost:8000/docs
# Dashboard: http://localhost:8501
```
