# Understanding the Hybrid Model (Markov + Gradient Boosting Machine)

## Overview

The hybrid model combines **spatial dynamics** (Markov Chain) with **temporal patterns** (Gradient Boosting) to predict COVID-19 case numbers in Brussels municipalities. This approach leverages both geographic relationships between municipalities and temporal features like day of week, month, and weekend indicators.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA PREPARATION                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │ Raw COVID    │ →  │ Cleaned Data │ →  │ Smoothed     │ │
│  │ Data (API)   │    │ (Brussels)   │    │ (Savitzky-   │ │
│  │              │    │              │    │  Golay)      │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              MARKOV TRANSITION MATRIX                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. Estimate base matrix (Least Squares)              │  │
│  │ 2. Apply geographic constraints (adjacency, distance)│  │
│  │ 3. Optimize via Expectation-Maximization (EM)       │  │
│  │ 4. Combine LS + EM with optimal weights             │  │
│  └──────────────────────────────────────────────────────┘  │
│  Result: P[i,j] = probability of case flow from j → i      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              MARKOV PREDICTION (Spatial)                    │
│                                                              │
│  For each date t:                                           │
│    X[t+1] = P @ X[t]                                        │
│                                                              │
│  Where:                                                      │
│    - X[t] = vector of cases per municipality at time t      │
│    - P = transition matrix (19×19 for 19 municipalities)    │
│    - Each row of P sums to 1 (stochastic matrix)            │
│                                                              │
│  This captures:                                             │
│    • Spatial diffusion between adjacent municipalities      │
│    • Geographic proximity effects                           │
│    • Historical transmission patterns                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING                             │
│                                                              │
│  From Markov predictions, extract:                          │
│  ┌────────────────────────────────────────────────────┐   │
│  │ 1. predict_markov[i] = Markov prediction for        │   │
│  │                       municipality i                 │   │
│  │ 2. total_markov = Σ predict_markov (all communes)  │   │
│  │ 3. moyenne_markov = mean(predict_markov)           │   │
│  └────────────────────────────────────────────────────┘   │
│                                                              │
│  From date, extract temporal features:                      │
│  ┌────────────────────────────────────────────────────┐   │
│  │ 4. jour_semaine = day of week (0=Monday, 6=Sunday)│   │
│  │ 5. est_weekend = 1 if weekend, 0 otherwise        │   │
│  │ 6. mois = month (1-12)                             │   │
│  └────────────────────────────────────────────────────┘   │
│                                                              │
│  Normalize Markov features using StandardScaler:           │
│    - predict_markov_mise_a_l_echelle                        │
│    - total_markov_mis_a_l_echelle                           │
│    - moyenne_markov_mise_a_l_echelle                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│         GRADIENT BOOSTING MACHINE (Temporal + Spatial)      │
│                                                              │
│  Input Features (6 features):                               │
│    [predict_markov_scaled, total_markov_scaled,            │
│     moyenne_markov_scaled, jour_semaine,                    │
│     est_weekend, mois]                                      │
│                                                              │
│  Target: log1p(cas_reel) = log(1 + actual_cases)           │
│                                                              │
│  Model: GradientBoostingRegressor                           │
│    - Optimized via Bayesian Search (BayesSearchCV)         │
│    - Hyperparameters: n_estimators, max_depth,              │
│                      min_samples_split                      │
│                                                              │
│  Output: log_prediction                                     │
│  Final: expm1(log_prediction) = final_case_prediction      │
└─────────────────────────────────────────────────────────────┘
```

---

## Detailed Component Explanation

### 1. Markov Chain Model (Spatial Dynamics)

#### What is a Markov Transition Matrix?

A **Markov transition matrix** `P` is a square matrix where:
- **P[i,j]** = probability that a case in municipality `j` at time `t` will be in municipality `i` at time `t+1`
- Each **row sums to 1** (stochastic property)
- The matrix captures **spatial relationships** between municipalities

#### How is the Matrix Estimated?

The project uses **two methods** and combines them:

**A. Least Squares (LS) Method:**
```python
# Mathematical formula:
P = X_t1 @ X_t.T @ inv(X_t @ X_t.T + λ*I)

Where:
- X_t = cases at time t (19 municipalities × n_dates)
- X_t1 = cases at time t+1 (aligned with X_t)
- λ = regularization parameter (1e-6) to avoid singular matrices
```

**B. Expectation-Maximization (EM) Method:**
- Iterative algorithm that alternates between:
  - **E-step**: Estimate expected case flows between municipalities
  - **M-step**: Update transition probabilities
- Includes **geographic constraints** (adjacency, distance) via `alpha_geo` parameter
- Converges when matrix changes become minimal

**C. Geographic Regularization:**
```python
# Apply geographic weights
P_geo = P_base * geographic_weights_matrix

# Combine with base matrix
P_final = (1 - alpha_geo) * P_base + alpha_geo * P_geo
```

**D. Model Selection:**
- Tests **400+ combinations** of LS/EM weights (0% to 100%)
- Selects the combination with **lowest MAE** (Mean Absolute Error)
- Best model saved as `best_combination_model.json`

#### Markov Prediction Process

For each day:
```python
# Current state: cases per municipality at time t
X_t = [cases_Anderlecht, cases_Bruxelles, ..., cases_Woluwe]

# Apply transition matrix
X_t1_predicted = P @ X_t

# Result: predicted cases for each municipality at t+1
```

**Key Properties:**
- **Spatial diffusion**: Cases spread to adjacent municipalities
- **Conservation**: Total cases can be preserved (normalization step)
- **Geographic awareness**: Adjacent municipalities have higher transition probabilities

---

### 2. Feature Engineering

The hybrid model creates **6 features** from Markov predictions and temporal information:

#### Markov Features (3 features):

1. **`predict_markov`**: Markov prediction for the specific municipality
   - Direct output from `P @ X_t` for municipality `i`
   - Captures spatial dynamics for that municipality

2. **`total_markov`**: Sum of all Markov predictions across all municipalities
   - `total_markov = Σ predict_markov[i]` for all i
   - Represents **overall epidemic intensity** in the region

3. **`moyenne_markov`**: Average of Markov predictions
   - `moyenne_markov = mean(predict_markov)`
   - Represents **average expected cases** per municipality

#### Temporal Features (3 features):

4. **`jour_semaine`**: Day of week (0=Monday, 6=Sunday)
   - Captures weekly patterns (e.g., lower reporting on weekends)

5. **`est_weekend`**: Binary indicator (1=weekend, 0=weekday)
   - Captures weekend effects (testing patterns, mobility)

6. **`mois`**: Month (1-12)
   - Captures seasonal patterns (e.g., winter peaks)

#### Normalization:

All Markov features are **standardized** using `StandardScaler`:
```python
scaler_predict.fit(predict_markov)
predict_markov_scaled = scaler_predict.transform(predict_markov)
```

This ensures:
- Features are on similar scales
- Gradient Boosting can learn effectively
- Predictions can be denormalized later

---

### 3. Gradient Boosting Machine (GBM)

#### Why Gradient Boosting?

**Gradient Boosting** is ideal for this problem because:
- Handles **non-linear relationships** between features
- Can learn **complex interactions** (e.g., "weekend + high total_markov → lower cases")
- Robust to **feature scaling** (after normalization)
- Provides **feature importance** insights

#### Training Process:

1. **Target Transformation**: `y = log1p(cas_reel)`
   - Uses `log1p` to handle zero cases and stabilize variance
   - Prevents negative predictions

2. **Hyperparameter Optimization**:
   ```python
   BayesSearchCV(
       GradientBoostingRegressor(),
       {
           'n_estimators': (50, 500),
           'max_depth': (2, 10),
           'min_samples_split': (2, 10)
       },
       n_iter=40,
       scoring='neg_mean_squared_error',
       cv=3
   )
   ```
   - Tests 40 combinations via Bayesian optimization
   - Uses 3-fold cross-validation
   - Selects best model based on negative MSE

3. **Training**:
   - Input: 6 normalized features
   - Output: `log_prediction`
   - Loss: Mean Squared Error on log-transformed target

#### Prediction Process:

```python
# 1. Get Markov prediction
markov_pred = P @ X_t

# 2. Extract features
features = [
    scaler_predict.transform(markov_pred[i]),      # Scaled Markov for municipality i
    scaler_total.transform(sum(markov_pred)),      # Scaled total
    scaler_moyenne.transform(mean(markov_pred)),  # Scaled average
    jour_semaine,                                  # Day of week
    est_weekend,                                    # Weekend indicator
    mois                                           # Month
]

# 3. Predict with GBM
log_prediction = gbm_model.predict(features)

# 4. Transform back to original scale
final_prediction = expm1(log_prediction)  # exp(log_prediction) - 1
```

---

## Why This Hybrid Approach Works

### 1. **Spatial Dynamics (Markov)**
- Captures **geographic diffusion** of cases
- Models **adjacency effects** (cases spread to neighboring municipalities)
- Incorporates **historical spatial patterns**

### 2. **Temporal Patterns (GBM)**
- Captures **weekly cycles** (weekend effects, reporting patterns)
- Models **seasonal trends** (winter peaks, summer lows)
- Learns **non-linear interactions** (e.g., "high total + weekend → different pattern")

### 3. **Synergy**
- Markov provides **spatial context** that pure temporal models miss
- GBM **refines** Markov predictions using temporal knowledge
- **Aggregated features** (total, moyenne) provide regional context

### 4. **Robustness**
- Markov handles **sparse data** (smooth spatial transitions)
- GBM handles **complex patterns** (non-linear relationships)
- **Normalization** ensures stable training

---

## Role in Predicting Epidemic Dynamics

### What the Model Captures:

1. **Spatial Propagation**:
   - How cases **spread geographically** between municipalities
   - Which municipalities are **transmission hubs**
   - **Adjacency effects** (cases in neighboring areas)

2. **Temporal Evolution**:
   - **Weekly patterns** (testing schedules, mobility)
   - **Seasonal trends** (winter peaks, summer declines)
   - **Day-of-week effects** (weekend reporting delays)

3. **Regional Context**:
   - **Total epidemic intensity** (overall case load)
   - **Average case levels** (baseline expectations)
   - **Municipality-specific patterns** (local dynamics)

### Prediction Workflow:

```
Day t (Current State):
├─ Current cases per municipality: X_t
├─ Apply Markov: X_t1_markov = P @ X_t
└─ Extract features:
   ├─ predict_markov[i] (for municipality i)
   ├─ total_markov = Σ X_t1_markov
   ├─ moyenne_markov = mean(X_t1_markov)
   ├─ jour_semaine, est_weekend, mois
   └─ Normalize Markov features

Day t+1 (Prediction):
├─ GBM predicts: log_prediction = GBM(features)
└─ Final prediction: cases = expm1(log_prediction)
```

### Advantages Over Pure Models:

**vs. Pure Markov:**
- ✅ Captures temporal patterns (weekends, seasons)
- ✅ Handles non-linear relationships
- ✅ Better accuracy on complex scenarios

**vs. Pure Temporal Models:**
- ✅ Incorporates spatial relationships
- ✅ Models geographic diffusion
- ✅ Better for multi-municipality predictions

**vs. Pure Machine Learning:**
- ✅ Interpretable spatial component
- ✅ Physically meaningful (transition probabilities)
- ✅ Better generalization (Markov captures domain knowledge)

---

## Key Files in the Implementation

1. **`markov_models.py`**: Markov matrix estimation (LS, EM, geographic constraints)
2. **`evaluate_and_select_model.py`**: Model selection and optimization
3. **`utils_preprocessing.py`**: Feature engineering (`enrichir_le_dataset`)
4. **`08_markov_sklearn_integration.py`**: GBM training and integration
5. **`09_interroger_modele_de_prediction_integre.py`**: Prediction interface
6. **`07_previsions_selon_le_modele_markov.py`**: Pure Markov predictions (for comparison)
7. **`11_gbm_proximity_distribution.py`** *(repository root)*: GBM predictions **re-spread** across municipalities using a **proximity matrix** (Markov × geographic weights); writes `Data/gbm_proximity_distribution.json` and `visualizations_gbm_proximity/`.
8. **`12_analyse_corridors_epidemiques.py`** *(repository root)*: **Epidemic corridor** analysis from those distribution matrices—temporal **persistence**, bifurcation-style metrics (**β**, **μ**), JSON summaries, and Plotly outputs under `visualizations_corridors/` (see also `Pipeline_Mesoscopique_Explicatif.md`).

---

## Mesoscopic extension: GBM redistribution and epidemic corridors

After the hybrid model produces a scalar GBM forecast per municipality and date, the **mesoscopic pipeline** asks: *how would that mass split across neighbors if spatial coupling followed the learned Markov structure and geographic proximity?*

1. **`11_gbm_proximity_distribution.py`** implements that **spatial redistribution**, producing time-indexed **distribution matrices** suitable for downstream interpretation.
2. **`12_analyse_corridors_epidemiques.py`** scans those matrices for **persistent high-intensity directed links** (“corridors”), aggregates **metrics**, and exports structured JSON plus **Plotly** dashboards.

This is **interpretability and spatial dynamics research** layered on top of the hybrid predictor—not a replacement for the Markov+GBM forecast itself. Dependencies: **plotly**, optional **kaleido** for static image export (see `requirements.txt`).

---

## Example: Predicting Cases for "Bruxelles" on 2023-04-05

```python
# Step 1: Get current state (2023-04-04)
X_t = [10, 25, 5, ..., 8]  # Cases per municipality on 2023-04-04

# Step 2: Apply Markov transition
X_t1_markov = P @ X_t
# Result: [12.3, 28.1, 6.2, ..., 9.5]
# For "Bruxelles" (index 3): predict_markov = 28.1

# Step 3: Extract features
total_markov = 245.8  # Sum of all predictions
moyenne_markov = 12.9  # Average prediction
jour_semaine = 2  # Wednesday
est_weekend = 0  # Not weekend
mois = 4  # April

# Step 4: Normalize Markov features
predict_markov_scaled = scaler_predict.transform([[28.1]])  # e.g., 1.23
total_markov_scaled = scaler_total.transform([[245.8]])     # e.g., 0.87
moyenne_markov_scaled = scaler_moyenne.transform([[12.9]])  # e.g., 0.45

# Step 5: GBM prediction
features = [[1.23, 0.87, 0.45, 2, 0, 4]]
log_prediction = gbm_model.predict(features)  # e.g., 3.15

# Step 6: Transform back
final_prediction = expm1(3.15)  # e.g., 22.3 cases
```

**Interpretation:**
- Markov predicted **28.1 cases** based on spatial dynamics
- GBM refined this to **22.3 cases** considering:
  - It's a Wednesday (normal reporting)
  - Not a weekend (no reporting delays)
  - April (moderate season)
  - Total regional intensity (245.8 cases)

---

## Summary

The hybrid model combines:
- **Markov Chain**: Spatial dynamics, geographic diffusion, adjacency effects
- **Gradient Boosting**: Temporal patterns, non-linear relationships, regional context

**Result**: A robust prediction system that captures both **where** cases will appear (spatial) and **when** they will peak (temporal), making it highly effective for epidemic forecasting in a multi-municipality setting.

**Current codebase progress:** the same spatial signal (Markov + geography) now also drives **post-hoc redistribution** of GBM outputs and **corridor** detection (`11_*` / `12_*` at repo root), documented in French in `readme.md` and `Pipeline_Mesoscopique_Explicatif.md`.

