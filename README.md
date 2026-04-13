# ML Data Pipeline Utilities

> Production-grade data preprocessing and validation framework for ML pipelines — handling the 80% of ML engineering that isn't modelling: missing data, schema drift, feature encoding, scaling, and automated quality gates.

[![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas_2.2-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Great Expectations](https://img.shields.io/badge/Great_Expectations-FF6B35?style=flat)](https://greatexpectations.io)

---

## The Problem

Bad data doesn't announce itself. In production ML pipelines for enterprise systems — financial market data, government payments, retail transactions — data quality failures manifest as:

- Silent model degradation (accuracy drops without any error)
- Training/serving skew (preprocessing applied differently at train vs inference time)
- Downstream regulatory risk (incorrect features in compliance models)
- Cascading failures across dependent pipelines

This utility addresses all four with a principled, testable preprocessing framework.

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    INPUT DATA                              │
│  Raw DataFrame (CSV, Parquet, DB query, streaming batch)   │
└───────────────────────────┬────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│                 SCHEMA VALIDATION LAYER                    │
│  Expected columns · Data types · Value ranges · Nullability │
│  → Fail fast: catch issues before expensive transforms     │
└───────────────────────────┬────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│               MISSING VALUE HANDLER                        │
│                                                            │
│  Numeric  → Median imputation (robust to outliers)         │
│  Categoric→ Mode imputation or "Unknown" sentinel          │
│  Time     → Forward fill with staleness threshold          │
│  High null→ Drop column if >threshold% missing             │
└───────────────────────────┬────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│                FEATURE ENCODING                            │
│                                                            │
│  Low cardinality  → OneHotEncoder (drop='first')           │
│  High cardinality → TargetEncoder / OrdinalEncoder         │
│  Ordinal          → OrdinalEncoder with explicit ordering  │
│  Binary           → Direct 0/1 mapping                     │
└───────────────────────────┬────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│                 FEATURE SCALING                            │
│                                                            │
│  StandardScaler  → Neural nets, SVMs, gradient descent     │
│  MinMaxScaler    → Tree models, image pixel values         │
│  RobustScaler    → Datasets with significant outliers      │
│                    (financial returns, sensor anomalies)   │
└───────────────────────────┬────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│              POST-PROCESSING VALIDATION                    │
│  Schema check · Distribution drift · NaN/inf scan          │
│  → Emit quality report + block pipeline on violation       │
└────────────────────────────────────────────────────────────┘
```

---

## Core Design Principles

### 1. Fit on Train, Transform on Test — Always

```python
class MLPreprocessor:
    """
    Stateful preprocessor — fit parameters learned only from training data.
    Prevents data leakage that inflates validation metrics in production.

    Anti-pattern (leaky):
        scaler.fit_transform(X_full)  # Test statistics contaminate train

    Correct pattern:
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled  = scaler.transform(X_test)   # Uses train stats only
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "MLPreprocessor":
        self._fit_imputers(X)
        self._fit_encoders(X)
        self._fit_scalers(X)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Call fit() on training data before transform()")
        X = self._apply_imputers(X)
        X = self._apply_encoders(X)
        return self._apply_scalers(X)
```

### 2. Schema Validation Before Processing

```python
def validate_schema(df: pd.DataFrame, schema: DataSchema) -> ValidationReport:
    """
    Fail fast with actionable errors rather than silent corruption downstream.
    Inspired by Great Expectations — applied at pipeline entry point.
    """
    violations = []

    for col, expected_type in schema.column_types.items():
        if col not in df.columns:
            violations.append(f"Missing required column: '{col}'")
        elif not pd.api.types.is_dtype_equal(df[col].dtype, expected_type):
            violations.append(
                f"Type mismatch on '{col}': expected {expected_type}, got {df[col].dtype}"
            )

    for col, (min_val, max_val) in schema.value_ranges.items():
        out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
        if out_of_range > 0:
            violations.append(f"'{col}': {out_of_range} values outside [{min_val}, {max_val}]")

    return ValidationReport(passed=len(violations) == 0, violations=violations)
```

### 3. Imputation Strategy Matrix

| Data Type | Distribution | Recommended Strategy | Rationale |
|---|---|---|---|
| Numeric | Normal | Mean | Preserves distribution mean |
| Numeric | Skewed / outliers | **Median** | Robust to extreme values |
| Numeric | Time series | Forward fill + cap | Preserves temporal continuity |
| Categorical | Low cardinality | Mode | Most frequent class |
| Categorical | High cardinality | "Unknown" sentinel | Avoids frequency bias |
| Boolean | Any | Mode | Binary — most common value |

---

## Usage

```python
from data_prep import MLPreprocessor, DataSchema

schema = DataSchema(
    column_types={"age": "float64", "income": "float64", "category": "object"},
    value_ranges={"age": (0, 120), "income": (0, 10_000_000)},
    nullable_columns=["income"],
    required_columns=["age", "category"],
)

preprocessor = MLPreprocessor(
    numeric_strategy="median",
    categorical_strategy="onehot",
    scaling="standard",
    schema=schema,
)

# Fit on training data only
X_train_processed = preprocessor.fit_transform(X_train, y_train)

# Apply same transforms to test — no refitting
X_test_processed = preprocessor.transform(X_test)

# Persist for serving
preprocessor.save("models/preprocessor_v1.pkl")
```

---

## Production Deployment Pattern

```python
# Inference service — load pre-fitted preprocessor
preprocessor = MLPreprocessor.load("models/preprocessor_v1.pkl")

@app.post("/predict")
def predict(features: InputFeatures) -> PredictionResponse:
    df = pd.DataFrame([features.dict()])

    # Validate before processing
    report = preprocessor.validate(df)
    if not report.passed:
        raise HTTPException(400, detail=report.violations)

    X = preprocessor.transform(df)
    prediction = model.predict(X)
    return PredictionResponse(prediction=prediction[0])
```

---

## Related Work

- [ml-neural-network-projects](https://github.com/codebygarrysingh/ml-neural-network-projects) — downstream consumer of this preprocessing pipeline
- [real-time-data-platform](https://github.com/codebygarrysingh/real-time-data-platform) — enterprise-scale data quality with Great Expectations
- [production-rag-pipeline](https://github.com/codebygarrysingh/production-rag-pipeline) — document preprocessing and embedding pipelines

---

## Author

**Garry Singh** — Principal AI & Data Engineer · MSc Oxford

[Portfolio](https://garrysingh.dev) · [LinkedIn](https://linkedin.com/in/singhgarry) · [Book a Consultation](https://calendly.com/garry-singh2902)
