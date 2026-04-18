"""
anomaly_detector.py
-------------------
Reusable wrappers around the anomaly detection logic in
notebooks/04_anomaly_detection.ipynb.
"""
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_isolation_forest(contamination: float = 0.02) -> IsolationForest:
    return IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )


def build_rf_pipeline() -> Pipeline:
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        ))
    ])
