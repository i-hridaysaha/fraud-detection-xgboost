"""
config.py — central configuration for the fraud detection pipeline.
Keeping these in one place makes the pipeline reproducible and makes it
obvious what to tune when adapting to a new data source.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    raw_path: str = "data/transactions.csv"
    train_frac: float = 0.7
    val_frac: float = 0.15
    test_frac: float = 0.15  # inferred as remainder; kept for clarity
    time_col: str = "timestamp"
    target_col: str = "Class"
    random_state: int = 42


@dataclass
class FeatureConfig:
    velocity_windows_minutes: List[int] = field(default_factory=lambda: [10, 60, 1440])  # 10m, 1h, 24h
    pca_cols: List[str] = field(default_factory=lambda: [f"V{i}" for i in range(1, 29)])
    amount_col: str = "Amount"
    customer_col: str = "customer_id"
    merchant_col: str = "merchant_id"
    merchant_category_col: str = "merchant_category"
    device_col: str = "device_id"


@dataclass
class ImbalanceConfig:
    strategy: str = "class_weight"  # one of: "class_weight", "smote", "none"
    smote_target_ratio: float = 0.10  # SMOTE brings minority class up to 10% of majority


@dataclass
class ModelConfig:
    xgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
    })
    rf_params: dict = field(default_factory=lambda: {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 5,
        "n_jobs": -1,
        "random_state": 42,
    })
    lr_params: dict = field(default_factory=lambda: {
        "max_iter": 1000,
        "random_state": 42,
    })


@dataclass
class PathsConfig:
    model_dir: str = "models"
    reports_dir: str = "reports"
    feature_list_path: str = "models/feature_list.joblib"
    merchant_risk_map_path: str = "models/merchant_risk_map.joblib"
    device_map_path: str = "models/device_map.joblib"
    xgb_model_path: str = "models/xgb_fraud_model.joblib"
    rf_model_path: str = "models/rf_fraud_model.joblib"
    lr_model_path: str = "models/lr_fraud_model.joblib"
    threshold_path: str = "models/decision_threshold.joblib"


DATA = DataConfig()
FEATURES = FeatureConfig()
IMBALANCE = ImbalanceConfig()
MODEL = ModelConfig()
PATHS = PathsConfig()
