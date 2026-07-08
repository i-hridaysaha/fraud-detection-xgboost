"""
train.py — end-to-end training pipeline:
  1. Load data, time-based split
  2. Feature engineering (leakage-safe, fit on train only)
  3. Handle imbalance (class weights and/or SMOTE)
  4. Train LR baseline -> RF -> XGBoost (final model)
  5. Tune decision threshold on validation set
  6. Evaluate all models on held-out test set
  7. Generate SHAP explainability report
  8. Persist model artifacts for serving (see src/api.py)

Run:
    python train.py --data data/transactions.csv --imbalance_strategy class_weight
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd

from src.config import DATA, PATHS, IMBALANCE
from src.data_loader import load_raw, time_based_split
from src.feature_engineering import build_feature_pipeline, get_feature_columns
from src.imbalance import get_class_weight_ratio, apply_smote, tune_threshold_for_f1
from src.models import build_logistic_regression, build_random_forest, build_xgboost
from src.evaluate import evaluate_model, compare_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=DATA.raw_path)
    parser.add_argument("--imbalance_strategy", type=str, default="class_weight",
                         choices=["class_weight", "smote", "none"])
    args = parser.parse_args()

    os.makedirs(PATHS.model_dir, exist_ok=True)
    os.makedirs(PATHS.reports_dir, exist_ok=True)

    # ---- 1. Load & split ----
    print("Loading data...")
    df = load_raw(args.data)
    train_df, val_df, test_df = time_based_split(df)

    # ---- 2. Feature engineering ----
    print("\nEngineering features (leakage-safe, fit on train only)...")
    train_df, val_df, test_df, risk_maps, device_map = build_feature_pipeline(train_df, val_df, test_df)
    feature_cols = get_feature_columns()
    joblib.dump(feature_cols, PATHS.feature_list_path)

    X_train, y_train = train_df[feature_cols], train_df[DATA.target_col]
    X_val, y_val = val_df[feature_cols], val_df[DATA.target_col]
    X_test, y_test = test_df[feature_cols], test_df[DATA.target_col]

    # ---- 3. Imbalance handling ----
    print(f"\nHandling imbalance with strategy: {args.imbalance_strategy}")
    scale_pos_weight = get_class_weight_ratio(y_train)
    print(f"Class imbalance ratio (neg:pos): {scale_pos_weight:.1f} : 1")

    X_train_model, y_train_model = X_train, y_train
    sk_class_weight = None
    xgb_scale_pos_weight = 1.0

    if args.imbalance_strategy == "class_weight":
        sk_class_weight = "balanced"
        xgb_scale_pos_weight = scale_pos_weight
    elif args.imbalance_strategy == "smote":
        X_train_model, y_train_model = apply_smote(X_train, y_train)
        # after SMOTE has already balanced the data, don't ALSO reweight
        sk_class_weight = None
        xgb_scale_pos_weight = 1.0
    # "none": no reweighting, no resampling — useful as a "what if we ignored imbalance" baseline

    # ---- 4. Train models ----
    results = []

    print("\nTraining Logistic Regression baseline...")
    lr = build_logistic_regression(class_weight=sk_class_weight)
    lr.fit(X_train_model, y_train_model)
    lr_val_proba = lr.predict_proba(X_val)[:, 1]
    joblib.dump(lr, PATHS.lr_model_path)

    print("Training Random Forest...")
    rf = build_random_forest(class_weight=sk_class_weight)
    rf.fit(X_train_model, y_train_model)
    rf_val_proba = rf.predict_proba(X_val)[:, 1]
    joblib.dump(rf, PATHS.rf_model_path)

    print("Training XGBoost (final model)...")
    xgb = build_xgboost(scale_pos_weight=xgb_scale_pos_weight)
    xgb.fit(
        X_train_model, y_train_model,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    xgb_val_proba = xgb.predict_proba(X_val)[:, 1]
    joblib.dump(xgb, PATHS.xgb_model_path)

    # ---- 5. Threshold tuning (on validation set, for the FINAL model only) ----
    print("\nTuning decision threshold on validation set (XGBoost)...")
    threshold_info = tune_threshold_for_f1(y_val, xgb_val_proba)
    print(threshold_info)
    joblib.dump(threshold_info, PATHS.threshold_path)
    best_threshold = threshold_info["best_threshold"]

    # ---- 6. Evaluate all models on TEST set (never used until now) ----
    print("\nEvaluating on held-out test set...")
    lr_test_proba = lr.predict_proba(X_test)[:, 1]
    rf_test_proba = rf.predict_proba(X_test)[:, 1]
    xgb_test_proba = xgb.predict_proba(X_test)[:, 1]

    results.append(evaluate_model(y_test, lr_test_proba, threshold=0.5, model_name="Logistic Regression"))
    results.append(evaluate_model(y_test, rf_test_proba, threshold=0.5, model_name="Random Forest"))
    results.append(evaluate_model(y_test, xgb_test_proba, threshold=best_threshold, model_name="XGBoost (tuned threshold)"))
    results.append(evaluate_model(y_test, xgb_test_proba, threshold=0.5, model_name="XGBoost (default 0.5 threshold)"))

    compare_models(results)

    # ---- 7. Explainability (SHAP) ----
    print("\nGenerating SHAP explainability report...")
    from src.explainability import compute_shap_values, save_global_summary_plot, get_top_global_features

    shap_sample = X_test.sample(min(2000, len(X_test)), random_state=42)
    explainer, shap_values = compute_shap_values(xgb, shap_sample)
    save_global_summary_plot(explainer, shap_values, shap_sample)
    top_features = get_top_global_features(explainer, shap_values, shap_sample)
    print("\nTop global features by mean |SHAP value|:")
    print(top_features.to_string(index=False))
    top_features.to_csv("reports/shap_top_features.csv", index=False)

    print("\nDone. Artifacts saved to models/ and reports/.")


if __name__ == "__main__":
    main()
