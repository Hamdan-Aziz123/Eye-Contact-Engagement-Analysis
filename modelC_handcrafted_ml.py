#!/usr/bin/env python3
"""
modelC_handcrafted_ml.py

Hand-crafted features + Traditional ML pipeline for Eye Contact & Engagement Analysis
Compatible with your dataset layout (train/val/test CSVs under processed_dir).
Saves model, test_report.json, classification_report.txt, confusion_matrix.png,
feature_importances.png, training_history.csv (if available).

Usage example (Colab):
!python modelC_handcrafted_ml.py \
    --processed_dir /content/project/data/processed \
    --output_dir /content/drive/MyDrive/ResNet_Project/outputs_modelC \
    --thr_att 10 --thr_dist 25 --seed 42
"""

import os
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Try to import CatBoost, XGBoost
USE_CATBOOST = False
USE_XGBOOST = False
try:
    from catboost import CatBoostClassifier, Pool
    USE_CATBOOST = True
except Exception:
    try:
        import xgboost as xgb
        USE_XGBOOST = True
    except Exception:
        USE_XGBOOST = False

# -------------------------
# Utility / Feature helpers
# -------------------------
def map_angles_to_label(pitch: float, vert: float, horiz: float, thr_att: float = 10.0, thr_dist: float = 25.0) -> int:
    """Same mapping as ViT_train.py so results are comparable."""
    mag = float(max(abs(float(pitch)), abs(float(vert)), abs(float(horiz))))
    if mag <= thr_att:
        return 0
    elif mag <= thr_dist:
        return 1
    else:
        return 2

def load_csv_safe(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)

def build_feature_dataframe(df, thr_att=10.0, thr_dist=25.0, verbose=False):
    """
    From input dataframe (with columns pitch, vert, horiz, distance, mean_brightness, std_brightness)
    produce a feature DataFrame and label column.
    """
    # Keep a copy to avoid changing original
    dfc = df.copy()

    # If 'distance' contains 'm' like '2m', convert to numeric
    if 'distance' in dfc.columns:
        dfc['distance_num'] = dfc['distance'].astype(str).str.replace(r'[^\d.]', '', regex=True)
        dfc['distance_num'] = pd.to_numeric(dfc['distance_num'], errors='coerce').fillna(-1).astype(int)
    else:
        dfc['distance_num'] = -1

    # Ensure numeric columns exist
    for c in ['pitch', 'vert', 'horiz', 'mean_brightness', 'std_brightness']:
        if c not in dfc.columns:
            dfc[c] = 0.0

    # Basic features
    features = pd.DataFrame({
        'distance': dfc['distance_num'],
        'pitch': pd.to_numeric(dfc['pitch'], errors='coerce').fillna(0.0),
        'vert': pd.to_numeric(dfc['vert'], errors='coerce').fillna(0.0),
        'horiz': pd.to_numeric(dfc['horiz'], errors='coerce').fillna(0.0),
        'mean_brightness': pd.to_numeric(dfc['mean_brightness'], errors='coerce').fillna(0.0),
        'std_brightness': pd.to_numeric(dfc['std_brightness'], errors='coerce').fillna(0.0),
    })

    # Derived features
    features['abs_pitch'] = features['pitch'].abs()
    features['abs_vert'] = features['vert'].abs()
    features['abs_horiz'] = features['horiz'].abs()
    features['mag'] = features[['abs_pitch', 'abs_vert', 'abs_horiz']].max(axis=1)  # same as used for labeling
    features['pitch_horiz'] = features['pitch'] * features['horiz']
    features['pitch_sq'] = features['pitch'] ** 2
    features['vert_sq'] = features['vert'] ** 2
    features['horiz_sq'] = features['horiz'] ** 2

    # Label (using mapping function so consistent with ViT)
    labels = dfc.apply(lambda r: map_angles_to_label(r.get('pitch', 0.0), r.get('vert', 0.0), r.get('horiz', 0.0),
                                                     thr_att=thr_att, thr_dist=thr_dist), axis=1)

    if verbose:
        print("Built features with columns:", features.columns.tolist())
        print("Label distribution:", labels.value_counts().to_dict())

    return features, labels

# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Model C: Hand-crafted features + ML (CatBoost/XGBoost/RandomForest)")
    p.add_argument("--processed_dir", type=str, default="/content/project/data/processed", help="Processed data dir (contains train_labels.csv etc.)")
    p.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/ResNet_Project/outputs_modelC", help="Where to save model & reports")
    p.add_argument("--thr_att", type=float, default=10.0, help="Attentive threshold (degrees)")
    p.add_argument("--thr_dist", type=float, default=25.0, help="Distracted threshold (degrees)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--catboost_iters", type=int, default=1000)
    p.add_argument("--test_size", type=float, default=0.0, help="If >0 and val csv missing, create test split from train (not recommended).")
    return p.parse_args()

def main():
    args = parse_args()
    np.random.seed(args.seed)

    processed_dir = os.path.abspath(args.processed_dir)
    out_dir = os.path.abspath(args.output_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    train_csv = os.path.join(processed_dir, "train_labels.csv")
    val_csv = os.path.join(processed_dir, "val_labels.csv")
    test_csv = os.path.join(processed_dir, "test_labels.csv")

    print("Loading CSVs from:", processed_dir)
    train_df = load_csv_safe(train_csv)
    val_df = load_csv_safe(val_csv)
    test_df = load_csv_safe(test_csv)
    print(f"Train rows: {len(train_df)}, Val rows: {len(val_df)}, Test rows: {len(test_df)}")

    # build features and labels for each split
    X_train, y_train = build_feature_dataframe(train_df, thr_att=args.thr_att, thr_dist=args.thr_dist, verbose=True)
    X_val, y_val = build_feature_dataframe(val_df, thr_att=args.thr_att, thr_dist=args.thr_dist)
    X_test, y_test = build_feature_dataframe(test_df, thr_att=args.thr_att, thr_dist=args.thr_dist)

    # Some safety checks
    if X_train.shape[0] == 0:
        raise RuntimeError("No training samples found after feature extraction. Check your CSVs and processed_dir.")
    # Impute missing values (median)
    imputer = SimpleImputer(strategy='median')
    imputer.fit(X_train)
    X_train_imputed = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns)
    X_val_imputed = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # Which model to use? Prefer CatBoost for tabular multiclass
    model = None
    model_name = None
    training_history = None

    if USE_CATBOOST:
        print("Using CatBoostClassifier (recommended for tabular data).")
        model_name = "catboost"
        cat_features = []  # we encoded distance as numeric; if you had categorical columns, list their indices here
        cb = CatBoostClassifier(
            iterations=args.catboost_iters,
            learning_rate=0.03,
            depth=6,
            loss_function="MultiClass",
            eval_metric="MultiClass",
            random_seed=args.seed,
            verbose=200,
            early_stopping_rounds=50
        )
        # CatBoost Pool
        train_pool = Pool(X_train_imputed, y_train, cat_features=cat_features)
        val_pool = Pool(X_val_imputed, y_val, cat_features=cat_features)
        print("Training CatBoost on features shape:", X_train_imputed.shape)
        cb.fit(train_pool, eval_set=val_pool, use_best_model=True)
        model = cb
        # get eval history if present
        try:
            training_history = cb.get_evals_result()
        except Exception:
            training_history = None

    elif USE_XGBOOST:
        print("CatBoost not available -> falling back to XGBoost.")
        model_name = "xgboost"
        # XGBoost requires numeric arrays & multiclass label encoded as 0..K-1
        num_class = len(np.unique(y_train))
        xgb_params = {
            'objective': 'multi:softprob',
            'num_class': num_class,
            'eval_metric': 'mlogloss',
            'eta': 0.05,
            'max_depth': 6,
            'seed': args.seed,
            'verbosity': 1
        }
        dtrain = xgb.DMatrix(X_train_imputed.values, label=y_train.values)
        dval = xgb.DMatrix(X_val_imputed.values, label=y_val.values)
        evals = [(dtrain, 'train'), (dval, 'val')]
        num_round = 1000
        bst = xgb.train(xgb_params, dtrain, num_boost_round=num_round, evals=evals,
                        early_stopping_rounds=50, verbose_eval=50)
        model = bst

    else:
        # final fallback: RandomForest
        print("CatBoost & XGBoost not available -> falling back to RandomForestClassifier.")
        model_name = "randomforest"
        rf = RandomForestClassifier(n_estimators=500, random_state=args.seed, n_jobs=-1)
        rf.fit(X_train_imputed, y_train)
        model = rf

    # -------------------------
    # Predict on test set
    # -------------------------
    print("Evaluating on test set...")
    if model_name == "catboost":
        preds_prob = model.predict_proba(X_test_imputed)
        preds = np.argmax(preds_prob, axis=1)
    elif model_name == "xgboost":
        dtest = xgb.DMatrix(X_test_imputed.values)
        preds_prob = model.predict(dtest)
        preds = np.argmax(preds_prob, axis=1)
    else:  # RandomForest
        preds = model.predict(X_test_imputed)
        try:
            preds_prob = model.predict_proba(X_test_imputed)
        except Exception:
            preds_prob = None

    # metrics
    acc = accuracy_score(y_test, preds)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_test, preds, average='weighted', zero_division=0)
    per_prec, per_rec, per_f1, support = precision_recall_fscore_support(y_test, preds, average=None, zero_division=0)
    class_report_str = classification_report(y_test, preds, target_names=["Attentive", "Distracted", "Disengaged"], zero_division=0)
    conf_mat = confusion_matrix(y_test, preds).tolist()

    test_report = {
        "test_accuracy": float(acc),
        "precision_weighted": float(prec_w),
        "recall_weighted": float(rec_w),
        "f1_weighted": float(f1_w),
        "per_class_precision": [float(x) for x in per_prec.tolist()],
        "per_class_recall": [float(x) for x in per_rec.tolist()],
        "per_class_f1": [float(x) for x in per_f1.tolist()],
        "support": [int(x) for x in support.tolist()],
        "confusion_matrix": conf_mat,
        "num_test_samples": int(len(y_test)),
        "model_type": model_name,
        "args": vars(args),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    }

    # Save JSON report
    report_path = os.path.join(out_dir, "test_report_modelC.json")
    with open(report_path, "w") as f:
        json.dump(test_report, f, indent=2)
    print("Saved test report:", report_path)

    # Save classification report text
    with open(os.path.join(out_dir, "classification_report_modelC.txt"), "w") as f:
        f.write(class_report_str)
    print("Saved classification report (text).")

    # Save confusion matrix plot
    cm = np.array(conf_mat)
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Attentive','Distracted','Disengaged'],
                yticklabels=['Attentive','Distracted','Disengaged'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Test Set (Model C)')
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrix_modelC.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print("Saved confusion matrix:", cm_path)

    # Save feature importances (if model supports it)
    fi_path = os.path.join(out_dir, "feature_importances_modelC.png")
    try:
        if model_name == "catboost":
            # CatBoost get_feature_importance
            importances = model.get_feature_importance()
            feat_names = X_train_imputed.columns.tolist()
            df_fi = pd.DataFrame({"feature": feat_names, "importance": importances})
            df_fi = df_fi.sort_values("importance", ascending=False).head(30)
            plt.figure(figsize=(6, max(4, 0.3*len(df_fi))))
            sns.barplot(x="importance", y="feature", data=df_fi)
            plt.title("CatBoost Feature Importances")
            plt.tight_layout()
            plt.savefig(fi_path, dpi=150)
            plt.close()
            # save raw importance csv
            df_fi.to_csv(os.path.join(out_dir, "feature_importances_modelC.csv"), index=False)
        elif model_name == "xgboost":
            # XGBoost feature importance (fscore)
            fmap = {f"f{i}": name for i, name in enumerate(X_train_imputed.columns.tolist())}
            bst = model
            # get score
            scores = bst.get_score(importance_type='gain')
            # map to names
            feat_names = X_train_imputed.columns.tolist()
            importances = [scores.get(f"f{i}", 0.0) for i in range(len(feat_names))]
            df_fi = pd.DataFrame({"feature": feat_names, "importance": importances})
            df_fi = df_fi.sort_values("importance", ascending=False).head(30)
            plt.figure(figsize=(6, max(4, 0.3*len(df_fi))))
            sns.barplot(x="importance", y="feature", data=df_fi)
            plt.title("XGBoost Feature Importances")
            plt.tight_layout()
            plt.savefig(fi_path, dpi=150)
            plt.close()
            df_fi.to_csv(os.path.join(out_dir, "feature_importances_modelC.csv"), index=False)
        else:
            # RandomForest
            importances = model.feature_importances_
            feat_names = X_train_imputed.columns.tolist()
            df_fi = pd.DataFrame({"feature": feat_names, "importance": importances})
            df_fi = df_fi.sort_values("importance", ascending=False).head(30)
            plt.figure(figsize=(6, max(4, 0.3*len(df_fi))))
            sns.barplot(x="importance", y="feature", data=df_fi)
            plt.title("RandomForest Feature Importances")
            plt.tight_layout()
            plt.savefig(fi_path, dpi=150)
            plt.close()
            df_fi.to_csv(os.path.join(out_dir, "feature_importances_modelC.csv"), index=False)
        print("Saved feature importances:", fi_path)
    except Exception as e:
        print("Could not compute/save feature importances:", e)

    # Save model
    if model_name == "catboost":
        model_path = os.path.join(out_dir, "model_catboost_modelC.cbm")
        model.save_model(model_path)
    elif model_name == "xgboost":
        model_path = os.path.join(out_dir, "model_xgboost_modelC.xgb")
        model.save_model(model_path)
    else:
        model_path = os.path.join(out_dir, "model_randomforest_modelC.joblib")
        joblib.dump(model, model_path)
    print("Saved model to:", model_path)

    # Save training history if available
    if training_history:
        hist_path = os.path.join(out_dir, "training_history_modelC.json")
        with open(hist_path, "w") as f:
            json.dump(training_history, f, indent=2)
        print("Saved training history:", hist_path)

    # Also save final test predictions (optional)
    preds_out = pd.DataFrame({
        "pred": preds.tolist(),
        "true": y_test.tolist()
    })
    preds_out.to_csv(os.path.join(out_dir, "test_predictions_modelC.csv"), index=False)
    print("Saved test predictions CSV.")

    print("\nModel C training & evaluation finished. Outputs in:", out_dir)
    print("Summary metrics:")
    print(json.dumps(test_report, indent=2))
    print("\nClassification report (text):\n")
    print(class_report_str)


if __name__ == "__main__":
    main()
