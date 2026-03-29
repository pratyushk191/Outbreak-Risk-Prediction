"""
disease_spread_model.py — CODECURE Track C  (v5 — fixed)
=========================================================
Key fixes vs v2:
  1. Rolling MA leakage fixed in data_fusion.py (shift-then-roll is fully causal)
  2. PR-AUC is the PRIMARY metric (accuracy/weighted-F1 are meaningless at high imbalance)
  3. Threshold chosen on validation PR curve at target precision >= 0.20
  4. GBM trained with sample_weight (pos weight = imbalance ratio, capped 60x)
     RF trained with class_weight='balanced_subsample' — no imblearn/XGBoost needed
  5. Metrics file leads with PR-AUC and outbreak F1, explains why accuracy is misleading
  6. Smoke test verifies saved model loads and produces valid probabilities

Key fixes vs v4 (retrained model regression):
  7. death_reporting_sparse excluded from training features.
     The flag is a post-2023 surveillance artefact correlated with sparse outbreak labels,
     not real epidemiology. In v5 it captured 23% of feature importance and pushed the
     threshold to 0.896, causing recall to collapse from 0.95 -> 0.60. Excluded here;
     kept in data_fusion.py for dashboard diagnostics only.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_fusion import build_temporal_features, load_fused_dataframe

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ── memory ────────────────────────────────────────────────────────────────────

def downcast(df: pd.DataFrame) -> pd.DataFrame:
    fc = df.select_dtypes(include="float64").columns
    ic = df.select_dtypes(include="int64").columns
    df[fc] = df[fc].astype(np.float32)
    df[ic] = df[ic].astype(np.int32)
    return df


# ── splits ────────────────────────────────────────────────────────────────────

def temporal_split(df: pd.DataFrame, test_frac: float = 0.20):
    dates = pd.to_datetime(df["date"], errors="coerce")
    cutoff = dates.quantile(1.0 - test_frac)
    return df[dates <= cutoff].copy(), df[dates > cutoff].copy()


def temporal_holdout_last(df: pd.DataFrame, holdout_frac: float):
    dates = pd.to_datetime(df["date"], errors="coerce")
    cutoff = dates.quantile(1.0 - holdout_frac)
    return df[dates <= cutoff].copy(), df[dates > cutoff].copy()


# ── preprocessor ─────────────────────────────────────────────────────────────

def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ])


# ── threshold on PR curve ─────────────────────────────────────────────────────

def find_threshold_pr(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_precision: float = 0.20,
    min_recall: float = 0.35,
) -> tuple[float, float, float, float]:
    """
    Choose threshold on the VALIDATION PR curve.

    Target operating point: precision >= target_precision AND recall >= min_recall,
    maximising F1. Falls back to recall >= min_recall only if no point meets both.

    Returns (threshold, precision, recall, f1).
    """
    prec_arr, rec_arr, thresholds = precision_recall_curve(y_true, y_proba)

    best = dict(t=0.5, f1=0.0, p=0.0, r=0.0)

    for pass_n, prec_floor in enumerate([target_precision, 0.0]):
        if pass_n == 1 and best["f1"] > 0.0:
            break
        if pass_n == 1:
            print(f"  [threshold] Relaxing precision floor to 0.0 (recall>={min_recall} only)")
        for p, r, t in zip(prec_arr[:-1], rec_arr[:-1], thresholds):
            if p < prec_floor or r < min_recall:
                continue
            denom = p + r
            f1 = 2 * p * r / denom if denom > 0 else 0.0
            if f1 > best["f1"]:
                best = dict(t=float(t), f1=f1, p=float(p), r=float(r))

    print(f"  Threshold={best['t']:.4f}  prec={best['p']:.3f}  "
          f"rec={best['r']:.3f}  F1={best['f1']:.3f}")
    return best["t"], best["p"], best["r"], best["f1"]


# ── smoke test ────────────────────────────────────────────────────────────────

def smoke_test(artefact_path: Path, threshold_path: Path) -> bool:
    """
    Integration check: load saved artefact, run predict_proba on a synthetic row.
    Verifies the full inference path works before submission.
    """
    try:
        art = joblib.load(artefact_path)
        preprocessor = art["preprocessor"]
        rf_est       = art["rf"]
        gbm_est      = art["gbm"]
        weights      = art["weights"]
        threshold    = float(threshold_path.read_text().strip())

        num_cols = preprocessor.transformers_[0][2]
        cat_cols = preprocessor.transformers_[1][2]
        row = pd.DataFrame(
            {c: [0.0] for c in num_cols} | {c: ["Unknown"] for c in cat_cols}
        )
        X = preprocessor.transform(row).astype(np.float32)

        w_sum = sum(weights)
        prob = float(
            ((rf_est.predict_proba(X)[:, 1] * weights[0] +
              gbm_est.predict_proba(X)[:, 1] * weights[1]) / w_sum)[0]
        )
        pred = int(prob >= threshold)
        assert 0.0 <= prob <= 1.0, f"predict_proba out of range: {prob}"
        print(f"  [smoke test] PASSED  prob={prob:.4f}  pred={pred}  threshold={threshold:.4f}")
        return True
    except Exception as exc:
        print(f"  [smoke test] FAILED: {exc}")
        return False


# ── train + evaluate ──────────────────────────────────────────────────────────

def train_and_evaluate(df: pd.DataFrame, out_dir: Path) -> None:
    np.random.seed(42)

    target = "outbreak_risk"
    df = downcast(df.dropna(subset=[target]).copy())

    drop = {"date", "outbreak_risk", "next_week_cases"}
    # Exclude death_reporting_sparse and ALL its derived lag/MA columns from training.
    # This flag marks rows where death reporting is zero — a post-2023 surveillance
    # wind-down artefact, not genuine epidemiology. Because outbreak labels are also
    # sparse in that same period, the model learns "no death reports = no outbreak"
    # (a spurious correlation). In the retrained model it captured 23% of total feature
    # importance and pushed the threshold to 0.896 — causing 40% of real outbreaks to
    # be missed. Keep the flag in data_fusion.py for dashboard diagnostics only.
    diagnostic_only = {c for c in df.columns if "death_reporting_sparse" in c}
    feat_cols = [
        c for c in df.columns
        if c not in drop
        and c not in diagnostic_only
        and not df[c].isna().all()
    ]
    if diagnostic_only:
        print(f"  Excluded {len(diagnostic_only)} diagnostic columns: {sorted(diagnostic_only)}")

    n_pos = int(df[target].sum())
    n_neg = len(df) - n_pos
    global_imb = n_neg / max(n_pos, 1)
    print(f"  Positive rate: {df[target].mean():.2%}  ({n_pos:,} / {len(df):,})")
    print(f"  Imbalance ratio: {global_imb:.1f}:1")

    train_df, test_df = temporal_split(df, test_frac=0.20)
    train_fit_df, val_df = temporal_holdout_last(train_df, holdout_frac=0.15)
    print(f"  Train-fit: {len(train_fit_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    cat_cols = [c for c in ["location"] if c in feat_cols]
    num_cols = [c for c in feat_cols if c not in cat_cols]

    y_tr = train_fit_df[target].values
    y_va = val_df[target].values
    y_te = test_df[target].values

    train_pos = int(y_tr.sum())
    train_imb = (len(y_tr) - train_pos) / max(train_pos, 1)
    pos_weight = float(np.clip(train_imb, 1.0, 60.0))
    print(f"  Train imbalance: {train_imb:.1f}:1  -> GBM sample weight={pos_weight:.1f}x")

    preprocessor = build_preprocessor(num_cols, cat_cols)
    preprocessor.fit(train_fit_df[feat_cols])

    X_tr = preprocessor.transform(train_fit_df[feat_cols]).astype(np.float32)
    X_va = preprocessor.transform(val_df[feat_cols]).astype(np.float32)
    X_te = preprocessor.transform(test_df[feat_cols]).astype(np.float32)

    # Sample weights for GBM (minority class gets pos_weight, majority = 1)
    sw_tr = np.where(y_tr == 1, pos_weight, 1.0)

    # Train RF with class_weight='balanced_subsample' (handles imbalance internally)
    print("  Training RandomForest (class_weight=balanced_subsample)...")
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)

    # Train GBM with sample_weight (directly weighted minority class)
    print("  Training GBM with sample_weight...")
    gbm = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.80,
        max_features="sqrt",
        min_samples_leaf=8,
        random_state=42,
    )
    gbm.fit(X_tr, y_tr, sample_weight=sw_tr)
    print("  Training complete.")

    # Weighted soft vote: RF=1, GBM=2
    weights = [1, 2]
    w_sum = sum(weights)

    def predict_proba_ensemble(X: np.ndarray) -> np.ndarray:
        return (rf.predict_proba(X) * weights[0] +
                gbm.predict_proba(X) * weights[1]) / w_sum

    # Threshold chosen on VALIDATION set only (test never touched)
    y_val_prob = predict_proba_ensemble(X_va)[:, 1]
    threshold, val_p, val_r, val_f1 = find_threshold_pr(
        y_va, y_val_prob, target_precision=0.20, min_recall=0.35,
    )

    # Final test evaluation
    y_prob = predict_proba_ensemble(X_te)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    pr_auc = average_precision_score(y_te, y_prob)
    roc    = roc_auc_score(y_te, y_prob)
    f1out  = f1_score(y_te, y_pred, pos_label=1, average="binary", zero_division=0)
    f1w    = f1_score(y_te, y_pred, average="weighted", zero_division=0)
    acc    = accuracy_score(y_te, y_pred)
    rep    = classification_report(
        y_te, y_pred, target_names=["No outbreak", "Outbreak"], zero_division=0
    )

    print(f"\n--- Test Results ---")
    print(f"  PR-AUC (primary):     {pr_auc:.4f}")
    print(f"  F1 (outbreak class):  {f1out:.4f}")
    print(f"  ROC-AUC:              {roc:.4f}")
    print(f"  Accuracy:             {acc:.4f}  (misleading at {global_imb:.0f}:1 imbalance)")
    print(f"  F1 (weighted):        {f1w:.4f}  (dominated by majority class)")
    print(f"  Threshold:            {threshold:.4f}")
    print(f"\n{rep}")

    # Save artefacts
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path     = out_dir / "outbreak_risk_model.joblib"
    threshold_path = out_dir / "optimal_threshold.txt"

    joblib.dump({
        "preprocessor": preprocessor,
        "rf":  rf,
        "gbm": gbm,
        "weights": weights,
        "_type": "weighted_vote_rf_gbm",
    }, model_path)
    threshold_path.write_text(str(threshold))

    # Metrics: PR-AUC first, with explanatory notes
    with (out_dir / "metrics.txt").open("w") as f:
        f.write(
            f"=== Primary metrics (imbalance-aware) ===\n"
            f"PR-AUC: {pr_auc:.4f}\n"
            f"F1 (outbreak class): {f1out:.4f}\n"
            f"ROC-AUC: {roc:.4f}\n"
            f"Optimal threshold: {threshold:.4f}\n"
            f"Val PR operating point: precision={val_p:.3f} recall={val_r:.3f} F1={val_f1:.3f}\n\n"
            f"=== Secondary metrics (misleading at {global_imb:.0f}:1 imbalance) ===\n"
            f"Accuracy: {acc:.4f}  "
            f"[baseline majority-class classifier: {1 - df[target].mean():.2%}]\n"
            f"F1 (weighted): {f1w:.4f}\n\n"
            f"{rep}"
        )

    # Feature importances from GBM (weighted minority class = more reliable)
    try:
        ohe = preprocessor.transformers_[1][1].named_steps["ohe"]
        all_feat = (list(preprocessor.transformers_[0][2]) +
                    list(ohe.get_feature_names_out(["location"])))
        if len(gbm.feature_importances_) == len(all_feat):
            fi = (
                pd.DataFrame({"feature": all_feat, "importance": gbm.feature_importances_})
                .query("not feature.str.startswith('location_')")
                .sort_values("importance", ascending=False)
                .head(30)
                .reset_index(drop=True)
            )
            fi.to_csv(out_dir / "feature_importances.csv", index=False)
            print("\nTop 10 features (GBM, weighted for outbreak class):")
            print(fi.head(10).to_string(index=False))
    except Exception as exc:
        print(f"[warn] Feature importance export skipped: {exc}")

    # Smoke test
    print("\n--- Smoke test ---")
    smoke_test(model_path, threshold_path)

    print(f"\nAll artefacts saved -> {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",   default=".")
    p.add_argument("--output-dir", default="outputs")
    return p.parse_args()


def main():
    args = parse_args()
    print("Loading datasets...")
    fused = load_fused_dataframe(Path(args.data_dir))
    print(f"  Fused: {fused.shape}")
    print("Building features (leakage-safe causal rolling MA)...")
    prepared = build_temporal_features(fused, add_target=True)
    print(f"  Prepared: {prepared.shape}")
    train_and_evaluate(prepared, Path(args.output_dir))


if __name__ == "__main__":
    main()
