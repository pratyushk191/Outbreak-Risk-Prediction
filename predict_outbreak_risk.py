"""
predict_outbreak_risk.py — generates latest predictions from trained model.

Compatible with:
  - v3 format: dict {"preprocessor": ..., "rf": ..., "gbm": ..., "weights": ...}
  - v2 format: dict {"preprocessor": ..., "ensemble": ...}
  - v1 format: sklearn Pipeline directly
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data_fusion import build_temporal_features, load_fused_dataframe


def load_threshold(output_dir: Path) -> float:
    path = output_dir / "optimal_threshold.txt"
    if path.exists():
        try:
            return float(path.read_text().strip())
        except Exception:
            pass
    return 0.5


def main() -> None:
    output_dir = Path("outputs")
    model_path = output_dir / "outbreak_risk_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Trained model not found. Run disease_spread_model.py first.")

    artefact  = joblib.load(model_path)
    threshold = load_threshold(output_dir)
    print(f"Using prediction threshold: {threshold:.4f}")

    fused    = load_fused_dataframe(Path("."))
    features = build_temporal_features(fused, add_target=False)
    latest   = features.groupby("location", as_index=False).tail(1).copy()
    latest   = latest.dropna(subset=["weekly_cases"])

    # Detect artefact format
    if isinstance(artefact, dict) and "rf" in artefact:
        # v3: separate RF + GBM, weighted soft vote
        preprocessor = artefact["preprocessor"]
        rf_est  = artefact["rf"]
        gbm_est = artefact["gbm"]
        weights = artefact.get("weights", [1, 2])
        w_sum   = sum(weights)

        num_cols  = list(preprocessor.transformers_[0][2])
        cat_cols  = list(preprocessor.transformers_[1][2])
        feat_cols = num_cols + cat_cols
        for col in feat_cols:
            if col not in latest.columns:
                latest[col] = np.nan

        X = preprocessor.transform(latest[feat_cols]).astype(np.float32)
        probs = (
            rf_est.predict_proba(X)[:, 1]  * weights[0] +
            gbm_est.predict_proba(X)[:, 1] * weights[1]
        ) / w_sum

    elif isinstance(artefact, dict) and "ensemble" in artefact:
        # v2: dict with single ensemble key
        preprocessor = artefact["preprocessor"]
        ensemble_est = artefact["ensemble"]
        num_cols  = list(preprocessor.transformers_[0][2])
        cat_cols  = list(preprocessor.transformers_[1][2])
        feat_cols = num_cols + cat_cols
        for col in feat_cols:
            if col not in latest.columns:
                latest[col] = np.nan
        X = preprocessor.transform(latest[feat_cols]).astype(np.float32)
        probs = ensemble_est.predict_proba(X)[:, 1]

    else:
        # v1: sklearn Pipeline
        model     = artefact
        feat_cols = [c for c in model.feature_names_in_ if c not in {"date", "outbreak_risk", "next_week_cases"}]
        for col in feat_cols:
            if col not in latest.columns:
                latest[col] = np.nan
        probs = model.predict_proba(latest[feat_cols])[:, 1]

    preds = (probs >= threshold).astype(int)

    out = latest[["location", "date", "weekly_cases", "weekly_deaths"]].copy()
    out["predicted_outbreak_risk"] = preds
    out["risk_probability"]        = probs.round(4)
    if "rt_estimate" in latest.columns:
        out["rt_estimate"] = latest["rt_estimate"].values

    out = out.sort_values("risk_probability", ascending=False)
    out.to_csv(output_dir / "latest_outbreak_risk_predictions.csv", index=False)

    print(f"Saved {len(out)} predictions -> outputs/latest_outbreak_risk_predictions.csv")
    print(f"High risk countries (>50%):  {int((probs >= 0.50).sum())}")
    print(f"Very high risk (>75%):       {int((probs >= 0.75).sum())}")

    if (probs >= threshold).sum() == 0:
        print(
            "\n[NOTE] No countries exceed the outbreak threshold in the most recent data.\n"
            "  This is expected for Aug 2024 rows where weekly_deaths is largely 0.0 —\n"
            "  COVID-19 death reporting became sparse globally post-2023, suppressing\n"
            "  death-related features that are among the top predictors.\n"
            "  This is a data recency artefact, not a model failure.\n"
            "  See README > Known Limitations for details."
        )

    print("\nTop 10 highest risk:")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
