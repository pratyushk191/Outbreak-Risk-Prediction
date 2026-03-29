"""
Unit tests for CODECURE Track C — Disease Spread Prediction
Run with: pytest tests/
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ── helpers so tests run without the full dataset ────────────────────────────

def _make_country_week_df(n_countries: int = 3, n_weeks: int = 20) -> pd.DataFrame:
    """Create a minimal synthetic fused DataFrame for testing."""
    dates = pd.date_range("2021-01-01", periods=n_weeks, freq="W")
    rows = []
    rng = np.random.default_rng(0)
    for loc in [f"Country_{i}" for i in range(n_countries)]:
        cases = rng.integers(10, 500, size=n_weeks).astype(float)
        rows.append(pd.DataFrame({
            "location": loc,
            "date": dates,
            "new_cases": cases,
            "new_deaths": (cases * 0.02).round(),
            "weekly_cases": cases,
            "weekly_deaths": (cases * 0.02).round(),
            "biweekly_cases": cases * 2,
            "people_vaccinated_per_hundred": rng.uniform(0, 80, n_weeks),
        }))
    return pd.concat(rows, ignore_index=True)


# ── data shape tests ─────────────────────────────────────────────────────────

class TestDataShape:
    def test_synthetic_df_columns(self):
        df = _make_country_week_df()
        assert "location" in df.columns
        assert "date" in df.columns
        assert "new_cases" in df.columns

    def test_synthetic_df_no_empty_rows(self):
        df = _make_country_week_df()
        assert len(df) > 0

    def test_synthetic_df_country_count(self):
        df = _make_country_week_df(n_countries=5)
        assert df["location"].nunique() == 5


# ── Rt estimator tests ────────────────────────────────────────────────────────

class TestRtEstimator:
    """Rt = (cases_t / cases_{t-7})^(SI/7), SI=5 days."""

    SERIAL_INTERVAL = 5

    def _compute_rt(self, cases_now: float, cases_prev: float) -> float:
        if cases_prev <= 0 or cases_now <= 0:
            return np.nan
        return (cases_now / cases_prev) ** (self.SERIAL_INTERVAL / 7)

    def test_rt_equals_one_when_stable(self):
        rt = self._compute_rt(100, 100)
        assert abs(rt - 1.0) < 1e-9

    def test_rt_greater_than_one_when_growing(self):
        rt = self._compute_rt(200, 100)
        assert rt > 1.0

    def test_rt_less_than_one_when_declining(self):
        rt = self._compute_rt(50, 100)
        assert rt < 1.0

    def test_rt_nan_on_zero_previous(self):
        rt = self._compute_rt(100, 0)
        assert np.isnan(rt)

    def test_rt_nan_on_zero_current(self):
        rt = self._compute_rt(0, 100)
        assert np.isnan(rt)


# ── label generation tests ───────────────────────────────────────────────────

class TestOutbreakLabel:
    """outbreak_risk=1 when next_week_cases >= current * 1.20 and cases >= 100."""

    GROWTH_THRESHOLD = 1.20
    CASE_FLOOR = 100

    def _label(self, current: float, next_week: float) -> int:
        if current < self.CASE_FLOOR:
            return 0
        return int(next_week >= current * self.GROWTH_THRESHOLD)

    def test_no_outbreak_when_stable(self):
        assert self._label(200, 200) == 0

    def test_outbreak_on_20pct_growth(self):
        assert self._label(200, 240) == 1

    def test_outbreak_on_large_surge(self):
        assert self._label(500, 1000) == 1

    def test_no_outbreak_below_case_floor(self):
        # Only 50 cases — below 100 floor
        assert self._label(50, 200) == 0

    def test_no_outbreak_on_slight_increase(self):
        # 10% growth is not ≥ 20%
        assert self._label(200, 219) == 0


# ── temporal split tests ──────────────────────────────────────────────────────

class TestTemporalSplit:
    def _temporal_split(self, df: pd.DataFrame, test_frac: float = 0.20):
        dates = pd.to_datetime(df["date"], errors="coerce")
        cutoff = dates.quantile(1.0 - test_frac)
        return df[dates <= cutoff].copy(), df[dates > cutoff].copy()

    def test_no_date_overlap(self):
        df = _make_country_week_df(n_weeks=30)
        train, test = self._temporal_split(df)
        train_dates = set(train["date"].astype(str))
        test_dates = set(test["date"].astype(str))
        assert train_dates.isdisjoint(test_dates)

    def test_split_ratio_approx(self):
        df = _make_country_week_df(n_weeks=100)
        train, test = self._temporal_split(df, test_frac=0.20)
        actual_frac = len(test) / len(df)
        assert 0.15 < actual_frac < 0.25

    def test_train_is_earlier_than_test(self):
        df = _make_country_week_df(n_weeks=30)
        train, test = self._temporal_split(df)
        assert train["date"].max() <= test["date"].min()


# ── model smoke test (v3 artefact format) ─────────────────────────────────────

class TestModelSmokeTest:
    """
    Integration test: verify the trained model loads and produces valid probabilities.
    Skipped if outputs/outbreak_risk_model.joblib does not exist (e.g. in CI without training).
    """

    MODEL_PATH     = "outputs/outbreak_risk_model.joblib"
    THRESHOLD_PATH = "outputs/optimal_threshold.txt"

    @pytest.mark.skipif(
        not __import__("pathlib").Path(MODEL_PATH).exists(),
        reason="Trained model not found — run disease_spread_model.py first",
    )
    def test_model_loads(self):
        import joblib
        art = joblib.load(self.MODEL_PATH)
        assert isinstance(art, dict), "Artefact should be a dict"
        assert "preprocessor" in art

    @pytest.mark.skipif(
        not __import__("pathlib").Path(MODEL_PATH).exists(),
        reason="Trained model not found — run disease_spread_model.py first",
    )
    def test_predict_proba_range(self):
        import joblib
        import numpy as np
        import pandas as pd

        art = joblib.load(self.MODEL_PATH)
        preprocessor = art["preprocessor"]
        rf_est  = art.get("rf")  or art.get("ensemble")
        gbm_est = art.get("gbm")
        weights = art.get("weights", [1, 1])
        w_sum   = sum(weights)

        # Build a minimal synthetic row
        num_cols = list(preprocessor.transformers_[0][2])
        cat_cols = list(preprocessor.transformers_[1][2])
        row = pd.DataFrame(
            {c: [0.0] for c in num_cols} | {c: ["Unknown"] for c in cat_cols}
        )
        X = preprocessor.transform(row).astype(np.float32)

        if gbm_est is not None:
            prob = float(
                (rf_est.predict_proba(X)[:, 1] * weights[0] +
                 gbm_est.predict_proba(X)[:, 1] * weights[1]) / w_sum
            )
        else:
            prob = float(rf_est.predict_proba(X)[0, 1])

        assert 0.0 <= prob <= 1.0, f"predict_proba out of range: {prob}"

    @pytest.mark.skipif(
        not __import__("pathlib").Path(THRESHOLD_PATH).exists(),
        reason="Threshold file not found — run disease_spread_model.py first",
    )
    def test_threshold_in_range(self):
        t = float(open(self.THRESHOLD_PATH).read().strip())
        assert 0.0 < t < 1.0, f"Threshold out of (0,1): {t}"

    @pytest.mark.skipif(
        not __import__("pathlib").Path(MODEL_PATH).exists(),
        reason="Trained model not found — run disease_spread_model.py first",
    )
    def test_rolling_ma_causal(self):
        """Verify that lag_1 feature is strictly older than lag_2 for same country."""
        import joblib
        # Just a data property test — does not need the model
        from data_fusion import build_temporal_features
        import pandas as pd, numpy as np

        dates = pd.date_range("2021-01-01", periods=15, freq="W")
        df = pd.DataFrame({
            "location": "TestCountry",
            "date": dates,
            "new_cases": np.arange(15, dtype=float) * 100 + 200,
            "new_deaths": np.ones(15) * 5,
            "weekly_cases": np.arange(15, dtype=float) * 100 + 200,
            "weekly_deaths": np.ones(15) * 2,
            "biweekly_cases": np.arange(15, dtype=float) * 200 + 400,
        })
        out = build_temporal_features(df, add_target=False)
        out = out.dropna(subset=["new_cases_lag_1", "new_cases_lag_2"])
        # lag_1 should be one step newer than lag_2
        assert (out["new_cases_lag_1"].values > out["new_cases_lag_2"].values).all(), \
            "lag_1 should always be newer (larger) than lag_2 for a monotonically increasing series"


# ── performance regression tests ──────────────────────────────────────────────

class TestMetricsRegression:
    """
    Guard rails: if the model is retrained and key metrics regress significantly,
    these tests fail loudly rather than silently shipping a worse model.
    Skipped if outputs/metrics.txt does not exist.
    """

    METRICS_PATH = "outputs/metrics.txt"

    # Minimum acceptable thresholds — based on v3/v5 baseline
    MIN_PR_AUC  = 0.30
    MIN_RECALL  = 0.55   # recall < 0.55 means missing too many real outbreaks
    MAX_THRESHOLD = 0.92  # threshold > 0.92 indicates model is too conservative

    @pytest.mark.skipif(
        not __import__("pathlib").Path(METRICS_PATH).exists(),
        reason="metrics.txt not found — run disease_spread_model.py first",
    )
    def _parse_metrics(self) -> dict:
        text = open(self.METRICS_PATH).read()
        import re
        metrics = {}
        for key, pattern in [
            ("pr_auc",    r"PR-AUC:\s*([\d.]+)"),
            ("recall",    r"Outbreak\s+[\d.]+\s+(\S+)"),
            ("threshold", r"Optimal threshold:\s*([\d.]+)"),
            ("f1",        r"F1 \(outbreak class\):\s*([\d.]+)"),
        ]:
            m = re.search(pattern, text)
            if m:
                metrics[key] = float(m.group(1))
        return metrics

    @pytest.mark.skipif(
        not __import__("pathlib").Path(METRICS_PATH).exists(),
        reason="metrics.txt not found — run disease_spread_model.py first",
    )
    def test_pr_auc_above_floor(self):
        m = self._parse_metrics()
        assert "pr_auc" in m, "PR-AUC not found in metrics.txt"
        assert m["pr_auc"] >= self.MIN_PR_AUC, (
            f"PR-AUC {m['pr_auc']:.4f} < floor {self.MIN_PR_AUC} — model regressed"
        )

    @pytest.mark.skipif(
        not __import__("pathlib").Path(METRICS_PATH).exists(),
        reason="metrics.txt not found — run disease_spread_model.py first",
    )
    def test_recall_above_floor(self):
        m = self._parse_metrics()
        assert "recall" in m, "Recall not found in metrics.txt"
        assert m["recall"] >= self.MIN_RECALL, (
            f"Outbreak recall {m['recall']:.3f} < floor {self.MIN_RECALL} — "
            f"model is missing too many real outbreaks"
        )

    @pytest.mark.skipif(
        not __import__("pathlib").Path(METRICS_PATH).exists(),
        reason="metrics.txt not found — run disease_spread_model.py first",
    )
    def test_threshold_not_too_conservative(self):
        t = float(open(self.THRESHOLD_PATH if hasattr(self, "THRESHOLD_PATH") else "outputs/optimal_threshold.txt").read().strip())
        assert t <= self.MAX_THRESHOLD, (
            f"Threshold {t:.4f} > {self.MAX_THRESHOLD} — model is too conservative, "
            f"likely caused by a spurious dominant feature"
        )

    @pytest.mark.skipif(
        not __import__("pathlib").Path(METRICS_PATH).exists(),
        reason="metrics.txt not found — run disease_spread_model.py first",
    )
    def test_no_spurious_dominant_feature(self):
        """No single feature family should exceed 15% of total importance."""
        import csv, re
        from pathlib import Path
        fi_path = Path("outputs/feature_importances.csv")
        if not fi_path.exists():
            pytest.skip("feature_importances.csv not found")
        rows = list(csv.DictReader(open(fi_path)))
        total = sum(float(r["importance"]) for r in rows)
        # Group by base feature family (strip _lag_N, _ma_N suffixes)
        from collections import defaultdict
        family_totals: dict = defaultdict(float)
        for r in rows:
            base = re.sub(r"_(lag_\d+|ma_\d+)$", "", r["feature"])
            family_totals[base] += float(r["importance"]) / total
        for family, share in family_totals.items():
            assert share <= 0.15, (
                f"Feature family '{family}' has {share:.1%} importance — "
                f"likely a spurious or leaky signal dominating the model"
            )
