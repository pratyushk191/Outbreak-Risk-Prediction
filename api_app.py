from pathlib import Path
import subprocess
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from data_fusion import build_temporal_features, load_fused_dataframe


# ── Input validation schemas ──────────────────────────────────────────────────

class AlertRequest(BaseModel):
    threshold: float = Field(default=0.7, ge=0.0, le=1.0,
        description="Risk probability threshold (0–1). Countries above this are flagged.")

class TrendRequest(BaseModel):
    weeks: int = Field(default=26, ge=4, le=104,
        description="Number of historical weeks to return (4–104).")
    include_vaccination: bool = Field(default=True,
        description="Include vaccination columns in the response.")

class PredictionsRequest(BaseModel):
    min_prob: float = Field(default=0.0, ge=0.0, le=1.0,
        description="Minimum risk probability filter.")
    limit: int = Field(default=50, ge=1, le=1000,
        description="Max number of results to return.")

    @field_validator("min_prob")
    @classmethod
    def round_prob(cls, v: float) -> float:
        return round(v, 4)


app = FastAPI(
    title="Disease Outbreak API",
    description="API to train model, generate predictions, and view outbreak analytics.",
    version="1.0.0",
)

BASE_DIR = Path(".")
OUTPUT_DIR = BASE_DIR / "outputs"
PRED_FILE = OUTPUT_DIR / "latest_outbreak_risk_predictions.csv"
MODEL_FILE = OUTPUT_DIR / "outbreak_risk_model.joblib"


def _risk_band(prob: float) -> str:
    if prob >= 0.75:
        return "Very High"
    if prob >= 0.50:
        return "High"
    if prob >= 0.25:
        return "Medium"
    return "Low"


def _run_cmd(args: list[str]) -> str:
    try:
        result = subprocess.run(args, cwd=BASE_DIR, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise HTTPException(status_code=500, detail=detail) from exc


def _load_predictions() -> pd.DataFrame:
    if not PRED_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Predictions file not found. Call POST /predict first.",
        )
    df = pd.read_csv(PRED_FILE)
    if "risk_probability" not in df.columns:
        raise HTTPException(status_code=500, detail="Predictions file is malformed.")
    df["risk_level"] = df["risk_probability"].apply(_risk_band)
    return df


@app.get("/")
def root() -> dict:
    alert_count = 0
    if PRED_FILE.exists():
        df = pd.read_csv(PRED_FILE)
        alert_count = int((df["risk_probability"] > 0.7).sum())
    return {
        "message": "Disease Outbreak API running",
        "docs": "/docs",
        "model_exists": MODEL_FILE.exists(),
        "predictions_exists": PRED_FILE.exists(),
        "high_risk_alert_count": alert_count,
        "alert": f"⚠️ {alert_count} countries with outbreak risk > 70%" if alert_count else "✅ No high-risk alerts",
    }


@app.post("/train")
def train_model() -> dict:
    output = _run_cmd(
        [
            "py",
            "-3",
            "disease spread  feasibility model.py",
            "--data-dir",
            ".",
            "--output-dir",
            "outputs",
        ]
    )
    return {"status": "ok", "message": "Model trained successfully.", "output": output}


@app.post("/predict")
def generate_predictions() -> dict:
    output = _run_cmd(["py", "-3", "predict_outbreak_risk.py"])
    return {"status": "ok", "message": "Predictions generated successfully.", "output": output}


@app.get("/predictions")
def get_predictions(params: PredictionsRequest = Query()) -> dict:
    """Return paginated predictions, filtered by minimum risk probability."""
    df = _load_predictions()
    filt = (
        df[df["risk_probability"] >= params.min_prob]
        .sort_values("risk_probability", ascending=False)
        .head(params.limit)
    )
    return {"count": int(len(filt)), "items": filt.to_dict(orient="records")}


@app.get("/hotspots")
def get_hotspots(top_n: int = Query(default=10, ge=1, le=100)) -> dict:
    df = _load_predictions().sort_values("risk_probability", ascending=False).head(top_n)
    return {"count": int(len(df)), "items": df.to_dict(orient="records")}


@app.get("/alerts")
def get_alerts(threshold: float = Query(default=0.7, ge=0.0, le=1.0)) -> dict:
    """Return all countries whose predicted outbreak risk exceeds the threshold."""
    df = _load_predictions()
    alerts = df[df["risk_probability"] > threshold].sort_values("risk_probability", ascending=False)
    return {
        "threshold": threshold,
        "alert_count": int(len(alerts)),
        "message": f"⚠️ {len(alerts)} countries with outbreak risk > {threshold:.0%}" if len(alerts) else "✅ No alerts",
        "items": alerts[["location", "risk_probability", "risk_level", "weekly_cases", "weekly_deaths"]].to_dict(orient="records"),
    }


@app.get("/trend/{location_name}")
def get_location_trend(
    location_name: str,
    weeks: int = Query(default=26, ge=4, le=104),
    include_vaccination: bool = Query(default=True),
) -> dict:
    fused = load_fused_dataframe(BASE_DIR)
    feat = build_temporal_features(fused, add_target=False)
    loc = feat[feat["location"] == location_name].sort_values("date").tail(weeks)
    if loc.empty:
        raise HTTPException(status_code=404, detail=f"No data found for location: {location_name}")

    cols = ["date", "new_cases", "weekly_cases", "new_deaths", "weekly_deaths"]
    if include_vaccination:
        cols.extend(
            [
                c
                for c in [
                    "vax_daily_vaccinations",
                    "people_vaccinated_per_hundred",
                    "people_fully_vaccinated_per_hundred",
                ]
                if c in loc.columns
            ]
        )

    trend = loc[cols].copy()
    trend["date"] = trend["date"].dt.strftime("%Y-%m-%d")

    # 8-week risk proxy based on weekly case growth.
    risk_proxy = loc[["date", "weekly_cases"]].copy()
    risk_proxy["prev_week_cases"] = risk_proxy["weekly_cases"].shift(1)
    risk_proxy["weekly_growth"] = (
        (risk_proxy["weekly_cases"] - risk_proxy["prev_week_cases"])
        / risk_proxy["prev_week_cases"].replace(0, pd.NA)
    )
    risk_proxy["risk_proxy_score"] = (risk_proxy["weekly_growth"] / 0.20).clip(lower=0, upper=1).fillna(0)
    risk_proxy = risk_proxy.tail(8)
    risk_proxy["date"] = risk_proxy["date"].dt.strftime("%Y-%m-%d")

    return {
        "location": location_name,
        "weeks_returned": int(len(trend)),
        "trend": trend.to_dict(orient="records"),
        "risk_proxy_last_8_weeks": risk_proxy[["date", "risk_proxy_score"]].to_dict(orient="records"),
    }


@app.get("/locations")
def get_locations(search: Optional[str] = None, limit: int = Query(default=300, ge=1, le=1000)) -> dict:
    if PRED_FILE.exists():
        locations = pd.read_csv(PRED_FILE)["location"].dropna().astype(str).unique().tolist()
    else:
        fused = load_fused_dataframe(BASE_DIR)
        locations = fused["location"].dropna().astype(str).unique().tolist()

    if search:
        query = search.lower()
        locations = [loc for loc in locations if query in loc.lower()]
    locations = sorted(locations)[:limit]
    return {"count": len(locations), "items": locations}
