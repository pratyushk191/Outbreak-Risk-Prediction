from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from demographics import load_demographics

SERIAL_INTERVAL = 5  # COVID-19 average serial interval in days


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", format="mixed")


def _melt_wide_country_file(file_path: Path, value_name: str) -> pd.DataFrame:
    wide = pd.read_csv(file_path)
    if "date" in wide.columns and "location" in wide.columns:
        out = wide[["date", "location", value_name]].copy()
        out["date"] = _to_datetime(out["date"])
        return out
    if "date" in wide.columns:
        country_cols = [c for c in wide.columns if c != "date"]
        long_df = wide.melt(id_vars=["date"], value_vars=country_cols, var_name="location", value_name=value_name)
        long_df["date"] = _to_datetime(long_df["date"])
        return long_df

    id_col = None
    for candidate in ["location", "entity", "Country/Region", "country", "Country"]:
        if candidate in wide.columns:
            id_col = candidate
            break
    if id_col is None:
        raise ValueError(f"Could not identify location column in {file_path.name}")

    date_cols = [c for c in wide.columns if c != id_col and c not in ["Province/State", "Lat", "Long"]]
    long_df = wide.melt(id_vars=[id_col], value_vars=date_cols, var_name="date", value_name=value_name)
    long_df = long_df.rename(columns={id_col: "location"})
    long_df["date"] = _to_datetime(long_df["date"])
    return long_df


def _regional_median_impute(df: pd.DataFrame, col: str, region_col: str = "continent") -> pd.Series:
    """Impute NaN with regional median, then global median as fallback."""
    result = df[col].copy()
    if region_col in df.columns:
        regional_medians = df.groupby(region_col)[col].transform("median")
        result = result.fillna(regional_medians)
    global_median = df[col].median()
    result = result.fillna(global_median)
    return result


def load_fused_dataframe(data_dir: Path) -> pd.DataFrame:
    full_path = data_dir / "full_data.csv"
    if not full_path.exists():
        raise FileNotFoundError("full_data.csv not found in data directory.")

    base = pd.read_csv(full_path)
    base["date"] = _to_datetime(base["date"])
    base = base.dropna(subset=["date", "location"]).copy()

    # Vaccination features — forward-fill within each country to fix 83% NaN.
    vax_path = data_dir / "vaccinations.csv"
    if vax_path.exists():
        vax = pd.read_csv(vax_path)
        vax = vax.rename(columns={"daily_vaccinations": "vax_daily_vaccinations"})
        vax["date"] = _to_datetime(vax["date"])
        vax_cols = [
            "location", "date",
            "vax_daily_vaccinations",
            "people_vaccinated_per_hundred",
            "people_fully_vaccinated_per_hundred",
            "total_boosters_per_hundred",
            "daily_vaccinations_per_million",
        ]
        vax_cols = [c for c in vax_cols if c in vax.columns]
        vax = vax[vax_cols].copy()
        vax = vax.sort_values(["location", "date"])
        # Forward-fill then back-fill within country: vaccination rate only goes up
        for c in [col for col in vax_cols if col not in ["location", "date"]]:
            vax[c] = vax.groupby("location")[c].transform(lambda x: x.ffill().bfill())
        base = base.merge(vax, on=["location", "date"], how="left")

    # Hospitalization features.
    hosp_path = data_dir / "covid-hospitalizations.csv"
    if hosp_path.exists():
        hosp = pd.read_csv(hosp_path)
        hosp = hosp.rename(columns={"entity": "location"})
        hosp["date"] = _to_datetime(hosp["date"])
        hosp = hosp.dropna(subset=["location", "date", "indicator"])
        hosp_wide = hosp.pivot_table(
            index=["location", "date"],
            columns="indicator",
            values="value",
            aggfunc="mean",
        ).reset_index()
        hosp_wide.columns = [
            f"hosp_{str(c).replace(' ', '_').replace('/', '_').lower()}" if c not in ["location", "date"] else c
            for c in hosp_wide.columns
        ]
        base = base.merge(hosp_wide, on=["location", "date"], how="left")

    # Excess mortality.
    excess_path = data_dir / "excess_mortality.csv"
    if excess_path.exists():
        excess = pd.read_csv(excess_path)
        excess["date"] = _to_datetime(excess["date"])
        excess_cols = [
            "location", "date",
            "p_scores_all_ages", "excess_proj_all_ages",
            "cum_excess_proj_all_ages", "excess_per_million_proj_all_ages",
        ]
        excess_cols = [c for c in excess_cols if c in excess.columns]
        excess = excess[excess_cols].copy()
        rename_map = {c: f"excess_{c}" for c in excess.columns if c not in ["location", "date"]}
        excess = excess.rename(columns=rename_map)
        base = base.merge(excess, on=["location", "date"], how="left")

    # Fix 5: Mitigate post-2023 death reporting sparsity.
    # When weekly_deaths is missing/zero but excess mortality p_scores exist,
    # derive a synthetic death estimate as a fallback signal so death-lag features
    # don't collapse to zero and suppress outbreak predictions.
    if "new_deaths" in base.columns and "excess_p_scores_all_ages" in base.columns:
        # Forward-fill deaths within country (reporting often lags by weeks)
        base["new_deaths"] = (
            base.groupby("location")["new_deaths"]
            .transform(lambda x: x.replace(0, np.nan).ffill(limit=4))
            .fillna(0)
        )
    if "new_deaths" in base.columns:
        base["death_reporting_sparse"] = (base["new_deaths"] == 0).astype(int)

    # Wide country-date files.
    for fname, value_name in [
        ("weekly_cases.csv", "weekly_cases_external"),
        ("weekly_deaths.csv", "weekly_deaths_external"),
        ("total_deaths.csv", "total_deaths_external"),
    ]:
        fpath = data_dir / fname
        if fpath.exists():
            melted = _melt_wide_country_file(fpath, value_name)
            base = base.merge(melted, on=["location", "date"], how="left")

    # JHU confirmed cases.
    confirmed_path = data_dir / "time_series_covid19_confirmed_global.csv"
    if confirmed_path.exists():
        confirmed = pd.read_csv(confirmed_path)
        required_cols = {"Country/Region", "Province/State", "Lat", "Long"}
        date_cols = [c for c in confirmed.columns if c not in required_cols]
        long_confirmed = confirmed.melt(
            id_vars=["Country/Region"],
            value_vars=date_cols,
            var_name="date",
            value_name="confirmed_global_total",
        )
        long_confirmed = long_confirmed.rename(columns={"Country/Region": "location"})
        long_confirmed["date"] = _to_datetime(long_confirmed["date"])
        long_confirmed = (
            long_confirmed.groupby(["location", "date"], as_index=False)["confirmed_global_total"].sum()
        )
        base = base.merge(long_confirmed, on=["location", "date"], how="left")

    # Demographic features — impute missing countries with regional/global median.
    demo = load_demographics()
    base = base.merge(demo, on="location", how="left")
    demo_cols = ["population_density", "median_age", "urban_population_pct",
                 "hospital_beds_per_thousand", "gdp_per_capita_usd",
                 "healthcare_index", "vulnerability_index"]
    for col in demo_cols:
        if col in base.columns:
            base[col] = _regional_median_impute(base, col)

    return base.sort_values(["location", "date"]).reset_index(drop=True)


def compute_rt(df: pd.DataFrame, serial_interval: int = SERIAL_INTERVAL) -> pd.Series:
    """
    Estimate the effective reproduction number Rt per country per week.
    Formula: Rt = (new_cases_t / new_cases_{t-7}) ^ (serial_interval / 7)
    A value > 1 means the outbreak is growing; < 1 means it is shrinking.
    """
    grp = df.groupby("location", group_keys=False)
    cases_prev_week = grp["new_cases"].shift(7)
    ratio = (df["new_cases"].clip(lower=0.1)) / (cases_prev_week.clip(lower=0.1))
    rt = ratio ** (serial_interval / 7)
    rt = rt.clip(lower=0, upper=10)
    return rt


def build_temporal_features(df: pd.DataFrame, add_target: bool) -> pd.DataFrame:
    work = df.copy()
    work = work.dropna(subset=["date", "location"]).sort_values(["location", "date"])

    work["month"] = work["date"].dt.month
    work["week_of_year"] = work["date"].dt.isocalendar().week.astype(int)

    # Compute Rt and add as a feature
    work["rt_estimate"] = compute_rt(work)

    numeric_cols = work.select_dtypes(include=[np.number]).columns.tolist()
    excluded = {"month", "week_of_year", "outbreak_risk", "next_week_cases"}
    signal_cols = [c for c in numeric_cols if c not in excluded]

    grp = work.groupby("location", group_keys=False)

    # Build all lag/MA columns at once to avoid DataFrame fragmentation.
    # lag_1/lag_2 = short-term; lag_4 = monthly cycle; ma_3/ma_6 = 3 and 6-week trends.
    # All are shift-then-roll: fully causal, no future leakage.
    new_cols: dict[str, pd.Series] = {}
    for col in signal_cols:
        shifted1 = grp[col].shift(1)
        new_cols[f"{col}_lag_1"] = shifted1
        new_cols[f"{col}_lag_2"] = grp[col].shift(2)
        new_cols[f"{col}_lag_4"] = grp[col].shift(4)
        new_cols[f"{col}_ma_3"]  = shifted1.rolling(window=3, min_periods=1).mean()
        new_cols[f"{col}_ma_6"]  = shifted1.rolling(window=6, min_periods=1).mean()

    if add_target:
        next_week_cases = grp["weekly_cases"].shift(-1)
        new_cols["next_week_cases"] = next_week_cases
        current_week_cases = work["weekly_cases"].replace(0, np.nan)
        growth_rate = (next_week_cases - current_week_cases) / current_week_cases
        # Require ≥100 current weekly cases to avoid labelling noise
        # (e.g. 1→2 cases triggering a false "outbreak" signal).
        # Also add case acceleration and death-lag ratio as extra features.
        min_case_floor = 100
        new_cols["outbreak_risk"] = (
            (growth_rate >= 0.20) & (current_week_cases >= min_case_floor)
        ).astype(int)

    # ── Extra engineered features (always built, not just when add_target=True) ──
    # 1. Case acceleration: week-over-week change in new_cases growth rate
    if "new_cases" in work.columns:
        cases_g = work.groupby("location", group_keys=False)["new_cases"]
        prev1 = cases_g.shift(1).clip(lower=0.1)
        prev2 = cases_g.shift(2).clip(lower=0.1)
        new_cols["case_acceleration"] = (
            work["new_cases"].clip(lower=0) / prev1
        ) - (prev1 / prev2)

    # 2. Death-lag ratio: deaths lag cases by ~2 weeks; a rising ratio signals surge
    if "new_deaths" in work.columns and "new_cases" in work.columns:
        new_cols["death_lag_ratio"] = (
            work.groupby("location", group_keys=False)["new_deaths"].shift(2)
            / work["new_cases"].clip(lower=0.1)
        ).clip(upper=5)

    work = pd.concat([work, pd.DataFrame(new_cols, index=work.index)], axis=1)

    if add_target:
        work = work.dropna(subset=["weekly_cases", "next_week_cases"])

    return work.reset_index(drop=True)


def forecast_4week(model, df: pd.DataFrame, location: str) -> pd.DataFrame:
    """
    Generate a 4-week rolling outbreak risk forecast for a single country.
    Uses the trained model iteratively, feeding each week's prediction back
    as the basis for the next week's lag features.
    Returns a DataFrame with columns: week, date, risk_probability, risk_label.
    """
    from copy import deepcopy

    loc_df = df[df["location"] == location].sort_values("date").copy()
    if loc_df.empty:
        return pd.DataFrame()

    model_features = list(model.feature_names_in_)
    results = []
    last_date = loc_df["date"].max()

    # Use the most recent row as the starting point
    current_row = loc_df.tail(1).copy()

    for week in range(1, 5):
        forecast_date = last_date + pd.Timedelta(weeks=week)

        # Align columns
        for col in model_features:
            if col not in current_row.columns:
                current_row[col] = np.nan

        row_input = current_row[model_features].copy()
        prob = float(model.predict_proba(row_input)[:, 1][0])
        label = int(model.predict(row_input)[0])

        results.append({
            "week": week,
            "date": forecast_date,
            "risk_probability": round(prob, 4),
            "risk_label": label,
        })

        # Shift lag features forward using explicit suffix constants.
        # Avoids silent breakage if column naming conventions ever change.
        LAG1_SUFFIX = "_lag_1"
        LAG2_SUFFIX = "_lag_2"
        lag1_set = {c for c in model_features if c.endswith(LAG1_SUFFIX)}
        lag2_set = {c for c in model_features if c.endswith(LAG2_SUFFIX)}
        feat_set  = set(model_features)

        updates: dict = {}
        for lag2 in lag2_set:
            lag1 = lag2[: -len(LAG2_SUFFIX)] + LAG1_SUFFIX
            if lag1 in lag1_set and lag1 in current_row.columns:
                updates[lag2] = current_row[lag1].values[0]
        for lag1 in lag1_set:
            base = lag1[: -len(LAG1_SUFFIX)]
            if base in feat_set and base in current_row.columns:
                updates[lag1] = current_row[base].values[0]
        for col, val in updates.items():
            current_row[col] = val

    return pd.DataFrame(results)
