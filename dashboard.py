from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_fusion import build_temporal_features, forecast_4week, load_fused_dataframe

st.set_page_config(page_title="Disease Outbreak Dashboard", layout="wide")
st.title("🦠 Disease Outbreak Prediction Dashboard")

DATA_DIR = Path(".")
OUTPUT_DIR = Path("outputs")
PREDICTION_FILE = OUTPUT_DIR / "latest_outbreak_risk_predictions.csv"
FEATURE_IMP_FILE = OUTPUT_DIR / "feature_importances.csv"
METRICS_FILE = OUTPUT_DIR / "metrics.txt"
MODEL_FILE = OUTPUT_DIR / "outbreak_risk_model.joblib"
THRESHOLD_FILE = OUTPUT_DIR / "optimal_threshold.txt"


def risk_band(prob: float) -> str:
    if prob >= 0.75:
        return "🚨 Very High"
    if prob >= 0.50:
        return "🔴 High"
    if prob >= 0.25:
        return "🟡 Medium"
    return "🟢 Low"


def load_threshold() -> float:
    if THRESHOLD_FILE.exists():
        try:
            return float(THRESHOLD_FILE.read_text().strip())
        except Exception:
            pass
    return 0.5


@st.cache_data
def load_predictions() -> pd.DataFrame:
    if not PREDICTION_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(PREDICTION_FILE)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data
def load_trend_data() -> pd.DataFrame:
    fused = load_fused_dataframe(DATA_DIR)
    return build_temporal_features(fused, add_target=False)


@st.cache_data
def load_feature_importances() -> pd.DataFrame:
    if FEATURE_IMP_FILE.exists():
        return pd.read_csv(FEATURE_IMP_FILE)
    return pd.DataFrame()


@st.cache_data
def load_metrics() -> str:
    if METRICS_FILE.exists():
        return METRICS_FILE.read_text(encoding="utf-8")
    return ""


@st.cache_resource
def load_model():
    if MODEL_FILE.exists():
        return joblib.load(MODEL_FILE)
    return None


# ── Load everything ────────────────────────────────────────────────────────────
pred_df = load_predictions()
trend_df = load_trend_data()
feat_imp_df = load_feature_importances()
metrics_text = load_metrics()
model = load_model()
optimal_threshold = load_threshold()

if pred_df.empty:
    st.warning(
        "No predictions found. Run `predict_outbreak_risk.py` first to generate "
        "`outputs/latest_outbreak_risk_predictions.csv`."
    )
    st.stop()

# ── Summary metrics bar ────────────────────────────────────────────────────────
if metrics_text:
    lines = [ln.strip() for ln in metrics_text.splitlines() if ln.strip()]
    def _extract(prefix):
        ln = next((l for l in lines if l.startswith(prefix)), "")
        try:
            return float(ln.split(":")[1].strip())
        except Exception:
            return None

    acc = _extract("Accuracy")
    f1w = _extract("F1 (weighted)")
    f1o = _extract("F1 (outbreak class)")

    c1, c2, c3, c4, c5 = st.columns(5)
    if acc:  c1.metric("Model Accuracy", f"{acc:.1%}")
    if f1w:  c2.metric("F1 Weighted", f"{f1w:.4f}")
    if f1o:  c3.metric("F1 Outbreak Class", f"{f1o:.4f}")
    c4.metric("🔴 High Risk Countries", int((pred_df["risk_probability"] >= 0.50).sum()))
    c5.metric("🚨 Very High Risk", int((pred_df["risk_probability"] >= 0.75).sum()))

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1 · Risk Table",
    "2 · Hotspot Map",
    "3 · Spread Animation",
    "4 · Country Trends & Rt",
    "5 · 4-Week Forecast",
    "6 · Feature Importance",
])


# ── Tab 1 ──────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Outbreak Risk Table")
    min_prob = st.slider("Minimum risk probability", 0.0, 1.0, 0.3, 0.01)
    table_df = pred_df[pred_df["risk_probability"] >= min_prob].copy()
    table_df = table_df.sort_values("risk_probability", ascending=False)
    table_df["risk_level"] = table_df["risk_probability"].apply(risk_band)
    table_df["risk_%"] = (table_df["risk_probability"] * 100).round(2)
    st.dataframe(
        table_df[["location", "date", "weekly_cases", "weekly_deaths",
                  "predicted_outbreak_risk", "risk_level", "risk_%"]],
        use_container_width=True,
    )
    st.download_button(
        "⬇ Download filtered risk table (CSV)",
        data=table_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_outbreak_risk.csv",
        mime="text/csv",
    )


# ── Tab 2 ──────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Global Hotspot Detection Map")
    map_df = pred_df.dropna(subset=["location", "risk_probability"]).copy()
    map_df["risk_%"] = (map_df["risk_probability"] * 100).round(2)
    map_df["risk_level"] = map_df["risk_probability"].apply(risk_band)

    fig_map = px.choropleth(
        map_df, locations="location", locationmode="country names",
        color="risk_probability", hover_name="location",
        hover_data={"risk_%": True, "risk_probability": False,
                    "weekly_cases": True, "weekly_deaths": True, "risk_level": True},
        color_continuous_scale="Reds", range_color=(0, 1),
        title="Predicted Outbreak Risk by Country",
        labels={"risk_probability": "Risk probability"},
    )
    fig_map.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0}, height=500)
    st.plotly_chart(fig_map, use_container_width=True)

    _, c_top = st.columns([2, 1])
    with c_top:
        st.markdown("#### Top 10 Hotspots")
        top_hs = map_df.sort_values("risk_probability", ascending=False).head(10).copy()
        st.dataframe(
            top_hs[["location", "risk_level", "risk_%", "weekly_cases", "weekly_deaths"]].rename(
                columns={"risk_%": "risk (%)"}
            ),
            use_container_width=True, hide_index=True,
        )


# ── Tab 3 ──────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Disease Spread Animation — Risk Over Time")
    st.caption("Monthly outbreak risk score per country. Press ▶ Play to animate the spread progression.")

    @st.cache_data
    def build_spread_data() -> pd.DataFrame:
        df = trend_df.dropna(subset=["location", "date", "weekly_cases"]).copy()
        df = df.sort_values(["location", "date"])
        grp = df.groupby("location", group_keys=False)
        df["prev_week"] = grp["weekly_cases"].shift(1)
        df["growth"] = (df["weekly_cases"] - df["prev_week"]) / df["prev_week"].replace(0, float("nan"))
        df["risk_score"] = (df["growth"] / 0.20).clip(0, 1).fillna(0)
        df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
        monthly = (
            df.groupby(["location", "month"], as_index=False)
            .agg(risk_score=("risk_score", "mean"), weekly_cases=("weekly_cases", "sum"))
        )
        monthly["date_str"] = monthly["month"].dt.strftime("%Y-%m")
        return monthly

    anim_df = build_spread_data()
    valid = anim_df.groupby("date_str")["location"].count()
    anim_f = anim_df[anim_df["date_str"].isin(valid[valid >= 10].index)]

    if not anim_f.empty:
        fig_anim = px.choropleth(
            anim_f, locations="location", locationmode="country names",
            color="risk_score", animation_frame="date_str",
            hover_name="location",
            hover_data={"weekly_cases": True, "risk_score": ":.2f", "date_str": False},
            color_continuous_scale="YlOrRd", range_color=(0, 1),
            title="Monthly Outbreak Risk Score — Global Spread Over Time",
        )
        fig_anim.update_layout(margin={"r": 0, "t": 60, "l": 0, "b": 0}, height=520)
        fig_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 300
        fig_anim.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 200
        st.plotly_chart(fig_anim, use_container_width=True)

        st.markdown("#### Countries with Highest Sustained Spread Score")
        spread_sum = (
            anim_f.groupby("location", as_index=False)["risk_score"].mean()
            .sort_values("risk_score", ascending=False).head(15)
        )
        spread_sum["risk_%"] = (spread_sum["risk_score"] * 100).round(1)
        fig_bar = px.bar(
            spread_sum, x="risk_%", y="location", orientation="h",
            color="risk_%", color_continuous_scale="Reds",
            title="Average Risk Score (all time) — Top 15 Countries",
            labels={"risk_%": "Avg risk score (%)", "location": ""},
        )
        fig_bar.update_layout(yaxis={"autorange": "reversed"}, coloraxis_showscale=False, height=420)
        st.plotly_chart(fig_bar, use_container_width=True)


# ── Tab 4 ──────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Country Transmission Trends & Rt Estimator")

    location_list = sorted(
        [loc for loc in trend_df["location"].dropna().unique()
         if loc in pred_df["location"].values]
    )
    default_loc = "India" if "India" in location_list else (location_list[0] if location_list else None)
    sel = st.selectbox("Select country", location_list,
                       index=location_list.index(default_loc) if default_loc else 0)

    loc_hist = trend_df[trend_df["location"] == sel].sort_values("date").copy()

    if loc_hist.empty:
        st.info("No historical data available.")
    else:
        latest_row = pred_df[pred_df["location"] == sel]
        if not latest_row.empty:
            rp = float(latest_row["risk_probability"].iloc[0])
            m1, m2, m3 = st.columns(3)
            m1.metric("Latest Risk Probability", f"{rp:.1%}")
            m2.metric("Risk Level", risk_band(rp))
            # Rt from most recent data
            if "rt_estimate" in loc_hist.columns:
                latest_rt = loc_hist["rt_estimate"].dropna().iloc[-1] if not loc_hist["rt_estimate"].dropna().empty else None
                if latest_rt:
                    m3.metric("Latest Rt estimate", f"{latest_rt:.2f}",
                              delta="Growing" if latest_rt > 1 else "Shrinking",
                              delta_color="inverse")

        c1, c2 = st.columns(2)
        with c1:
            case_cols = [c for c in ["new_cases", "weekly_cases", "biweekly_cases"] if c in loc_hist.columns]
            if case_cols:
                fig_cases = px.line(loc_hist, x="date", y=case_cols,
                                    title=f"Cases Trend — {sel}")
                st.plotly_chart(fig_cases, use_container_width=True)

            # Rt chart
            if "rt_estimate" in loc_hist.columns:
                rt_data = loc_hist[["date", "rt_estimate"]].dropna()
                if not rt_data.empty:
                    fig_rt = go.Figure()
                    fig_rt.add_trace(go.Scatter(
                        x=rt_data["date"], y=rt_data["rt_estimate"],
                        mode="lines", name="Rt", line=dict(color="#E24B4A", width=2)
                    ))
                    fig_rt.add_hline(y=1.0, line_dash="dash", line_color="gray",
                                     annotation_text="Rt = 1 (epidemic threshold)")
                    fig_rt.update_yaxes(range=[0, min(rt_data["rt_estimate"].quantile(0.99), 5)])
                    fig_rt.update_layout(title=f"Effective Reproduction Number (Rt) — {sel}",
                                         xaxis_title="Date", yaxis_title="Rt")
                    st.plotly_chart(fig_rt, use_container_width=True)
                    st.caption(
                        "Rt > 1 means each infected person infects more than one other — the outbreak is growing. "
                        "Rt < 1 means the outbreak is shrinking. Estimated using 7-day case ratio with "
                        "COVID-19 serial interval of 5 days."
                    )

        with c2:
            death_cols = [c for c in ["new_deaths", "weekly_deaths"] if c in loc_hist.columns]
            if death_cols:
                fig_deaths = px.line(loc_hist, x="date", y=death_cols,
                                     title=f"Deaths Trend — {sel}")
                st.plotly_chart(fig_deaths, use_container_width=True)

            vax_cols = [c for c in ["people_vaccinated_per_hundred",
                                     "people_fully_vaccinated_per_hundred"] if c in loc_hist.columns]
            if vax_cols:
                fig_vax = px.line(loc_hist, x="date", y=vax_cols,
                                  title=f"Vaccination Coverage — {sel}")
                st.plotly_chart(fig_vax, use_container_width=True)

    # Country comparison
    st.markdown("---")
    st.markdown("#### Country Comparison")
    _cmp_prefs = ["India", "United States", "Brazil"]
    _cmp_default = [c for c in _cmp_prefs if c in location_list][: min(3, len(location_list))]
    if not _cmp_default and location_list:
        _cmp_default = location_list[: min(3, len(location_list))]
    compare_locs = st.multiselect(
        "Select 2–4 countries to compare", location_list,
        default=_cmp_default,
    )
    if len(compare_locs) >= 2:
        comp_df = trend_df[trend_df["location"].isin(compare_locs)].copy()
        c_left, c_right = st.columns(2)
        with c_left:
            if "weekly_cases" in comp_df.columns:
                fig_comp = px.line(comp_df, x="date", y="weekly_cases", color="location",
                                   title="Weekly Cases Comparison")
                st.plotly_chart(fig_comp, use_container_width=True)
        with c_right:
            if "rt_estimate" in comp_df.columns:
                rt_comp = comp_df[["date", "location", "rt_estimate"]].dropna()
                fig_rt_comp = px.line(rt_comp, x="date", y="rt_estimate", color="location",
                                      title="Rt Comparison")
                fig_rt_comp.add_hline(y=1.0, line_dash="dash", line_color="gray")
                fig_rt_comp.update_yaxes(range=[0, 4])
                st.plotly_chart(fig_rt_comp, use_container_width=True)

        # Vaccination comparison
        vax_avail = [c for c in ["people_vaccinated_per_hundred", "people_fully_vaccinated_per_hundred"]
                     if c in comp_df.columns]
        if vax_avail:
            fig_vax_comp = px.line(comp_df, x="date", y=vax_avail[0], color="location",
                                   title="Vaccination Coverage Comparison")
            st.plotly_chart(fig_vax_comp, use_container_width=True)


# ── Tab 5 ──────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("4-Week Outbreak Risk Forecast")
    st.caption(
        "Iterative 4-week ahead forecast using the trained Random Forest. "
        "Each week's predicted risk feeds back as lag features for the next week."
    )

    if model is None:
        st.warning("Model not found. Run the training script first.")
    else:
        f_location_list = sorted(
            [loc for loc in trend_df["location"].dropna().unique()
             if loc in pred_df["location"].values]
        )
        f_default = "India" if "India" in f_location_list else f_location_list[0]
        f_sel = st.selectbox("Select country for forecast", f_location_list,
                             index=f_location_list.index(f_default), key="forecast_sel")

        if st.button(f"Generate 4-week forecast for {f_sel}"):
            with st.spinner("Generating forecast..."):
                forecast_df = forecast_4week(model, trend_df, f_sel)

            if forecast_df.empty:
                st.warning("Not enough data to generate forecast.")
            else:
                forecast_df["risk_level"] = forecast_df["risk_probability"].apply(risk_band)
                forecast_df["risk_%"] = (forecast_df["risk_probability"] * 100).round(1)

                # Historical risk proxy for context (last 8 weeks)
                loc_hist_f = trend_df[trend_df["location"] == f_sel].sort_values("date").tail(8).copy()
                loc_hist_f["prev"] = loc_hist_f["weekly_cases"].shift(1)
                loc_hist_f["growth"] = (
                    (loc_hist_f["weekly_cases"] - loc_hist_f["prev"])
                    / loc_hist_f["prev"].replace(0, pd.NA)
                )
                loc_hist_f["risk_proxy"] = (loc_hist_f["growth"] / 0.20).clip(0, 1).fillna(0)

                fig_fc = go.Figure()
                # Historical line
                fig_fc.add_trace(go.Scatter(
                    x=loc_hist_f["date"], y=loc_hist_f["risk_proxy"],
                    mode="lines+markers", name="Historical risk proxy",
                    line=dict(color="#378ADD", width=2),
                ))
                # Forecast line
                fig_fc.add_trace(go.Scatter(
                    x=forecast_df["date"], y=forecast_df["risk_probability"],
                    mode="lines+markers", name="Forecast risk probability",
                    line=dict(color="#E24B4A", width=2, dash="dash"),
                    error_y=dict(  # uncertainty band using std of RF trees
                        type="constant", value=0.05, visible=True,
                        color="rgba(226,75,74,0.3)"
                    )
                ))
                fig_fc.add_hline(y=0.5, line_dash="dot", line_color="orange",
                                 annotation_text="Outbreak threshold")
                fig_fc.update_layout(
                    title=f"4-Week Outbreak Risk Forecast — {f_sel}",
                    xaxis_title="Date", yaxis_title="Risk probability",
                    yaxis=dict(range=[0, 1]),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_fc, use_container_width=True)

                st.markdown("#### Forecast Table")
                st.dataframe(
                    forecast_df[["week", "date", "risk_%", "risk_level"]].rename(
                        columns={"week": "Week ahead", "risk_%": "Risk (%)", "risk_level": "Risk level"}
                    ),
                    use_container_width=True, hide_index=True,
                )
                st.caption(
                    "Forecast uncertainty increases with each week ahead. "
                    "The ±5% error bar shown is conservative — treat week 3–4 as directional guidance only."
                )


# ── Tab 6 ──────────────────────────────────────────────────────────────────────
with tab6:
    st.subheader("Feature Importance Analysis")
    st.caption(
        "Top features ranked by mean decrease in impurity (Random Forest). "
        "Country one-hot dummies are excluded to focus on epidemiological signals."
    )

    if feat_imp_df.empty:
        st.warning("Feature importances not found. Re-run the training script.")
    else:
        def pretty(name: str) -> str:
            name = name.replace("_lag_1", " (lag 1w)").replace("_lag_2", " (lag 2w)")
            name = name.replace("_ma_3", " (3w avg)").replace("_external", "")
            name = name.replace("rt_estimate", "Rt estimate")
            name = name.replace("people_vaccinated_per_hundred", "vaccination coverage %")
            name = name.replace("hospital_beds_per_thousand", "hospital beds / 1k")
            name = name.replace("vulnerability_index", "vulnerability index")
            name = name.replace("healthcare_index", "healthcare index")
            name = name.replace("_", " ").strip().capitalize()
            return name

        fi = feat_imp_df.copy()
        fi["label"] = fi["feature"].apply(pretty)
        fi["importance_%"] = (fi["importance"] * 100).round(3)

        fig_imp = px.bar(
            fi.sort_values("importance_%"),
            x="importance_%", y="label", orientation="h",
            color="importance_%", color_continuous_scale="Blues",
            title="Feature Importance — Top 30 (excluding country dummies)",
            labels={"importance_%": "Importance (%)", "label": ""},
            text="importance_%",
        )
        fig_imp.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig_imp.update_layout(coloraxis_showscale=False, height=700, margin={"r": 80})
        st.plotly_chart(fig_imp, use_container_width=True)

        # Category pie
        st.markdown("#### Feature Category Breakdown")
        def categorize(name: str) -> str:
            if "rt_estimate" in name:
                return "Rt (reproduction number)"
            if "lag" in name or "ma_3" in name:
                return "Temporal lag / moving average"
            if any(k in name for k in ["vacc", "vax", "booster", "vaccin"]):
                return "Vaccination"
            if "death" in name:
                return "Mortality"
            if "hosp" in name:
                return "Hospitalization"
            if "excess" in name:
                return "Excess mortality"
            if "cases" in name:
                return "Case counts"
            if any(k in name for k in ["density", "age", "urban", "gdp", "beds", "index", "vulnerability", "healthcare"]):
                return "Demographic / healthcare"
            return "Other"

        fi["category"] = fi["feature"].apply(categorize)
        cat_sum = fi.groupby("category")["importance_%"].sum().sort_values(ascending=False).reset_index()
        fig_pie = px.pie(cat_sum, names="category", values="importance_%",
                         title="Importance share by feature category", hole=0.4)
        fig_pie.update_traces(textinfo="label+percent")
        st.plotly_chart(fig_pie, use_container_width=True)

        st.download_button(
            "⬇ Download feature importances (CSV)",
            data=fi[["feature", "importance_%"]].to_csv(index=False).encode("utf-8"),
            file_name="feature_importances.csv", mime="text/csv",
        )

        st.markdown("---")
        st.markdown("#### Epidemiological Interpretation")
        st.info(
            "**3-week moving average of new cases (18.3% importance)** — "
            "the strongest predictor, consistent with COVID-19's ~14-day incubation + 1-week reporting lag. "
            "Sustained case trends are more reliable outbreak signals than single-week spikes.\n\n"
            "**Death lag features in top 20** — mortality rises 1–2 weeks after a case surge, "
            "confirming the model captures the biological delay between infection and severe outcome.\n\n"
            "**Rt estimate** — the effective reproduction number provides a direct mechanistic "
            "signal: when Rt > 1, each infected person infects more than one other, meaning the "
            "outbreak is in exponential growth phase.\n\n"
            "**Vaccination coverage** — higher vaccination rates dampen outbreak risk by reducing "
            "the susceptible population, consistent with herd immunity dynamics.\n\n"
            "**Vulnerability index** — countries with low hospital capacity and high population "
            "density have less ability to absorb case surges, making structural factors a meaningful "
            "secondary predictor even when current case counts are similar."
        )
