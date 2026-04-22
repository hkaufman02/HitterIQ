import sys
import os
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.abspath("src"))

from data_loader import load_hitter_data
from features import build_hitter_features
from targets import create_breakout_target, create_breakout_score


# ---------- Helper Functions ----------
def render_text_block(title: str, df: pd.DataFrame):
    st.markdown(f"#### {title}")
    st.code(df.to_string(index=False), language=None)


def get_age_group(age: int) -> str:
    if age < 25:
        return "Young Stars (Under 25)"
    elif 25 <= age <= 29:
        return "Prime Hitters (25-29)"
    else:
        return "Veterans (30+)"


def get_similar_players(df: pd.DataFrame, player_name: str, feature_cols: list, top_n: int = 5):
    comp_df = df.copy()
    latest_season = comp_df["Season"].max()
    comp_df = comp_df[comp_df["Season"] == latest_season].copy()
    comp_df = comp_df.dropna(subset=feature_cols).reset_index(drop=True)

    if player_name not in comp_df["Name"].values:
        return pd.DataFrame()

    X = comp_df[feature_cols].fillna(0)
    similarity_matrix = cosine_similarity(X)

    player_index = comp_df[comp_df["Name"] == player_name].index[0]
    similarities = list(enumerate(similarity_matrix[player_index]))
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

    similar_players = []
    for idx, score in similarities[1:top_n + 1]:
        similar_players.append(
            {
                "Name": comp_df.iloc[idx]["Name"],
                "Team": comp_df.iloc[idx]["Team"],
                "Age": int(comp_df.iloc[idx]["Age"]),
                "Similarity": f"{score:.2f}",
            }
        )

    return pd.DataFrame(similar_players)


# ---------- App Config ----------
st.set_page_config(page_title="HitterIQ", layout="wide")

# ---------- Styling ----------
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #b0b7c3;
        margin-bottom: 1rem;
    }

    .info-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 16px 18px;
        margin-bottom: 14px;
    }

    .badge {
        display: inline-block;
        padding: 6px 10px;
        margin-right: 8px;
        margin-bottom: 8px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.08);
    }

    .mini-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 12px 14px;
        text-align: center;
    }

    .mini-label {
        font-size: 0.9rem;
        color: #aab3bf;
        margin-bottom: 4px;
    }

    .mini-value {
        font-size: 1.2rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Load Data ----------
raw_df = load_hitter_data(2021, 2025)
feat_df = build_hitter_features(raw_df)
feat_df = create_breakout_target(feat_df)
feat_df = create_breakout_score(feat_df)

breakout_model = joblib.load("models/breakout_model.pkl")
ops_model = joblib.load("models/ops_regressor.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")

latest_season = int(feat_df["Season"].max())
latest_df = feat_df[feat_df["Season"] == latest_season].copy()

X_all = latest_df[feature_cols].fillna(0)
latest_df["predicted_breakout_prob"] = breakout_model.predict_proba(X_all)[:, 1]
latest_df["projected_ops"] = ops_model.predict(X_all)

latest_df["ai_score"] = (
    latest_df["predicted_breakout_prob"] * 50
    + latest_df["projected_ops"] * 35
    + latest_df["breakout_score"] * 0.15
)

latest_df["age_group"] = latest_df["Age"].apply(get_age_group)

# ---------- Header ----------
st.markdown('<div class="main-title">⚾ HitterIQ</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-powered MLB hitter analytics dashboard for breakout probability, projected OPS, rankings, player comps, and model insights.</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-card">
    <div style="font-size:1.15rem;font-weight:700;margin-bottom:0.6rem;">What this app does</div>
    <span class="badge">Breakout Probability</span>
    <span class="badge">Projected OPS</span>
    <span class="badge">AI Score Ranking</span>
    <span class="badge">Power / Discipline / Production</span>
    <span class="badge">Age Filters</span>
    <span class="badge">Similar Players</span>
    <span class="badge">Model Insights</span>
    <div style="margin-top:0.8rem;">
        This dashboard uses engineered baseball features and machine learning models to rank hitters and evaluate future upside.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- Filter ----------
st.divider()
st.subheader("Filter View")

filter_choice = st.selectbox(
    "Choose player group",
    [
        "All Hitters",
        "Young Stars (Under 25)",
        "Prime Hitters (25-29)",
        "Veterans (30+)"
    ]
)

if filter_choice == "All Hitters":
    filtered_df = latest_df.copy()
else:
    filtered_df = latest_df[latest_df["age_group"] == filter_choice].copy()

if filtered_df.empty:
    st.warning("No players match this filter.")
    st.stop()

# ---------- Top Summary Cards ----------
top1, top2, top3 = st.columns(3)

leader_name = filtered_df.sort_values("ai_score", ascending=False).iloc[0]["Name"]
leader_breakout = filtered_df.sort_values("predicted_breakout_prob", ascending=False).iloc[0]["Name"]
leader_power = filtered_df.sort_values("power_score", ascending=False).iloc[0]["Name"]

with top1:
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="mini-label">Top AI Score</div>
            <div class="mini-value">{leader_name}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with top2:
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="mini-label">Top Breakout Pick</div>
            <div class="mini-value">{leader_breakout}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with top3:
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="mini-label">Top Power Bat</div>
            <div class="mini-value">{leader_power}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["📊 Rankings", "👤 Player Breakdown", "📄 Raw Data"])

# ---------- Rankings Tab ----------
with tab1:
    st.subheader(f"{latest_season} Leaderboards — {filter_choice}")

    leaderboard = filtered_df[
        [
            "Name", "Team", "Age", "OPS", "HR", "wOBA",
            "breakout_score", "predicted_breakout_prob",
            "projected_ops", "ai_score"
        ]
    ].sort_values("ai_score", ascending=False).head(10).copy()

    leaderboard_display = leaderboard.copy()
    leaderboard_display["OPS"] = leaderboard_display["OPS"].map(lambda x: f"{x:.3f}")
    leaderboard_display["wOBA"] = leaderboard_display["wOBA"].map(lambda x: f"{x:.3f}")
    leaderboard_display["breakout_score"] = leaderboard_display["breakout_score"].map(lambda x: f"{x:.2f}")
    leaderboard_display["predicted_breakout_prob"] = leaderboard_display["predicted_breakout_prob"].map(lambda x: f"{x:.1%}")
    leaderboard_display["projected_ops"] = leaderboard_display["projected_ops"].map(lambda x: f"{x:.3f}")
    leaderboard_display["ai_score"] = leaderboard_display["ai_score"].map(lambda x: f"{x:.2f}")

    render_text_block("Top 10 AI Hitters", leaderboard_display)

    col_a, col_b = st.columns(2)

    with col_a:
        top_breakouts = filtered_df.sort_values("predicted_breakout_prob", ascending=False)[
            ["Name", "Team", "Age", "predicted_breakout_prob"]
        ].head(5).copy()

        top_breakouts["predicted_breakout_prob"] = top_breakouts["predicted_breakout_prob"].map(lambda x: f"{x:.1%}")
        render_text_block("🔥 Top Breakout Candidates", top_breakouts)

    with col_b:
        top_power = filtered_df.sort_values("power_score", ascending=False)[
            ["Name", "Team", "Age", "power_score"]
        ].head(5).copy()

        top_power["power_score"] = top_power["power_score"].map(lambda x: f"{x:.3f}")
        render_text_block("💣 Top Power Hitters", top_power)

# ---------- Player Breakdown Tab ----------
with tab2:
    available_players = sorted(filtered_df["Name"].dropna().unique())
    selected_player = st.selectbox("Choose a hitter", available_players)

    player_df = feat_df[feat_df["Name"] == selected_player].sort_values("Season").copy()
    latest = player_df.iloc[-1]

    X_player = pd.DataFrame([latest[feature_cols].fillna(0)])
    breakout_prob = breakout_model.predict_proba(X_player)[0][1]
    projected_ops = ops_model.predict(X_player)[0]

    player_ai_score = (
        breakout_prob * 50
        + projected_ops * 35
        + float(latest.get("breakout_score", 0)) * 0.15
    )

    if breakout_prob >= 0.70:
        outlook = "🔥 Strong Breakout Outlook"
    elif breakout_prob >= 0.45:
        outlook = "⚡ Moderate Breakout Outlook"
    else:
        outlook = "❄️ Lower Breakout Outlook"

    st.subheader(f"👤 Player Breakdown: {selected_player}")
    st.markdown(
        f"**Team:** {latest['Team']}  |  **Season:** {int(latest['Season'])}  |  **Age:** {int(latest['Age'])}  |  **Group:** {get_age_group(int(latest['Age']))}  |  **Outlook:** {outlook}"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("OPS", round(float(latest["OPS"]), 3))
    col2.metric("Breakout Probability", f"{breakout_prob:.1%}")
    col3.metric("Projected OPS", f"{projected_ops:.3f}")
    col4.metric("AI Score", round(float(player_ai_score), 2))

    col5, col6, col7 = st.columns(3)
    col5.metric("Power Score", round(float(latest.get("power_score", 0)), 3))
    col6.metric("Discipline Score", round(float(latest.get("discipline_score", 0)), 3))
    col7.metric("Production Score", round(float(latest.get("production_score", 0)), 3))

    st.info("""
**How AI Score works**
- Breakout probability is weighted the most
- Projected OPS estimates future offensive output
- Breakout score rewards year-over-year improvement

Higher AI Score usually means a better overall hitter outlook.
""")

    st.divider()
    similar_players_df = get_similar_players(
        feat_df,
        selected_player,
        feature_cols,
        top_n=5
    )

    if not similar_players_df.empty:
        render_text_block("🧬 Similar Players", similar_players_df)

    st.divider()
    st.subheader("📈 Player Trend")

    trend_cols = [c for c in ["Season", "OPS", "HR", "wOBA", "breakout_score"] if c in player_df.columns]
    trend_df = player_df[trend_cols].copy()

    fig = px.line(
        trend_df,
        x="Season",
        y=[col for col in ["OPS", "breakout_score"] if col in trend_df.columns],
        markers=True,
        title=f"{selected_player} Performance Trend"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("🧠 Model Insights")

    importances = breakout_model.feature_importances_

    feature_importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(10).copy()

    feature_importance_df["Importance"] = feature_importance_df["Importance"].map(lambda x: f"{x:.4f}")

    render_text_block("Top 10 Most Important Features", feature_importance_df)

# ---------- Raw Data Tab ----------
with tab3:
    st.subheader("Full Player Dataset View")
    raw_players = sorted(filtered_df["Name"].dropna().unique())
    raw_player = st.selectbox("Choose a hitter for raw stats", raw_players, key="raw_player_select")
    raw_player_df = feat_df[feat_df["Name"] == raw_player].sort_values("Season").copy()

    st.code(raw_player_df.to_string(index=False), language=None)