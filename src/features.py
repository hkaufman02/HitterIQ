import pandas as pd
import numpy as np


def build_hitter_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = data.sort_values(["Name", "Season"]).reset_index(drop=True)

    base_cols = [
        "PA", "HR", "RBI", "SB", "BB%", "K%", "ISO", "BABIP",
        "AVG", "OBP", "SLG", "OPS", "wOBA", "OPS+"
    ]

    for col in base_cols:
        if col in data.columns:
            data[f"prev_{col}"] = data.groupby("Name")[col].shift(1)

    for col in ["HR", "OPS", "wOBA", "BB%", "K%", "ISO", "RBI"]:
        if col in data.columns:
            data[f"rolling2_{col}"] = (
                data.groupby("Name")[col]
                .transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean())
            )

    for col in ["HR", "OPS", "wOBA", "OPS+", "BB%", "K%", "ISO"]:
        if col in data.columns and f"prev_{col}" in data.columns:
            data[f"delta_{col}"] = data[col] - data[f"prev_{col}"]

    if "HR" in data.columns and "PA" in data.columns:
        data["HR_per_PA"] = np.where(data["PA"] > 0, data["HR"] / data["PA"], 0)

    if "RBI" in data.columns and "PA" in data.columns:
        data["RBI_per_PA"] = np.where(data["PA"] > 0, data["RBI"] / data["PA"], 0)

    if "SB" in data.columns and "PA" in data.columns:
        data["SB_per_PA"] = np.where(data["PA"] > 0, data["SB"] / data["PA"], 0)

    if all(col in data.columns for col in ["ISO", "SLG", "HR_per_PA"]):
        data["power_score"] = (
            data["ISO"].fillna(0) * 0.45 +
            data["SLG"].fillna(0) * 0.35 +
            data["HR_per_PA"].fillna(0) * 8.0
        )

    if all(col in data.columns for col in ["BB%", "K%", "OBP"]):
        data["discipline_score"] = (
            data["BB%"].fillna(0) * 1.2 -
            data["K%"].fillna(0) * 0.8 +
            data["OBP"].fillna(0)
        )

    if all(col in data.columns for col in ["OPS", "wOBA", "RBI_per_PA"]):
        data["production_score"] = (
            data["OPS"].fillna(0) * 0.5 +
            data["wOBA"].fillna(0) * 0.4 +
            data["RBI_per_PA"].fillna(0) * 3.0
        )

    required_prev = [c for c in ["prev_OPS", "prev_HR"] if c in data.columns]
    if required_prev:
        data = data.dropna(subset=required_prev)

    return data.reset_index(drop=True)