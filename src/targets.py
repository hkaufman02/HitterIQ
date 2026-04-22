import pandas as pd
import numpy as np


def create_breakout_target(df: pd.DataFrame, ops_jump: float = 0.040) -> pd.DataFrame:
    data = df.copy()
    data["breakout"] = np.where((data["OPS"] - data["prev_OPS"]) >= ops_jump, 1, 0)
    return data


def create_breakout_score(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    ops_gain = (data["OPS"] - data["prev_OPS"]).fillna(0) if "prev_OPS" in data.columns else 0
    hr_gain = (data["HR"] - data["prev_HR"]).fillna(0) if "prev_HR" in data.columns else 0
    bb_gain = (data["BB%"] - data["prev_BB%"]).fillna(0) if "prev_BB%" in data.columns else 0
    k_improve = (data["prev_K%"] - data["K%"]).fillna(0) if "prev_K%" in data.columns else 0

    data["breakout_score"] = (
        ops_gain * 100 +
        hr_gain * 0.6 +
        bb_gain * 20 +
        k_improve * 15
    )

    return data