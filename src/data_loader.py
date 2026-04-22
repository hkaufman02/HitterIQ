import pandas as pd


def load_hitter_data(start_season: int = 2021, end_season: int = 2025) -> pd.DataFrame:
    """
    Load hitter data from a local CSV file instead of relying on live scraping.
    """
    df = pd.read_csv("data/hitter_data.csv")

    df = df[(df["Season"] >= start_season) & (df["Season"] <= end_season)].copy()

    numeric_cols = [col for col in df.columns if col not in ["Name", "Team"]]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Name", "Season", "OPS"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    hitters = load_hitter_data()
    print(hitters.head())
    print("Rows:", len(hitters))