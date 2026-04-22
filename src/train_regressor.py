import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

from data_loader import load_hitter_data
from features import build_hitter_features


def train_ops_regressor():
    df = load_hitter_data(2021, 2025)
    df = build_hitter_features(df)

    feature_cols = [
        col for col in df.columns
        if (
            col.startswith("prev_") or
            col.startswith("delta_") or
            col.startswith("rolling2_") or
            col in [
                "Age", "HR_per_PA", "RBI_per_PA", "SB_per_PA",
                "power_score", "discipline_score", "production_score"
            ]
        )
    ]

    X = df[feature_cols].fillna(0)
    y = df["OPS"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=7,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("MAE:", round(mean_absolute_error(y_test, preds), 4))
    print("RMSE:", round(mean_squared_error(y_test, preds) ** 0.5, 4))
    print("R2:", round(r2_score(y_test, preds), 4))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/ops_regressor.pkl")

    return model, feature_cols, df


if __name__ == "__main__":
    train_ops_regressor()