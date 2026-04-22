import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from data_loader import load_hitter_data
from features import build_hitter_features
from targets import create_breakout_target


def train_breakout_model():
    df = load_hitter_data(2021, 2025)
    df = build_hitter_features(df)
    df = create_breakout_target(df)

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

    X = df[feature_cols].copy().fillna(0)
    y = df["breakout"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=6,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", round(accuracy_score(y_test, preds), 4))
    print("ROC-AUC:", round(roc_auc_score(y_test, probs), 4))
    print(classification_report(y_test, preds))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/breakout_model.pkl")
    joblib.dump(feature_cols, "models/feature_cols.pkl")

    return model, feature_cols, df


if __name__ == "__main__":
    train_breakout_model()