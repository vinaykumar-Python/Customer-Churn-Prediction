# train_model.py
import pandas as pd
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

CSV_PATH = "Telco-Customer-Churn.csv"
MODEL_PATH = "churn_best_model.pkl"

def load_and_clean_data(path=CSV_PATH):
    df = pd.read_csv(path)
    print("Shape before cleaning:", df.shape)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"]

    print("Shape after cleaning:", X.shape)
    print("Churn distribution:\n", y.value_counts(normalize=True))
    return X, y, df

def build_preprocessor(X):
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in X.select_dtypes(include=["int64", "float64"]).columns:
        if col not in numeric_features:
            numeric_features.append(col)

    categorical_features = [c for c in X.columns if c not in numeric_features]

    print("Numeric columns:", numeric_features)
    print("Categorical columns:", categorical_features)

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor

def plot_confusion_matrix(cm, labels, title, filename):
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def train_models():
    X, y, df = load_and_clean_data()
    preprocessor = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []
    best_auc = -1
    best_name = None
    best_pipe = None

    for name, clf in models.items():
        print(f"\nTraining {name} ...")
        pipe = Pipeline([("preprocessor", preprocessor), ("model", clf)])
        pipe.fit(X_train, y_train)

        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = pipe.predict(X_test)

        auc = roc_auc_score(y_test, y_prob)
        print(f"{name} ROC-AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, ["No", "Yes"],
                              f"{name} Confusion Matrix",
                              f"{name}_cm.png")

        results.append((name, auc))

        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_pipe = pipe

    print("\nModel AUC Scores:")
    for n, a in results:
        print(f"{n}: {a:.4f}")

    print(f"\nBest model: {best_name} (ROC-AUC = {best_auc:.4f})")
    joblib.dump(best_pipe, MODEL_PATH)
    print(f"Saved best model pipeline to {MODEL_PATH}")

if __name__ == "__main__":
    train_models()
