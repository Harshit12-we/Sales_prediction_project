import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

DATA_FILE = "Advertising.csv"
MODEL_FILE = "sales_predictor.joblib"
RANDOM_STATE = 42

def load_dataframe(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def detect_features_target(df):
    possible_targets = [c for c in df.columns if c.lower() in ("sales", "y", "target")]
    target = possible_targets[0] if possible_targets else df.columns[-1]
    feature_candidates = [c for c in df.columns if c != target and df[c].dtype != "object"]
    return feature_candidates, target

def prepare_data(df, features, target):
    df = df.dropna(subset=features + [target]).reset_index(drop=True)
    X = df[features].astype(float)
    y = df[target].astype(float)
    return X, y

def train_models(X_train, y_train):
    models = {}
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models["LinearRegression"] = lr

    rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    models["RandomForest"] = rf

    ridge = Ridge(random_state=RANDOM_STATE)
    lasso = Lasso(random_state=RANDOM_STATE)
    gs = GridSearchCV(Pipeline([("scaler", StandardScaler()), ("clf", ridge)]),
                      {"clf__alpha": [0.1, 1.0, 10.0]}, cv=4, n_jobs=-1)
    gs.fit(X_train, y_train)
    models["Ridge"] = gs.best_estimator_.named_steps["clf"]

    gs2 = GridSearchCV(Pipeline([("scaler", StandardScaler()), ("clf", lasso)]),
                       {"clf__alpha": [0.001, 0.01, 0.1]}, cv=4, n_jobs=-1)
    gs2.fit(X_train, y_train)
    models["Lasso"] = gs2.best_estimator_.named_steps["clf"]

    return models

def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "pred": pred}

def plot_results(df, features, target, model, X_test, y_test, pred):
    plt.figure(figsize=(8,5))
    sns.regplot(x=y_test, y=pred, line_kws={"color":"red"})
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.show()

    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        idx = np.argsort(fi)[::-1]
        names = [features[i] for i in idx]
        plt.figure(figsize=(8,4))
        sns.barplot(x=fi[idx], y=names)
        plt.title("Feature importances")
        plt.tight_layout()
        plt.show()
    elif hasattr(model, "coef_"):
        coef = model.coef_
        plt.figure(figsize=(8,4))
        sns.barplot(x=coef, y=features)
        plt.title("Model coefficients")
        plt.tight_layout()
        plt.show()

def main():
    df = load_dataframe(DATA_FILE)
    features, target = detect_features_target(df)
    X, y = prepare_data(df, features, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    models = train_models(X_train, y_train)

    results = {}
    for name, m in models.items():
        res = evaluate(m, X_test, y_test)
        results[name] = res
        print(f"{name} -> MAE: {res['mae']:.4f}, RMSE: {res['rmse']:.4f}, R2: {res['r2']:.4f}")

    best_name = max(results.items(), key=lambda kv: kv[1]["r2"])[0]
    best_model = models[best_name]
    best_pred = results[best_name]["pred"]
    print("Best model:", best_name)

    joblib.dump({"model": best_model, "features": features}, MODEL_FILE)
    print("Saved model to", MODEL_FILE)

    plot_results(df, features, target, best_model, X_test, y_test, best_pred)

    sample = X_test.iloc[0:1]
    sample_pred = best_model.predict(sample)[0]
    print("Sample input:\n", sample.to_string(index=False))
    print("Predicted sales:", sample_pred)

if __name__ == "__main__":
    main()
