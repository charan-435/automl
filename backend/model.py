from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pandas as pd


def train_model(df, target):
    if target not in df.columns:
        raise ValueError("Target column not found")
    X = df.drop(columns=[target])
    y = df[target]

    X = pd.get_dummies(X)
    columns = X.columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=42)
    }

    best_model = None
    best_score = -1
    scores = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = r2_score(y_test, preds)

        scores[name] = score

        if score > best_score:
            best_score = score
            best_model = model

    return best_model, scores, columns


def predict_model(model, data, columns):
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)

    df = df.reindex(columns=columns, fill_value=0)

    return model.predict(df).tolist()