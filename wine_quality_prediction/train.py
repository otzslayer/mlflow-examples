import os
import sys
from typing import Tuple
import warnings

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def load_data() -> Tuple[pd.DataFrame, np.ndarray]:
    wine_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../data/wine-quality.csv"
    )
    data = pd.read_csv(wine_path)

    X = data.drop(["quality"], axis=1)
    y = data["quality"].values

    return X, y

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    X, y = load_data()
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=42)

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(X_train, y_train)

        predicted_qualities = lr.predict(X_valid)

        (rmse, mae, r2) = eval_metrics(y_valid, predicted_qualities)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(lr, "ElasticNet")