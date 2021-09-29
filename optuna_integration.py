import os
from typing import Tuple

import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
from optuna.integration.mlflow import MLflowCallback


def load_data() -> Tuple[pd.DataFrame, np.ndarray]:
    # 데이터는 UCI에서 다운로드한 Wine quality 데이터입니다.
    wine_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "./data/wine-quality.csv"
    )
    data = pd.read_csv(wine_path)

    X = data.drop(["quality"], axis=1)
    y = data["quality"].values

    return X, y


class Objective(object):
    def __init__(self, data):
        self.data = data

    def __call__(self, trial):
        data = self.data

        num_leaves = trial.suggest_int("num_leaves", 7, 127)
        feature_fraction = trial.suggest_float("feature_fraction", 0.75, 1, step=.05)
        lambda_l2 = trial.suggest_float("lambda_l2", 1e-16, 1e-5, log=True)

        params = {
            "objective": "regression",
            "learning_rate": 0.1,
            "random_seed": 42,
            "num_leaves": num_leaves,
            "feature_fraction": feature_fraction,
            "lambda_l2": lambda_l2,
            "verbose": -1,
        }

        cv_result = lgb.cv(
            params=params,
            train_set=data,
            num_boost_round=1000,
            nfold=5,
            stratified=False,
            metrics="rmse",
            early_stopping_rounds=20,
            verbose_eval=False,
        )

        rmse = np.min(cv_result["rmse-mean"])

        return rmse


def make_mlflow_callback():
    cb = MLflowCallback(tracking_uri="mlruns", metric_name="RMSE")
    return cb


if __name__ == "__main__":
    X, y = load_data()

    dtrain = lgb.Dataset(X, label=y)

    study = optuna.create_study(
        study_name="MLflow - Optuna integration",
        direction="minimize",
        pruner=optuna.pruners.HyperbandPruner(max_resource="auto"),
    )

    objective = Objective(dtrain)
    mlflow_callback = make_mlflow_callback()

    study.optimize(
        objective, n_trials=30, callbacks=[mlflow_callback], show_progress_bar=True
    )