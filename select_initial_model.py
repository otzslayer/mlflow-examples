import os
import sys
from typing import Tuple
import warnings

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def load_data() -> Tuple[pd.DataFrame, np.ndarray]:
    # 데이터는 UCI에서 다운로드한 Wine quality 데이터입니다.
    wine_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "./data/wine-quality.csv"
    )
    data = pd.read_csv(wine_path)

    X = data.drop(["quality"], axis=1)
    y = data["quality"].values

    return X, y


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # 데이터는 UCI에서 다운로드한 Wine quality 데이터입니다.
    wine_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "./data/wine-quality.csv")
    data = pd.read_csv(wine_path)

    X, y = load_data()
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=42)

    # 단일 실험을 생성합니다. create_experiment 메서드에는 실험명을 입력합니다.
    experiment_id = mlflow.create_experiment("초도 모델링")
    
    # start_run 메서드를 통해 각각의 모델을 실행하여 로깅합니다.
    with mlflow.start_run(experiment_id=experiment_id):
        lr = ElasticNet(alpha=0.01, l1_ratio=0.01, random_state=42)
        lr.fit(X_train, y_train)

        predicted_qualities = lr.predict(X_valid)

        (rmse, mae, r2) = eval_metrics(y_valid, predicted_qualities)

        # set_tag 메서드는 해당 모델에 태그를 기록합니다. 여기에선 모델명을 입력합니다.
        mlflow.set_tag("model", "ElasticNet")
        # log_metric 메서드로 여러 메트릭을 기록합니다. 수치형 데이터만 넣을 수 있습니다.
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # ElasticNet은 Scikit-learn의 모듈이기 때문에
        # mlflow.sklearn.log_model 로 모델 아티팩트를 저장합니다.
        mlflow.sklearn.log_model(lr, "ElasticNet")

    with mlflow.start_run(experiment_id=experiment_id):
        rf = RandomForestRegressor(n_estimators=100,
                                   max_features="auto",
                                   random_state=42)
        rf.fit(X_train, y_train)

        predicted_qualities = rf.predict(X_valid)

        (rmse, mae, r2) = eval_metrics(y_valid, predicted_qualities)

        mlflow.set_tag("model", "RandomForestRegressor")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(rf, "RandomForestRegressor")

    with mlflow.start_run(experiment_id=experiment_id):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_valid, label=y_valid)
        param = {'objective': 'reg:squarederror'}
        xgb_bst = xgb.train(param, dtrain, num_boost_round=100)

        predicted_qualities = xgb_bst.predict(dtest)

        (rmse, mae, r2) = eval_metrics(y_valid, predicted_qualities)

        mlflow.set_tag("model", "XGBoost")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.xgboost.log_model(xgb_bst, "XGBoost")

