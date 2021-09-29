import mlflow

project_uri = "https://github.com/mlflow/mlflow-example"
params = {"alpha": 0.5, "l1_ratio": 0.01}

# Run MLflow project and create a reproducible conda environment
# on a local host
mlflow.run(project_uri, parameters=params)