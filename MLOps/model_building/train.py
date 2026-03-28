# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow

# Set MLflow tracking URI from environment variable
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("MLOps_CICD_experiment") # Use the same experiment name

api = HfApi(token=os.getenv("HF_TOKEN"))

data_repo_id = "ashikr/tourist_package_prediction"
Xtrain_path = f"hf://datasets/{data_repo_id}/Xtrain.csv"
Xtest_path = f"hf://datasets/{data_repo_id}/Xtest.csv"
ytrain_path = f"hf://datasets/{data_repo_id}/ytrain.csv"
ytest_path = f"hf://datasets/{data_repo_id}/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


# Define numeric and categorical features for the tourism dataset
numeric_features = [
    'Age', 'TypeofContact', 'CityTier', 'DurationOfPitch', 'Occupation',
    'NumberOfPersonVisiting', 'NumberOfFollowups', 'ProductPitched', 'PreferredPropertyStar',
    'MaritalStatus', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar',
    'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome'
]
categorical_features = [
    'Gender'
]

# Preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost Regressor
xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)

# Hyperparameter grid
param_grid = {
    'xgbregressor__n_estimators': [50, 100, 150],
    'xgbregressor__max_depth': [3, 5, 7],
    'xgbregressor__learning_rate': [0.01, 0.05, 0.1],
    'xgbregressor__subsample': [0.7, 0.8, 1.0],
    'xgbregressor__colsample_bytree': [0.7, 0.8, 1.0],
    'xgbregressor__reg_lambda': [0.1, 1, 10]
}

# Pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    # Grid Search
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(Xtrain, ytrain)

    # Log parameter sets
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_neg_mse", mean_score)

    # Best model
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    # Metrics
    train_rmse = mean_squared_error(ytrain, y_pred_train, squared=False)
    test_rmse = mean_squared_error(ytest, y_pred_test, squared=False)

    train_mae = mean_absolute_error(ytrain, y_pred_train)
    test_mae = mean_absolute_error(ytest, y_pred_test)

    train_r2 = r2_score(ytrain, y_pred_train)
    test_r2 = r2_score(ytest, y_pred_test)

    # Log metrics
    mlflow.log_metrics({
        "train_RMSE": train_rmse,
        "test_RMSE": test_rmse,
        "train_MAE": train_mae,
        "test_MAE": test_mae,
        "train_R2": train_r2,
        "test_R2": test_r2
    })

    # Save the model locally
    model_path = "best_tourist_package_prediction_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    model_repo_id = "ashikr/tourist_package_prediction_model" # Using a dedicated model repo
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=model_repo_id, repo_type=repo_type)
        print(f"Space '{model_repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{model_repo_id}' not found. Creating new space...")
        create_repo(repo_id=model_repo_id, repo_type=repo_type, private=False)
        print(f"Space '{model_repo_id}' created.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=os.path.basename(model_path),
        repo_id=model_repo_id,
        repo_type=repo_type,
    )
