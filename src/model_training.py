import os
import pandas as pd
import numpy as np
import joblib
from datasets import load_dataset
from huggingface_hub import login, create_repo, upload_file
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = "DeeptaV/SuperKart-dataset"
HF_MODEL_REPO = "DeeptaV/SuperKart-sales-model"

login(token=HF_TOKEN)

hf_data = load_dataset(HF_DATASET_REPO)
train_df = hf_data["train"].to_pandas()
test_df = hf_data["test"].to_pandas()

train_df.drop(columns=["__index_level_0__"], inplace=True, errors="ignore")
test_df.drop(columns=["__index_level_0__"], inplace=True, errors="ignore")

target_col = "Product_Store_Sales_Total"

X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]

X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X_train.select_dtypes(exclude=["object"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

models = {
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "Bagging": BaggingRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(random_state=42),
    "AdaBoost": AdaBoostRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, objective="reg:squarederror")
}

param_grids = {
    "DecisionTree": {
        "model__max_depth": [5, 10, None]
    },
    "Bagging": {
        "model__n_estimators": [50, 100]
    },
    "RandomForest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [10, None]
    },
    "AdaBoost": {
        "model__n_estimators": [50, 100],
        "model__learning_rate": [0.05, 0.1]
    },
    "GradientBoosting": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1]
    },
    "XGBoost": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 5]
    }
}

results = []
best_estimators = {}

for model_name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    grid_search = GridSearchCV(
        pipeline,
        param_grids[model_name],
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_estimators[model_name] = best_model

    y_pred = best_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": model_name,
        "Best_Params": str(grid_search.best_params_),
        "RMSE": rmse,
        "MAE": mae,
        "R2_Score": r2
    })

results_df = pd.DataFrame(results).sort_values(by="RMSE").reset_index(drop=True)
best_model_name = results_df.loc[0, "Model"]
best_model = best_estimators[best_model_name]

os.makedirs("artifacts", exist_ok=True)

model_path = f"artifacts/{best_model_name}_best_model.pkl"
metrics_path = "artifacts/model_comparison_results.csv"

joblib.dump(best_model, model_path)
results_df.to_csv(metrics_path, index=False)

create_repo(repo_id=HF_MODEL_REPO, repo_type="model", exist_ok=True)

upload_file(
    path_or_fileobj=model_path,
    path_in_repo=os.path.basename(model_path),
    repo_id=HF_MODEL_REPO,
    repo_type="model"
)

upload_file(
    path_or_fileobj=metrics_path,
    path_in_repo=os.path.basename(metrics_path),
    repo_id=HF_MODEL_REPO,
    repo_type="model"
)

print("Model training and registration completed successfully.")