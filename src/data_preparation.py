import os
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from huggingface_hub import login

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set.It is missing. Add it in GitHub Secrets")

HF_DATASET_REPO = "DeeptaV/SuperKart-dataset"

login(token=HF_TOKEN)

hf_data = load_dataset(HF_DATASET_REPO)
df = hf_data["train"].to_pandas()

# Remove unwanted index column if present
df.drop(columns=["__index_level_0__"], inplace=True, errors="ignore")

# Basic cleaning
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str).str.strip()
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    else:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

target_col = "Product_Store_Sales_Total"

X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_df = X_train.copy()
train_df[target_col] = y_train

test_df = X_test.copy()
test_df[target_col] = y_test

train_df.to_csv("train_superkart.csv", index=False)
test_df.to_csv("test_superkart.csv", index=False)

train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

dataset_dict = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

dataset_dict.push_to_hub(HF_DATASET_REPO)

print("Data preparation completed successfully.")