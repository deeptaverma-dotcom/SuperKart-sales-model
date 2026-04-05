import pandas as pd

results_df = pd.read_csv("artifacts/model_comparison_results.csv")

print("Model evaluation summary:")
print(results_df)

best_model = results_df.sort_values(by="RMSE").iloc[0]
print("\nBest model details:")
print(best_model)