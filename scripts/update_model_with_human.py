import pandas as pd
import torch
from metric_semantic_predicate.utils.data_preparation import prepare_dataset, split_dataset
from metric_semantic_predicate.utils.model_io import load_model, save_model
from metric_semantic_predicate.models.bayesian_combined_bnn import train_bnn_combined, bnn_predict, BayesianNNcombined
from metric_semantic_predicate.utils.evaluation import evaluate_with_distributions
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load human dataset and split
df_human = pd.read_csv("data/3DSceneGraph_Beechwood_dataset_human.csv")
df_h_train, df_h_val, df_h_test = split_dataset(df_human)

feature_cols = ["metric", "encoded_predicate", "encoded_semantic", "width", "height", "depth"]
target_cols = ["P_combined"]

# Prepare train/test from human dataset
X_h_train, Y_h_train, _, _ = prepare_dataset(df_h_train, feature_cols, target_cols)
X_h_test, Y_h_test, _, _ = prepare_dataset(df_h_test, feature_cols, target_cols)

X_h_train_t = torch.tensor(X_h_train, dtype=torch.float32).to(device)
Y_h_train_t = torch.tensor(Y_h_train, dtype=torch.float32).to(device)
X_h_test_t = torch.tensor(X_h_test, dtype=torch.float32).to(device)

# Load model trained on original dataset
net = BayesianNNcombined(input_dim=len(feature_cols)).to(device)
load_model("bnn_before_update")

# Predict before update
mean_old_human, _ = bnn_predict(X_h_test_t, net)

# Bayesian update on human dataset train split
svi, net = train_bnn_combined(X_h_train_t, Y_h_train_t, input_dim=len(feature_cols),
                              num_steps=2000, device=device, model_name="bnn_after_update")

# Save the updated model
save_model("bnn_after_update")

# Predict after update
mean_updated_human, _ = bnn_predict(X_h_test_t, net)

# Also test both models on original dataset test
df_orig = pd.read_csv("data/3DSceneGraph_Beechwood_dataset.csv")
_, _, df_orig_test = split_dataset(df_orig)
X_orig_test, Y_orig_test, _, _ = prepare_dataset(df_orig_test, feature_cols, target_cols)
X_orig_test_t = torch.tensor(X_orig_test, dtype=torch.float32).to(device)

# Load fresh copy of pre-update model
net_before = BayesianNNcombined(input_dim=len(feature_cols)).to(device)
load_model("bnn_before_update")
mean_old_orig, _ = bnn_predict(X_orig_test_t, net_before)

# Evaluate updated model on original test set
mean_updated_orig, _ = bnn_predict(X_orig_test_t, net)

# Evaluate using KDE + Wasserstein
print("\n📊 Evaluation on HUMAN TEST SET:")
evaluate_with_distributions(torch.tensor(Y_h_test), mean_old_human, mean_updated_human)

print("\n📊 Evaluation on ORIGINAL TEST SET:")
evaluate_with_distributions(torch.tensor(Y_orig_test), mean_old_orig, mean_updated_orig)
