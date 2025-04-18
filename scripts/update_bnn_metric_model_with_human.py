import argparse
import pandas as pd
import torch
import numpy as np
import yaml
import pyro

from metric_semantic_predicate.dataset.feature_utils import prepare_dataset, split_dataset
from metric_semantic_predicate.training.model_io import load_model, save_model
from metric_semantic_predicate.models.bnn_metric_model import train_bnn_model_metric, bnn_predict_metric, BayesianNN_metric, guide_bnn_metric
from metric_semantic_predicate.training.evaluation_metrics import evaluate_with_distributions


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Update metric BNN model using human data")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--task", type=str, default="metric", help="Task type")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Using device: {device}")

    # Clear previous Pyro parameter store
    pyro.clear_param_store()

    model_name_before = cfg["update"]["model_before"]
    model_name_after = cfg["update"]["model_after"]

    # --- Load and prepare human dataset ---
    df_human = pd.read_csv(cfg["update"]["human_dataset"])
    df_h_train, df_h_val, df_h_test = split_dataset(df_human)

    feature_cols = cfg["data"]["feature_cols"]
    target_cols = cfg["data"]["target_cols"]

    X_h_train, Y_h_train, _, _, scaler = prepare_dataset(df_h_train, feature_cols, target_cols)
    X_h_val, Y_h_val, _, _, _ = prepare_dataset(df_h_val, feature_cols, target_cols, scaler, False)
    X_h_test, Y_h_test, _, _, _ = prepare_dataset(df_h_test, feature_cols, target_cols, scaler, False)

    X_h_train_t = torch.tensor(X_h_train, dtype=torch.float32).to(device)
    Y_h_train_t = torch.tensor(Y_h_train, dtype=torch.float32).to(device)
    X_h_val_t = torch.tensor(X_h_val, dtype=torch.float32).to(device)
    X_h_test_t = torch.tensor(X_h_test, dtype=torch.float32).to(device)

    # --- Load model trained on original dataset ---
    input_dim = len(feature_cols)
    output_dim = len(target_cols)
    net = BayesianNN_metric(input_dim=input_dim, hidden_dim=cfg["training"]["hidden_dim"], output_dim=output_dim).to(device)
    load_model(model_name_before)

    # --- Predict on human test set before update ---
    mean_old_human, _ = bnn_predict_metric(X_h_test_t, net)

    # --- Update using human train split ---
    svi, net = train_bnn_model_metric(
        X_h_train_t,
        Y_h_train_t,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=cfg["training"]["hidden_dim"],
        num_steps=cfg["training"]["num_steps"],
        lr=cfg["training"]["lr"],
        device=device,
        model_name=model_name_after
    )

    # Save and reload updated model to refresh pyro param store
    save_model(model_name_after)
    print(f"✅ Saved updated model as {model_name_after}.pt")

    # Clear store and re-load to register parameters
    pyro.clear_param_store()
    load_model(model_name_after)
    _ = guide_bnn_metric(
        torch.zeros(1, input_dim).to(device),
        torch.zeros(1, output_dim).to(device),
        net
    )


    # --- Predict after update ---
    mean_updated_human, _ = bnn_predict_metric(X_h_test_t, net)

    # --- Evaluate both on original test set ---
    df_orig = pd.read_csv(cfg["training"]["dataset"])
    _, _, df_orig_test = split_dataset(df_orig)
    X_orig_test, Y_orig_test, _, _, _ = prepare_dataset(df_orig_test, feature_cols, target_cols, scaler, False)
    X_orig_test_t = torch.tensor(X_orig_test, dtype=torch.float32).to(device)

    # Predict pre-update
    net_pre = BayesianNN_metric(input_dim=input_dim, hidden_dim=cfg["training"]["hidden_dim"], output_dim=output_dim).to(device)
    load_model(model_name_before)
    mean_old_orig, _ = bnn_predict_metric(X_orig_test_t, net_pre)

    # Predict post-update
    mean_updated_orig, _ = bnn_predict_metric(X_orig_test_t, net)

    # Ensure targets are torch tensors for evaluation
    Y_h_test = torch.tensor(Y_h_test, dtype=torch.float32)
    Y_orig_test = torch.tensor(Y_orig_test, dtype=torch.float32)

    # --- Evaluate ---
    print("\n📊 Evaluation on HUMAN TEST SET:")
    evaluate_with_distributions(Y_h_test, mean_old_human, mean_updated_human, label_prefix="MetricBNN")

    print("\n📊 Evaluation on ORIGINAL TEST SET:")
    evaluate_with_distributions(Y_orig_test, mean_old_orig, mean_updated_orig, label_prefix="MetricBNN")


if __name__ == "__main__":
    main()
