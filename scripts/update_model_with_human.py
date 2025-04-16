import argparse
import os
import datetime
import pandas as pd
import torch
import pyro
import mlflow
from torch.utils.tensorboard import SummaryWriter

from metric_semantic_predicate.dataset.feature_utils import prepare_dataset, split_dataset
from metric_semantic_predicate.training.model_io import load_model, save_model
from metric_semantic_predicate.models.bayesian_combined_bnn import (
    train_bnn_combined,
    bnn_predict,
    BayesianNNcombined,
)
from metric_semantic_predicate.training.evaluation_metrics import evaluate_with_distributions
import yaml

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Update BNN with human-annotated data")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task", type=str, default="combined")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    feature_cols = cfg["features"]
    target_cols = cfg["targets"]
    num_steps = cfg["training"]["num_steps"]
    model_name = cfg["training"]["model_name"]

    tb_logdir = f"runs/{model_name}_update_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=tb_logdir)

    # Load human dataset
    df_human = pd.read_csv(cfg["update"]["human_dataset"])
    df_h_train, df_h_val, df_h_test = split_dataset(df_human)

    # Prepare tensors
    X_h_train, Y_h_train, _,_,_ = prepare_dataset(df_h_train, feature_cols, target_cols)
    X_h_val, Y_h_val, _,_,_= prepare_dataset(df_h_val, feature_cols, target_cols)
    X_h_test, Y_h_test, _,_,_ = prepare_dataset(df_h_test, feature_cols, target_cols)

    X_h_train_t = torch.tensor(X_h_train, dtype=torch.float32).to(device)
    Y_h_train_t = torch.tensor(Y_h_train, dtype=torch.float32).to(device)
    X_h_val_t = torch.tensor(X_h_val, dtype=torch.float32).to(device)
    Y_h_val_t = torch.tensor(Y_h_val, dtype=torch.float32).to(device)
    X_h_test_t = torch.tensor(X_h_test, dtype=torch.float32).to(device)

    # Load original model and predict before update
    net = BayesianNNcombined(input_dim=len(feature_cols)).to(device)
    load_model("combined_bnn_before_update")
    mean_old_human, _ = bnn_predict(X_h_test_t, net)

    mlflow.set_experiment(cfg["experiment"]["name"])
    with mlflow.start_run(run_name="update_with_human"):
        mlflow.log_params({
            "update_steps": num_steps,
            "device": str(device),
            "task": args.task,
        })

        # Perform Bayesian update with human data
        svi, net = train_bnn_combined(
            X_h_train_t, Y_h_train_t,
            input_dim=len(feature_cols),
            hidden_dim=cfg["training"]["hidden_dim"],
            num_steps=num_steps,
            lr=cfg["training"]["lr"],
            device=device,
            model_name="bnn_after_update",
            writer=writer,
            X_val=X_h_val_t,
            Y_val=Y_h_val_t
        )

        # Save updated model
        save_model("combined_bnn_after_update")
        mlflow.log_artifact("data/models/combined_bnn_before_update.pt")

    writer.close()

    # Predict with updated model
    mean_updated_human, _ = bnn_predict(X_h_test_t, net)

    # Evaluate on original dataset too
    df_orig = pd.read_csv(cfg["training"]["dataset"])
    _, _, df_orig_test = split_dataset(df_orig)
    X_orig_test, Y_orig_test,_,_, _ = prepare_dataset(df_orig_test, feature_cols, target_cols)
    X_orig_test_t = torch.tensor(X_orig_test, dtype=torch.float32).to(device)

    net_before = BayesianNNcombined(input_dim=len(feature_cols)).to(device)
    load_model("combined_bnn_before_update")
    mean_old_orig, _ = bnn_predict(X_orig_test_t, net_before)

    mean_updated_orig, _ = bnn_predict(X_orig_test_t, net)

    print("\n📊 Evaluation on HUMAN TEST SET:")
    evaluate_with_distributions(torch.tensor(Y_h_test), mean_old_human, mean_updated_human)

    print("\n📊 Evaluation on ORIGINAL TEST SET:")
    evaluate_with_distributions(torch.tensor(Y_orig_test), mean_old_orig, mean_updated_orig)

if __name__ == "__main__":
    main()
