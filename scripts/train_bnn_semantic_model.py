import argparse
import pandas as pd
import torch
import yaml

from metric_semantic_predicate.dataset.feature_utils import prepare_dataset, split_dataset
from metric_semantic_predicate.training.model_io import save_model
from metric_semantic_predicate.models.bnn_semantic_model import train_bnn_sem_model, bnn_predict_sem, BayesianNNSem
from torch.utils.tensorboard import SummaryWriter
import os

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Train Semantic BNN using config file")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--task", type=str, default="semantic", help="Task type")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Using device: {device}")

    df = pd.read_csv(cfg["training"]["dataset"])
    df_train, df_val, df_test = split_dataset(df)

    feature_cols = cfg["data"]["feature_cols"]
    target_cols = cfg["data"]["target_cols"]

    X_train, Y_train, _, _, scaler = prepare_dataset(df_train, feature_cols, target_cols)
    X_val, Y_val, _, _, _ = prepare_dataset(df_val, feature_cols, target_cols, scaler, fit_scaler=False)
    X_test, Y_test, _, _, _ = prepare_dataset(df_test, feature_cols, target_cols, scaler, fit_scaler=False)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32).to(device)

    writer = SummaryWriter(log_dir=f"runs/{cfg['training']['model_name']}_semantic")

    input_dim = len(feature_cols)
    output_dim = len(target_cols)

    svi, net = train_bnn_sem_model(X_train_t, Y_train_t,
                             input_dim=input_dim,
                             output_dim=output_dim,
                             hidden_dim=cfg["training"]["hidden_dim"],
                             num_steps=cfg["training"]["num_steps"],
                             lr=cfg["training"]["lr"],
                             writer=writer,
                             X_val=X_val_t,
                             Y_val=Y_val_t)

    save_model(cfg["training"]["model_name"])
    print("✅ Semantic model training complete.")

if __name__ == "__main__":
    main()
