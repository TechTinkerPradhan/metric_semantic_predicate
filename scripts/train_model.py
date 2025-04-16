import argparse
import pandas as pd
import torch
import pyro
import mlflow
import yaml
import os
from torch.utils.tensorboard import SummaryWriter
import datetime

from metric_semantic_predicate.dataset.feature_utils import prepare_dataset, split_dataset
from metric_semantic_predicate.training.model_io import save_model
from metric_semantic_predicate.models.bayesian_combined_bnn import train_bnn_combined
from metric_semantic_predicate.training.training_utils import (
    get_training_device,
    get_feature_and_target_cols,
    prepare_tensor_data
)

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Train Bayesian Neural Network using YAML config")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--feature_config", type=str, default=None, help="Optional JSON file for features/targets")
    parser.add_argument("--task", type=str, default="combined", help="Task type: combined/metric/predicate/semantic")
    args = parser.parse_args()

    # Load YAML config
    cfg = load_config(args.config)
    dataset_path = cfg['training']['dataset']
    model_name = cfg['training']['model_name']
    num_steps = cfg['training']['num_steps']
    hidden_dim = cfg['training']['hidden_dim']
    lr = cfg['training']['lr']
    save_flag = cfg['training']['save_model']

    device = get_training_device()
    print(f"🚀 Using device: {device}")

    # Load and prepare data
    df = pd.read_csv(dataset_path)
    feature_cols, target_cols = get_feature_and_target_cols(config_path=args.feature_config, task=args.task)

    X, Y, df_encoded, _, scaler = prepare_dataset(df, feature_cols, target_cols, scaler=None, fit_scaler=True)

    df_train, df_val, df_test = split_dataset(df_encoded)

    X_train, Y_train, _, _, _ = prepare_dataset(df_train, feature_cols, target_cols, scaler=scaler, fit_scaler=False)
    X_val, Y_val, _, _, _ = prepare_dataset(df_val, feature_cols, target_cols, scaler=scaler, fit_scaler=False)
    X_test, Y_test, _, _, _ = prepare_dataset(df_test, feature_cols, target_cols, scaler=scaler, fit_scaler=False)


    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_t = torch.tensor(Y_train, dtype=torch.float32).to(device)

    # ✅ Add this to enable validation loss
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32).to(device)

    # TensorBoard logging
    tb_logdir = f"runs/{model_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=tb_logdir)

    # MLflow logging
    mlflow.set_experiment(cfg["experiment"]["name"])
    with mlflow.start_run(run_name=f"{model_name}_run"):
        mlflow.log_params({
            "dataset": dataset_path,
            "num_steps": num_steps,
            "hidden_dim": hidden_dim,
            "lr": lr,
            "device": str(device),
            "task": args.task
        })

        svi, net = train_bnn_combined(
            X=X_t,
            Y=Y_t,
            input_dim=len(feature_cols),
            hidden_dim=hidden_dim,
            num_steps=num_steps,
            lr=lr,
            device=device,
            model_name=model_name,
            writer=writer,
            X_val=X_val_t,
            Y_val=Y_val_t
        )

        writer.close()

        if save_flag:
            save_model(model_name)
            mlflow.log_artifact(f"data/models/{model_name}.pt")

        print("✅ Training complete and logged to MLflow & TensorBoard!")

if __name__ == "__main__":
    main()
