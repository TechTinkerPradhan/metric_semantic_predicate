import argparse
import pandas as pd
import torch
import yaml
import mlflow
import pyro
from torch.utils.tensorboard import SummaryWriter

from metric_semantic_predicate.dataset.feature_utils import prepare_dataset, split_dataset
from metric_semantic_predicate.models.bnn_metric_model import train_bnn_model_metric
from metric_semantic_predicate.training.model_io import save_model


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Train Metric BNN")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task", type=str, default="metric")
    args = parser.parse_args()

    # Clear previous param store
    pyro.clear_param_store()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    df = pd.read_csv(cfg["training"]["dataset"])
    feature_cols = ["metric", "width", "height", "depth"]
    target_cols = ["d0", "sigma_m"]

    X, Y, df_encoded, _, _ = prepare_dataset(df, feature_cols, target_cols)
    df_train, df_val, df_test = split_dataset(df_encoded)
    X_train, Y_train, _, _, _ = prepare_dataset(df_train, feature_cols, target_cols)

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_t = torch.tensor(Y_train, dtype=torch.float32).to(device)

    model_name = cfg["training"]["model_name"]
    writer = SummaryWriter(log_dir=f"runs/{model_name}")

    mlflow.set_experiment(cfg["experiment"]["name"])
    with mlflow.start_run(run_name=f"{model_name}_run"):
        mlflow.log_params({
            "dataset": cfg["training"]["dataset"],
            "num_steps": cfg["training"]["num_steps"],
            "hidden_dim": cfg["training"]["hidden_dim"],
            "lr": cfg["training"]["lr"],
            "device": str(device),
            "task": args.task
        })

        svi, net = train_bnn_model_metric(
            X_t, Y_t,
            input_dim=X_t.shape[1],
            output_dim=Y_t.shape[1],
            hidden_dim=cfg["training"]["hidden_dim"],
            num_steps=cfg["training"]["num_steps"],
            lr=cfg["training"]["lr"],
            device=device,
            model_name=model_name,
            writer=writer
        )

        if cfg["training"]["save_model"]:
            save_model(model_name)
            print(f"✅ Saved model as {model_name}.pt")
            mlflow.log_artifact(f"data/models/{model_name}.pt")

        writer.close()

if __name__ == "__main__":
    main()
