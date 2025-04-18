import argparse
import pandas as pd
import torch
import mlflow
import yaml
from torch.utils.tensorboard import SummaryWriter
import datetime

from metric_semantic_predicate.training.model_io import save_model
from metric_semantic_predicate.dataset.feature_utils import prepare_dataset, split_dataset
from metric_semantic_predicate.models.bnn_predicate_model import train_bnn_pred_model

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task", type=str, default="predicate")
    args = parser.parse_args()

    cfg = load_config(args.config)

    dataset_path = cfg['training']['dataset']
    model_name = cfg['training']['model_name']
    num_steps = cfg['training']['num_steps']
    hidden_dim = cfg['training']['hidden_dim']
    lr = cfg['training']['lr']
    save_flag = cfg['training']['save_model']
    feature_cols = cfg['data']['feature_cols']
    target_cols = cfg['data']['target_cols']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Using device: {device}")

    df = pd.read_csv(dataset_path)
    X, Y, df_encoded, _, _ = prepare_dataset(df, feature_cols, target_cols)
    df_train, df_val, df_test = split_dataset(df_encoded)
    X_train, Y_train, _, _, scaler = prepare_dataset(df_train, feature_cols, target_cols)
    X_val, Y_val, _, _, _ = prepare_dataset(df_val, feature_cols, target_cols, scaler, fit_scaler=False)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32).to(device)

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

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_dir=f"runs/{model_name}_{timestamp}")

        input_dim = X_train.shape[1]
        output_dim = Y_train.shape[1]

        svi, net = train_bnn_pred_model(
            X_train_t, Y_train_t,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_steps=num_steps,
            lr=lr,
            writer=writer,
            X_val=X_val_t,
            Y_val=Y_val_t,
        )

        if save_flag:
            save_model(model_name)
            mlflow.log_artifact(f"data/models/{model_name}.pt")

        writer.close()
        print("\n✅ Training complete and logged to MLflow")

if __name__ == "__main__":
    main()
