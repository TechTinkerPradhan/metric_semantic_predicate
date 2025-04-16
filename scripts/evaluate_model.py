import argparse
import pandas as pd
import torch
import numpy as np
import mlflow

from metric_semantic_predicate.utils.data_preparation import prepare_dataset
from metric_semantic_predicate.utils.model_io import load_model
from metric_semantic_predicate.models.bayesian_combined_bnn import BayesianNNcombined, bnn_predict


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Bayesian Neural Network model")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="combined_pdf_bnn")
    parser.add_argument("--hidden_dim", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Evaluating on device: {device}")

    df = pd.read_csv(args.dataset)
    feature_cols = ["metric", "encoded_predicate", "encoded_semantic", "width", "height", "depth"]
    target_cols = ["P_combined"]

    X, Y, _, _ = prepare_dataset(df, feature_cols, target_cols)
    X_t = torch.tensor(X, dtype=torch.float32).to(device)

    net = BayesianNNcombined(input_dim=len(feature_cols), hidden_dim=args.hidden_dim).to(device)
    load_model(args.model_name)

    mean, std = bnn_predict(X_t, net)
    mse = np.mean((mean - Y.squeeze()) ** 2)

    mlflow.set_experiment("CombinedPDF_BNN_Eval")
    with mlflow.start_run(run_name=f"{args.model_name}_eval"):
        mlflow.log_param("eval_dataset", args.dataset)
        mlflow.log_param("hidden_dim", args.hidden_dim)
        mlflow.log_metric("mse", mse)

    print(f"📊 MSE on {args.dataset}: {mse:.4f}")


if __name__ == "__main__":
    main()
