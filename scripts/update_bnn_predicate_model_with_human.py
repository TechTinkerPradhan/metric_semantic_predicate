import argparse
import pandas as pd
import torch
import yaml
from metric_semantic_predicate.models.bnn_predicate_model import train_bnn_pred_model, bnn_predict_predicate, BayesianNNPred
from metric_semantic_predicate.training.evaluation_metrics import evaluate_with_distributions
from metric_semantic_predicate.training.model_io import load_model, save_model
from metric_semantic_predicate.dataset.feature_utils import prepare_dataset, split_dataset

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--task", type=str, default="predicate")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Using device: {device}")

    feature_cols = ["encoded_predicate", "width", "height", "depth"]
    target_cols = ["theta0", "kappa"]

    df_human = pd.read_csv(cfg['paths']['human_dataset'])
    df_h_train, df_h_val, df_h_test = split_dataset(df_human)

    X_h_train, Y_h_train, _, _, scaler = prepare_dataset(df_h_train, feature_cols, target_cols)
    X_h_val, Y_h_val, _, _, _ = prepare_dataset(df_h_val, feature_cols, target_cols, scaler, fit_scaler=False)
    X_h_test, Y_h_test, _, _, _ = prepare_dataset(df_h_test, feature_cols, target_cols, scaler, fit_scaler=False)

    X_h_train_t = torch.tensor(X_h_train, dtype=torch.float32).to(device)
    Y_h_train_t = torch.tensor(Y_h_train, dtype=torch.float32).to(device)
    X_h_test_t = torch.tensor(X_h_test, dtype=torch.float32).to(device)
    X_h_val_t = torch.tensor(X_h_val, dtype=torch.float32).to(device)
    Y_h_val_t = torch.tensor(Y_h_val, dtype=torch.float32).to(device)

    model_name = cfg['update']['pretrained_model']
    updated_name = cfg['update']['updated_model']

    net = BayesianNNPred(input_dim=X_h_train.shape[1], hidden_dim=cfg['training']['hidden_dim'], output_dim=Y_h_train.shape[1]).to(device)
    load_model(model_name)

    mean_old, _ = bnn_predict_predicate(X_h_test_t, net)

    svi, net = train_bnn_pred_model(X_h_train_t, Y_h_train_t, input_dim=X_h_train.shape[1],
                                output_dim=Y_h_train.shape[1], hidden_dim=cfg['training']['hidden_dim'],
                                num_steps=cfg['training']['num_steps'], lr=cfg['training']['lr'])

    save_model(updated_name)

    mean_updated, _ = bnn_predict_predicate(X_h_test_t, net)

    print("\n📊 Evaluation on HUMAN TEST SET:")
    evaluate_with_distributions(torch.tensor(Y_h_test), mean_old, mean_updated)

    df_orig = pd.read_csv(cfg['paths']['original_dataset'])
    _, _, df_orig_test = split_dataset(df_orig)
    X_orig_test, Y_orig_test, _, _, _ = prepare_dataset(df_orig_test, feature_cols, target_cols, scaler, fit_scaler=False)
    X_orig_test_t = torch.tensor(X_orig_test, dtype=torch.float32).to(device)

    net_before = BayesianNNPred(input_dim=X_orig_test.shape[1], hidden_dim=cfg['training']['hidden_dim'], output_dim=Y_orig_test.shape[1]).to(device)
    load_model(model_name)
    mean_orig_old, _ = bnn_predict_predicate(X_orig_test_t, net_before)
    mean_orig_updated, _ = bnn_predict_predicate(X_orig_test_t, net)

    print("\n📊 Evaluation on ORIGINAL TEST SET:")
    evaluate_with_distributions(torch.tensor(Y_orig_test), mean_orig_old, mean_orig_updated)

if __name__ == "__main__":
    main()
